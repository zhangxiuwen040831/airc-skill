from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .chunker import chunk_text
from .config import DEFAULT_CONFIG, RewriteMode
from .rewriter import RewriteStats, split_sentences

_NORMALIZE_RE = re.compile(r"\s+")
_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_IMAGE_LINE_RE = re.compile(r"^\s*!\[[^\]]*]\([^)]+\)\s*$")
_IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]+\)")
_LINK_DEF_RE = re.compile(r"^\s*\[[^\]]+\]:\s+\S+")
_REFERENCE_LINE_RE = re.compile(r"^\s*\[\d+\]")
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$|^\s*[:\-\|\s]+\s*$")
_FORMULA_LINE_RE = re.compile(r"^\s*(?:\$\$|\\\[|\\\(|\\begin\{|\\end\{)")
_CAPTION_LINE_RE = re.compile(
    r"^\s*(?:"
    r"(?:图|表)\s*\d+(?:-\d+)?[^\n]*"
    r"|(?:Figure|Table)\s*\d+[^\n]*"
    r")\s*$"
)
_PATH_OR_CHECKPOINT_RE = re.compile(
    r"(?:[A-Za-z]:\\[^\s<>()]+|(?:[\w.-]+/)+[\w./-]+|checkpoint[s]?/[^\s<>()]+|[\w.-]+\.(?:pth|pt|ckpt|png|jpg|jpeg|pdf))"
)
_CITATION_RE = re.compile(
    r"\[\d+\](?:\[\d+\])*|\([A-Z][A-Za-z\-]+,\s*\d{4}[a-z]?\)|（[A-Z][A-Za-z\-]+,\s*\d{4}[a-z]?）"
)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_TECH_TOKEN_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_.\-/]{2,}|checkpoint[s]?/[^\s<>()]+|[\w.-]+\.(?:pth|pt|ckpt)|\[\d+\]"
)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LEADING_IMPLICATION_RE = re.compile(r"^(?:因此|由此|这也意味着|基于此|在这种情况下|正因为如此|在此基础上)[，,\s]*")
_META_SUBJECTS = ("本研究", "本文", "该研究", "该系统")
_STIFF_OPENERS = ("本研究", "本文", "从", "对", "这也意味着", "基于此", "在这种情况下", "总体来看", "综合来看")


@dataclass(frozen=True)
class BodyInventory:
    body_block_ids: list[int]
    body_blocks_total: int
    body_sentences_total: int
    body_characters: int
    document_scale: str


@dataclass(frozen=True)
class BodyRewriteMetrics:
    body_block_ids: list[int]
    body_blocks_total: int
    body_sentences_total: int
    body_characters: int
    body_changed_blocks: int
    body_changed_sentences: int
    body_changed_block_ratio: float
    body_rewrite_coverage: float
    body_discourse_change_score: int
    body_cluster_rewrite_score: int
    body_developmental_blocks_total: int
    body_developmental_changed_blocks: int
    document_scale: str
    rewrite_quota_met: bool
    rewrite_quota_reason_codes: list[str]
    required_body_rewrite_coverage: float
    required_body_changed_block_ratio: float
    required_body_discourse_change_score: int
    required_body_cluster_rewrite_score: int
    body_original_text: str
    body_revised_text: str


def build_body_inventory(block_policies: list[Any]) -> BodyInventory:
    body_blocks = [block for block in block_policies if is_body_policy(block)]
    body_block_ids = [int(block.block_id) for block in body_blocks]
    body_sentences_total = sum(_sentence_count(str(block.original_text)) for block in body_blocks)
    body_characters = sum(len(_normalize_text(str(block.original_text))) for block in body_blocks)
    return BodyInventory(
        body_block_ids=body_block_ids,
        body_blocks_total=len(body_blocks),
        body_sentences_total=body_sentences_total,
        body_characters=body_characters,
        document_scale=document_scale_for_body_size(body_characters),
    )


def document_scale_for_body_size(body_characters: int) -> str:
    if body_characters >= DEFAULT_CONFIG.document_scale_very_long_body_chars:
        return "very_long"
    if body_characters >= DEFAULT_CONFIG.document_scale_long_body_chars:
        return "long"
    if body_characters >= DEFAULT_CONFIG.document_scale_medium_body_chars:
        return "medium"
    return "short"


def compute_body_rewrite_metrics(
    *,
    original: str,
    revised: str,
    guidance: Any | None,
    mode: RewriteMode,
    rewrite_stats: list[RewriteStats],
    suffix: str,
) -> BodyRewriteMetrics:
    original_blocks = _body_blocks_from_guidance_or_text(original, guidance=guidance, suffix=suffix)
    revised_blocks = _revised_body_blocks(original_blocks, revised, guidance=guidance, suffix=suffix)
    stats_by_id = {stats.block_id: stats for stats in rewrite_stats if getattr(stats, "block_id", 0)}

    body_block_ids = [block_id for block_id, _ in original_blocks]
    body_sentences_total = sum(_sentence_count(text) for _, text in original_blocks)
    body_characters = sum(len(_normalize_text(text)) for _, text in original_blocks)
    body_blocks_total = len(original_blocks)
    document_scale = (
        str(getattr(guidance, "document_scale", "") or document_scale_for_body_size(body_characters))
        if guidance is not None
        else document_scale_for_body_size(body_characters)
    )

    changed_blocks = 0
    changed_sentences = 0
    discourse_score = 0
    cluster_score = 0
    developmental_total = 0
    developmental_changed = 0
    body_original_parts: list[str] = []
    body_revised_parts: list[str] = []

    policies_by_id = {
        int(block.block_id): block
        for block in getattr(guidance, "block_policies", [])
        if is_body_policy(block)
    } if guidance is not None else {}

    revised_by_id = {block_id: text for block_id, text in revised_blocks}
    for block_id, original_text in original_blocks:
        revised_text = revised_by_id.get(block_id, "")
        policy = policies_by_id.get(block_id)
        if policy is not None and getattr(policy, "rewrite_depth", "") == "developmental_rewrite":
            developmental_total += 1

        body_original_parts.append(original_text.strip())
        body_revised_parts.append(revised_text.strip())
        sentence_total = _sentence_count(original_text)
        stats = stats_by_id.get(block_id)
        if stats is not None:
            if stats.changed:
                changed_blocks += 1
                changed_units = max(stats.sentence_level_changes, stats.cluster_changes)
                if changed_units <= 0 and stats.rewrite_coverage > 0:
                    changed_units = round(stats.rewrite_coverage * sentence_total)
                if changed_units <= 0 and _normalize_text(original_text) != _normalize_text(revised_text):
                    changed_units = _changed_sentence_count(original_text, revised_text)
                changed_sentences += min(sentence_total, max(1, changed_units))
                discourse_score += stats.discourse_change_score
                cluster_score += stats.cluster_changes
                if policy is not None and getattr(policy, "rewrite_depth", "") == "developmental_rewrite":
                    if stats.cluster_changes > 0 or "sentence_cluster_rewrite" in stats.discourse_actions_used:
                        developmental_changed += 1
            continue

        if _normalize_text(original_text) == _normalize_text(revised_text):
            continue
        changed_blocks += 1
        changed_units = _changed_sentence_count(original_text, revised_text)
        changed_sentences += min(sentence_total, max(1, changed_units))
        block_discourse = _infer_discourse_change_score(original_text, revised_text)
        block_cluster = _infer_cluster_rewrite_score(original_text, revised_text)
        discourse_score += block_discourse
        cluster_score += block_cluster
        if policy is not None and getattr(policy, "rewrite_depth", "") == "developmental_rewrite" and block_cluster > 0:
            developmental_changed += 1

    coverage = changed_sentences / body_sentences_total if body_sentences_total else 0.0
    changed_block_ratio = changed_blocks / body_blocks_total if body_blocks_total else 0.0
    required_coverage = _required_body_coverage(document_scale, mode)
    required_block_ratio = _required_body_changed_block_ratio(document_scale, mode)
    required_discourse = _required_body_discourse_score(document_scale, mode)
    required_cluster = _required_body_cluster_score(document_scale, mode, body_sentences_total)
    quota_reasons = _quota_reason_codes(
        document_scale=document_scale,
        mode=mode,
        body_blocks_total=body_blocks_total,
        body_rewrite_coverage=coverage,
        body_changed_block_ratio=changed_block_ratio,
        body_discourse_change_score=discourse_score,
        body_cluster_rewrite_score=cluster_score,
        body_developmental_blocks_total=developmental_total,
        body_developmental_changed_blocks=developmental_changed,
        required_body_rewrite_coverage=required_coverage,
        required_body_changed_block_ratio=required_block_ratio,
        required_body_discourse_change_score=required_discourse,
        required_body_cluster_rewrite_score=required_cluster,
    )

    return BodyRewriteMetrics(
        body_block_ids=body_block_ids,
        body_blocks_total=body_blocks_total,
        body_sentences_total=body_sentences_total,
        body_characters=body_characters,
        body_changed_blocks=changed_blocks,
        body_changed_sentences=changed_sentences,
        body_changed_block_ratio=changed_block_ratio,
        body_rewrite_coverage=min(1.0, coverage),
        body_discourse_change_score=discourse_score,
        body_cluster_rewrite_score=cluster_score,
        body_developmental_blocks_total=developmental_total,
        body_developmental_changed_blocks=developmental_changed,
        document_scale=document_scale,
        rewrite_quota_met=not quota_reasons,
        rewrite_quota_reason_codes=quota_reasons,
        required_body_rewrite_coverage=required_coverage,
        required_body_changed_block_ratio=required_block_ratio,
        required_body_discourse_change_score=required_discourse,
        required_body_cluster_rewrite_score=required_cluster,
        body_original_text="\n\n".join(part for part in body_original_parts if part),
        body_revised_text="\n\n".join(part for part in body_revised_parts if part),
    )


def body_stats_only(rewrite_stats: list[RewriteStats], body_block_ids: list[int]) -> list[RewriteStats]:
    if not body_block_ids:
        return []
    allowed = set(body_block_ids)
    return [stats for stats in rewrite_stats if stats.block_id in allowed]


def strip_non_body_markdown_lines(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not _is_non_body_markdown_line(line)
    )


def is_body_policy(block: Any) -> bool:
    edit_policy = getattr(block, "edit_policy", "")
    block_type = getattr(block, "block_type", "")
    text = str(getattr(block, "original_text", "") or "")
    if edit_policy not in {"light_edit", "rewritable"}:
        return False
    if block_type not in {"narrative", "light_edit_narration"}:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if _IMAGE_RE.search(stripped):
        return False
    if all(_is_non_body_markdown_line(line) for line in stripped.splitlines() if line.strip()):
        return False
    if _is_pure_technical_dense_block(stripped):
        return False
    return True


def _is_body_text_candidate(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if _IMAGE_RE.search(stripped):
        return False
    lines = [line for line in stripped.splitlines() if line.strip()]
    if not lines:
        return False
    if all(_is_non_body_markdown_line(line) for line in lines):
        return False
    if _is_pure_technical_dense_block(stripped):
        return False
    return True


def _body_blocks_from_guidance_or_text(
    original: str,
    *,
    guidance: Any | None,
    suffix: str,
) -> list[tuple[int, str]]:
    if guidance is not None:
        return [
            (int(block.block_id), str(block.original_text))
            for block in getattr(guidance, "block_policies", [])
            if is_body_policy(block)
        ]

    blocks: list[tuple[int, str]] = []
    block_id = 1
    for paragraph in _paragraphs_from_text(strip_non_body_markdown_lines(original)):
        if _is_pure_technical_dense_block(paragraph):
            continue
        blocks.append((block_id, paragraph))
        block_id += 1
    return blocks


def _revised_body_blocks(
    original_blocks: list[tuple[int, str]],
    revised: str,
    *,
    guidance: Any | None,
    suffix: str,
) -> list[tuple[int, str]]:
    if guidance is not None:
        try:
            revised_chunks = chunk_text(revised, suffix=suffix, max_chars=DEFAULT_CONFIG.max_chunk_chars)
        except ValueError:
            revised_chunks = []
        revised_body_chunks = [chunk.text for chunk in revised_chunks if _is_body_text_candidate(chunk.text)]
        if len(revised_body_chunks) == len(original_blocks):
            return [
                (block_id, revised_body_chunks[index])
                for index, (block_id, _) in enumerate(original_blocks)
            ]

        revised_body_paragraphs = [
            paragraph
            for paragraph in _paragraphs_from_text(strip_non_body_markdown_lines(revised))
            if not _is_pure_technical_dense_block(paragraph)
        ]
        if len(revised_body_paragraphs) == len(original_blocks):
            return [
                (block_id, revised_body_paragraphs[index])
                for index, (block_id, _) in enumerate(original_blocks)
            ]

        result: list[tuple[int, str]] = []
        for block_id, original_text in original_blocks:
            index = block_id - 1
            if 0 <= index < len(revised_chunks):
                result.append((block_id, revised_chunks[index].text))
            else:
                result.append((block_id, original_text))
        return result

    revised_paragraphs = [
        paragraph
        for paragraph in _paragraphs_from_text(strip_non_body_markdown_lines(revised))
        if not _is_pure_technical_dense_block(paragraph)
    ]
    result = []
    for index, (block_id, original_text) in enumerate(original_blocks):
        result.append((block_id, revised_paragraphs[index] if index < len(revised_paragraphs) else original_text))
    return result


def _paragraphs_from_text(text: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
            continue
        if current:
            paragraphs.append("\n".join(current))
            current = []
    if current:
        paragraphs.append("\n".join(current))
    return paragraphs


def _is_non_body_markdown_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _HEADING_LINE_RE.match(stripped):
        return True
    if _IMAGE_LINE_RE.match(stripped):
        return True
    if _CAPTION_LINE_RE.match(stripped):
        return True
    if _LINK_DEF_RE.match(stripped):
        return True
    if _REFERENCE_LINE_RE.match(stripped):
        return True
    if _LIST_LINE_RE.match(stripped):
        return True
    if stripped.startswith(">"):
        return True
    if _TABLE_LINE_RE.match(stripped):
        return True
    if _FORMULA_LINE_RE.match(stripped):
        return True
    if re.fullmatch(r"[-*_]{3,}", stripped):
        return True
    if _PATH_OR_CHECKPOINT_RE.search(stripped) and len(_CJK_RE.findall(stripped)) < 12:
        return True
    return False


def _is_pure_technical_dense_block(text: str) -> bool:
    visible = _normalize_text(text)
    if not visible:
        return True
    cjk_count = len(_CJK_RE.findall(text))
    tech_count = len(_TECH_TOKEN_RE.findall(text))
    citation_count = len(_CITATION_RE.findall(text))
    number_count = len(_NUMBER_RE.findall(text))
    path_count = len(_PATH_OR_CHECKPOINT_RE.findall(text))
    if path_count and cjk_count < 40:
        return True
    if tech_count >= 6 and cjk_count < 80:
        return True
    if citation_count >= 3 and number_count >= 3 and _sentence_count(text) <= 2:
        return True
    return False


def _sentence_count(text: str) -> int:
    if not text.strip():
        return 0
    return max(1, len(split_sentences(text)))


def _changed_sentence_count(original: str, revised: str) -> int:
    original_sentences = [_normalize_sentence(sentence) for sentence in split_sentences(original)]
    revised_sentences = [_normalize_sentence(sentence) for sentence in split_sentences(revised)]
    if not original_sentences:
        return 0
    if not revised_sentences:
        return len(original_sentences)
    matcher = SequenceMatcher(a=original_sentences, b=revised_sentences)
    changed = 0
    for tag, left_start, left_end, _, _ in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += left_end - left_start
    return changed


def _infer_discourse_change_score(original: str, revised: str) -> int:
    score = 0
    original_sentences = split_sentences(original)
    revised_sentences = split_sentences(revised)
    if len(original_sentences) != len(revised_sentences):
        score += 3
    if _changed_sentence_count(original, revised) >= 2:
        score += 2
    if _count_implication_openers(original) > _count_implication_openers(revised):
        score += 2
    if _max_meta_subject_streak(_subject_heads_from_text(original)) > _max_meta_subject_streak(
        _subject_heads_from_text(revised)
    ):
        score += 2
    if _count_repeated_openers(original) > _count_repeated_openers(revised):
        score += 1
    return score


def _infer_cluster_rewrite_score(original: str, revised: str) -> int:
    score = 0
    if len(split_sentences(original)) != len(split_sentences(revised)):
        score += 1
    if _count_implication_openers(original) > _count_implication_openers(revised):
        score += 1
    if _max_meta_subject_streak(_subject_heads_from_text(original)) > _max_meta_subject_streak(
        _subject_heads_from_text(revised)
    ):
        score += 1
    return score


def _quota_reason_codes(
    *,
    document_scale: str,
    mode: RewriteMode,
    body_blocks_total: int,
    body_rewrite_coverage: float,
    body_changed_block_ratio: float,
    body_discourse_change_score: int,
    body_cluster_rewrite_score: int,
    body_developmental_blocks_total: int,
    body_developmental_changed_blocks: int,
    required_body_rewrite_coverage: float,
    required_body_changed_block_ratio: float,
    required_body_discourse_change_score: int,
    required_body_cluster_rewrite_score: int,
) -> list[str]:
    if body_blocks_total == 0:
        return []

    reasons: list[str] = []
    if body_rewrite_coverage < required_body_rewrite_coverage:
        reasons.append("body_rewrite_coverage_below_quota")
    if body_changed_block_ratio < required_body_changed_block_ratio:
        reasons.append("body_changed_blocks_below_quota")
    if body_discourse_change_score < required_body_discourse_change_score:
        reasons.append("body_discourse_change_score_below_quota")
    if body_cluster_rewrite_score < required_body_cluster_rewrite_score:
        reasons.append("body_cluster_rewrite_score_below_quota")
    if document_scale in {"long", "very_long"} and mode is RewriteMode.CONSERVATIVE:
        reasons.append("document_scale_disallows_conservative_fallback")
    if (
        document_scale == "very_long"
        and body_developmental_blocks_total > 0
        and body_developmental_changed_blocks < 1
    ):
        reasons.append("body_developmental_rewrite_missing_for_very_long_document")
    return reasons


def _required_body_coverage(document_scale: str, mode: RewriteMode) -> float:
    if mode is RewriteMode.CONSERVATIVE and document_scale not in {"long", "very_long"}:
        return 0.0
    return {
        "very_long": DEFAULT_CONFIG.body_rewrite_coverage_very_long_threshold,
        "long": DEFAULT_CONFIG.body_rewrite_coverage_long_threshold,
        "medium": DEFAULT_CONFIG.body_rewrite_coverage_medium_threshold,
        "short": DEFAULT_CONFIG.body_rewrite_coverage_short_threshold,
    }.get(document_scale, DEFAULT_CONFIG.body_rewrite_coverage_short_threshold)


def _required_body_changed_block_ratio(document_scale: str, mode: RewriteMode) -> float:
    if mode is RewriteMode.CONSERVATIVE and document_scale not in {"long", "very_long"}:
        return 0.0
    return {
        "very_long": DEFAULT_CONFIG.body_changed_block_ratio_very_long_threshold,
        "long": DEFAULT_CONFIG.body_changed_block_ratio_long_threshold,
        "medium": DEFAULT_CONFIG.body_changed_block_ratio_medium_threshold,
        "short": DEFAULT_CONFIG.body_changed_block_ratio_short_threshold,
    }.get(document_scale, DEFAULT_CONFIG.body_changed_block_ratio_short_threshold)


def _required_body_discourse_score(document_scale: str, mode: RewriteMode) -> int:
    if mode is RewriteMode.CONSERVATIVE and document_scale not in {"long", "very_long"}:
        return 0
    return {
        "very_long": DEFAULT_CONFIG.body_discourse_very_long_threshold,
        "long": DEFAULT_CONFIG.body_discourse_long_threshold,
        "medium": DEFAULT_CONFIG.body_discourse_medium_threshold,
        "short": DEFAULT_CONFIG.body_discourse_short_threshold,
    }.get(document_scale, DEFAULT_CONFIG.body_discourse_short_threshold)


def _required_body_cluster_score(document_scale: str, mode: RewriteMode, body_sentences_total: int) -> int:
    if mode is RewriteMode.CONSERVATIVE and document_scale not in {"long", "very_long"}:
        return 0
    if body_sentences_total < 2:
        return 0
    return {
        "very_long": DEFAULT_CONFIG.body_cluster_very_long_threshold,
        "long": DEFAULT_CONFIG.body_cluster_long_threshold,
        "medium": DEFAULT_CONFIG.body_cluster_medium_threshold,
        "short": DEFAULT_CONFIG.body_cluster_short_threshold,
    }.get(document_scale, DEFAULT_CONFIG.body_cluster_short_threshold)


def _normalize_text(text: str) -> str:
    return _NORMALIZE_RE.sub("", text)


def _normalize_sentence(text: str) -> str:
    stripped = re.sub(r"[。！？?!]+$", "", text)
    return _NORMALIZE_RE.sub("", stripped)


def _count_implication_openers(text: str) -> int:
    return sum(
        1 for sentence in split_sentences(text) if _LEADING_IMPLICATION_RE.match(sentence.strip())
    )


def _count_repeated_openers(text: str) -> int:
    openers = [_extract_opener(sentence) for sentence in split_sentences(text)]
    counts: dict[str, int] = {}
    for opener in openers:
        if opener:
            counts[opener] = counts.get(opener, 0) + 1
    return max(counts.values(), default=0)


def _subject_heads_from_text(text: str) -> list[str]:
    heads: list[str] = []
    for sentence in split_sentences(text):
        stripped = _LEADING_IMPLICATION_RE.sub("", sentence.strip())
        subject = next((item for item in _META_SUBJECTS if stripped.startswith(item)), "")
        heads.append(subject)
    return heads


def _max_meta_subject_streak(subject_heads: list[str]) -> int:
    current = ""
    streak = 0
    max_streak = 0
    for subject in subject_heads:
        if subject in _META_SUBJECTS and subject == current:
            streak += 1
        elif subject in _META_SUBJECTS:
            current = subject
            streak = 1
        else:
            current = ""
            streak = 0
        max_streak = max(max_streak, streak)
    return max_streak


def _extract_opener(sentence: str) -> str:
    stripped = _normalize_sentence(sentence)
    for opener in _STIFF_OPENERS:
        if stripped.startswith(opener):
            return opener
    return stripped[:4]
