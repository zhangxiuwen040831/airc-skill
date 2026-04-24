from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from .chunker import chunk_text
from .config import DEFAULT_CONFIG

_SENTENCE_END_RE = re.compile(r"[。！？?!]")
_NORMALIZE_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_CITATION_RE = re.compile(
    r"\[\d+\](?:\[\d+\])*|\([A-Z][A-Za-z\-]+,\s*\d{4}[a-z]?\)|（[A-Z][A-Za-z\-]+,\s*\d{4}[a-z]?）"
)
_TECH_TOKEN_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_.\-/]{2,}|checkpoints?/[^，。；\s]+|[\w.-]+\.(?:pth|pt|ckpt)|\[\d+\]"
)

_TOPIC_OPENING_RE = re.compile(
    r"^(?:"
    r"本研究|本文|本章|该研究|本系统|该系统|系统|模型|方法|实验|结果|"
    r"近年来|当前|目前|随着|在方法上|在实验设计上|从整体架构上看|"
    r"综合来看|总体来看|为了解决|针对|面向|围绕"
    r")"
)
_TOPIC_FUNCTION_RE = re.compile(
    r"(面向|围绕|聚焦|提出|构建|采用|具有|旨在|目标|问题|方法|模型|系统|实验|结果|"
    r"结论|挑战|背景|意义|风险|分析|研究对象|研究内容|总体|最终|核心)"
)
_EVIDENCE_RE = re.compile(r"(图|表|数据|样本|实验|结果|显示|表明|准确率|召回率|F1|AUC|RMSE|MAE|%|\[\d+\])")
_CONCLUSION_RE = re.compile(r"^(?:因此|综上|综上所述|总体来看|整体来看|综合来看|由此可见|这说明)")
_SUPPLEMENT_RE = re.compile(r"^(?:需要说明的是|具体而言|更具体地说)")
_DANGLING_OPENING_RE = re.compile(
    r"^(?:"
    r"并进一步|并且|同时|而是|在这种情况下|相应地|围绕这一点|其中|此外|由此|需要说明的是|"
    r"与此同时|另外|基于此|在此基础上|对[^，。；]{1,32}来说"
    r")"
)
_TECHNICAL_DENSE_RE = re.compile(r"(损失函数|公式|指标|阈值|checkpoint|路径|版本|参数|学习率|batch|epoch)")


@dataclass(frozen=True)
class SentenceSkeleton:
    text: str
    role: str
    index: int


@dataclass(frozen=True)
class ParagraphSkeleton:
    sentence_roles: list[SentenceSkeleton]
    opening_rewrite_allowed: bool
    opening_reorder_allowed: bool
    topic_sentence_text: str = ""
    opening_role: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def role_names(self) -> list[str]:
        return [sentence.role for sentence in self.sentence_roles]


def split_paragraph_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    current: list[str] = []
    round_depth = 0
    square_depth = 0
    brace_depth = 0

    for character in text.strip():
        current.append(character)
        if character in "（(":
            round_depth += 1
        elif character in "）)" and round_depth > 0:
            round_depth -= 1
        elif character == "[":
            square_depth += 1
        elif character == "]" and square_depth > 0:
            square_depth -= 1
        elif character == "{":
            brace_depth += 1
        elif character == "}" and brace_depth > 0:
            brace_depth -= 1
        if character in "。！？?!" and round_depth == square_depth == brace_depth == 0:
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []

    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)
    return sentences


def analyze_paragraph_skeleton(text: str) -> ParagraphSkeleton:
    sentences = split_paragraph_sentences(text)
    roles = [
        SentenceSkeleton(text=sentence, role=classify_sentence_role(sentence, index), index=index)
        for index, sentence in enumerate(sentences)
    ]
    notes: list[str] = []
    opening = sentences[0] if sentences else ""
    opening_role = roles[0].role if roles else ""
    opening_is_topic = bool(opening and (opening_role == "topic_sentence" or is_valid_topic_sentence(opening)))
    technical_dense = is_technical_dense_paragraph(text)
    english_block = is_english_block(text)
    opening_rewrite_allowed = bool(opening and not technical_dense and not english_block)
    opening_reorder_allowed = not opening_is_topic

    if opening_is_topic:
        notes.append("opening_topic_sentence_position_locked")
    if technical_dense:
        notes.append("technical_dense_paragraph_prefers_light_opening_edit")
    if english_block:
        notes.append("english_block_frozen_outside_natural_revision")

    return ParagraphSkeleton(
        sentence_roles=roles,
        opening_rewrite_allowed=opening_rewrite_allowed,
        opening_reorder_allowed=opening_reorder_allowed,
        topic_sentence_text=opening if opening_is_topic else "",
        opening_role=opening_role,
        notes=notes,
    )


def classify_sentence_role(sentence: str, index: int = 0) -> str:
    stripped = _strip_sentence(sentence)
    if not stripped:
        return "support_sentence"
    if re.match(r"^[（(](?:\d+|\[\[AIRC:CORE_NUMBER:\d+\]\])[）)]?[^：:。！？?!]{2,60}问题(?:主要体现为)?[：:]", stripped):
        return "topic_sentence"
    if is_valid_topic_sentence(stripped):
        return "topic_sentence"
    if _CONCLUSION_RE.match(stripped):
        return "conclusion_sentence"
    if _DANGLING_OPENING_RE.match(stripped) or _SUPPLEMENT_RE.match(stripped):
        return "transition_sentence"
    if _EVIDENCE_RE.search(stripped) or len(_NUMBER_RE.findall(stripped)) >= 2:
        return "evidence_sentence"
    if index == 0 and len(stripped) >= 18 and _TOPIC_FUNCTION_RE.search(stripped):
        return "topic_sentence"
    return "support_sentence"


def is_valid_topic_sentence(sentence: str) -> bool:
    stripped = _strip_sentence(sentence)
    if not stripped or is_dangling_opening_sentence(stripped):
        return False
    if _TOPIC_OPENING_RE.match(stripped) and _TOPIC_FUNCTION_RE.search(stripped):
        return True
    if re.match(r"^本研究(?:面向|具有|在|围绕|最终|采用|提出|构建)", stripped):
        return True
    if re.match(r"^本文(?:围绕|针对|从|以|主要|旨在|提出|构建)", stripped):
        return True
    if re.match(r"^(?:近年来|当前|目前|随着).{4,80}(?:问题|需求|挑战|研究|技术|场景|任务)", stripped):
        return True
    if re.match(r"^(?:在方法上|在实验设计上|从整体架构上看).{4,100}", stripped):
        return True
    if re.match(r"^[（(](?:\d+|\[\[AIRC:CORE_NUMBER:\d+\]\])[）)]?[^：:。！？?!]{2,60}问题(?:主要体现为)?[：:]", stripped):
        return True
    return False


def opening_style_valid(sentence: str, *, allow_supplement: bool = False) -> bool:
    stripped = _strip_sentence(sentence)
    if not stripped:
        return True
    if is_dangling_opening_sentence(stripped):
        if allow_supplement and _SUPPLEMENT_RE.match(stripped):
            return True
        return is_valid_topic_sentence(stripped)
    return True


def is_dangling_opening_sentence(sentence: str) -> bool:
    stripped = _strip_sentence(sentence)
    if not stripped:
        return False
    if re.match(r"^同时，?(?:本研究|本文|本章|系统|模型|方法|实验)", stripped):
        return False
    return bool(_DANGLING_OPENING_RE.match(stripped))


def paragraph_skeleton_checks(original: str, revised: str) -> dict[str, bool]:
    original_skeleton = analyze_paragraph_skeleton(original)
    revised_skeleton = analyze_paragraph_skeleton(revised)
    original_sentences = [sentence.text for sentence in original_skeleton.sentence_roles]
    revised_sentences = [sentence.text for sentence in revised_skeleton.sentence_roles]
    if not original_sentences or not revised_sentences:
        return {
            "paragraph_topic_sentence_preserved": True,
            "paragraph_opening_style_valid": True,
            "paragraph_skeleton_consistent": True,
            "no_dangling_opening_sentence": True,
            "topic_sentence_not_demoted_to_mid_paragraph": True,
        }

    original_has_topic = bool(original_skeleton.topic_sentence_text)
    allow_supplement = bool(_SUPPLEMENT_RE.match(_strip_sentence(original_sentences[0])))
    source_kept_dependent_opening = (
        not original_has_topic
        and (
            is_dangling_opening_sentence(original_sentences[0])
            or (original_skeleton.role_names and original_skeleton.role_names[0] == "conclusion_sentence")
        )
        and _similarity(original_sentences[0], revised_sentences[0]) >= 0.82
    )
    opening_valid = opening_style_valid(revised_sentences[0], allow_supplement=allow_supplement)
    no_dangling = not is_dangling_opening_sentence(revised_sentences[0]) or (
        allow_supplement and _SUPPLEMENT_RE.match(_strip_sentence(revised_sentences[0])) is not None
    )
    if source_kept_dependent_opening:
        opening_valid = True
        no_dangling = True
    topic_not_demoted = True
    topic_preserved = True
    skeleton_consistent = opening_valid and no_dangling

    if original_has_topic:
        anchor_preserved = semantic_topic_anchor_match(original_sentences[0], revised_sentences[0])
        revised_opening_topic = (
            bool(revised_skeleton.role_names and revised_skeleton.role_names[0] == "topic_sentence")
            or is_valid_topic_sentence(revised_sentences[0])
            or anchor_preserved
        )
        topic_not_demoted = _topic_sentence_not_demoted(original_sentences[0], revised_sentences)
        topic_preserved = revised_opening_topic and topic_not_demoted
        skeleton_consistent = skeleton_consistent and topic_preserved
    elif revised_skeleton.role_names and not source_kept_dependent_opening:
        original_opening_role = original_skeleton.role_names[0] if original_skeleton.role_names else ""
        revised_opening_role = revised_skeleton.role_names[0]
        if revised_opening_role in {"transition_sentence", "conclusion_sentence"}:
            skeleton_consistent = skeleton_consistent and revised_opening_role == original_opening_role

    return {
        "paragraph_topic_sentence_preserved": topic_preserved,
        "paragraph_opening_style_valid": opening_valid,
        "paragraph_skeleton_consistent": skeleton_consistent,
        "no_dangling_opening_sentence": no_dangling,
        "topic_sentence_not_demoted_to_mid_paragraph": topic_not_demoted,
    }


def semantic_topic_anchor_match(original_opening: str, revised_opening: str) -> bool:
    """Allow light topic-sentence rewrites when key semantic anchors stay up front."""

    original_anchors = _topic_anchors(original_opening)
    if not original_anchors:
        return _similarity(original_opening, revised_opening) >= 0.38
    revised_normalized = _normalize(revised_opening)
    matched = sum(1 for anchor in original_anchors if anchor in revised_normalized)
    required = 1 if len(original_anchors) <= 2 else 2
    return matched >= required


def role_preserving_paragraph_compare(original: str, revised: str) -> bool:
    """Check whether a paragraph keeps the source opening role under surface rewrite."""

    checks = paragraph_skeleton_checks(original, revised)
    return bool(
        checks["paragraph_topic_sentence_preserved"]
        and checks["paragraph_opening_style_valid"]
        and checks["topic_sentence_not_demoted_to_mid_paragraph"]
    )


def paragraph_skeleton_tolerance_window(original: str, revised: str) -> bool:
    """Accept topic-support rewrites that keep the opening anchor and avoid dangling starts."""

    original_sentences = split_paragraph_sentences(original)
    revised_sentences = split_paragraph_sentences(revised)
    if not original_sentences or not revised_sentences:
        return True
    if is_dangling_opening_sentence(revised_sentences[0]):
        return False
    return semantic_topic_anchor_match(original_sentences[0], revised_sentences[0])


def document_paragraph_skeleton_review(
    original: str,
    revised: str,
    *,
    guidance: Any | None = None,
    rewrite_stats: list[Any] | None = None,
    suffix: str = ".txt",
) -> dict[str, Any]:
    pairs = _body_paragraph_pairs(original, revised, guidance=guidance, rewrite_stats=rewrite_stats, suffix=suffix)
    failing_topic: list[int] = []
    failing_opening: list[int] = []
    failing_skeleton: list[int] = []
    failing_dangling: list[int] = []
    failing_demoted: list[int] = []
    topic_paragraphs = 0

    for block_id, original_text, revised_text in pairs:
        if is_english_block(original_text) or is_technical_dense_paragraph(original_text):
            continue
        original_skeleton = analyze_paragraph_skeleton(original_text)
        if original_skeleton.topic_sentence_text:
            topic_paragraphs += 1
        checks = paragraph_skeleton_checks(original_text, revised_text)
        if not checks["paragraph_topic_sentence_preserved"]:
            failing_topic.append(block_id)
        if not checks["paragraph_opening_style_valid"]:
            failing_opening.append(block_id)
        if not checks["paragraph_skeleton_consistent"]:
            failing_skeleton.append(block_id)
        if not checks["no_dangling_opening_sentence"]:
            failing_dangling.append(block_id)
        if not checks["topic_sentence_not_demoted_to_mid_paragraph"]:
            failing_demoted.append(block_id)

    return {
        "paragraphs_checked": len(pairs),
        "topic_paragraphs_checked": topic_paragraphs,
        "paragraph_topic_sentence_preserved": not failing_topic,
        "paragraph_opening_style_valid": not failing_opening,
        "paragraph_skeleton_consistent": not failing_skeleton,
        "no_dangling_opening_sentence": not failing_dangling,
        "topic_sentence_not_demoted_to_mid_paragraph": not failing_demoted,
        "paragraph_topic_sentence_failed_block_ids": failing_topic,
        "paragraph_opening_style_failed_block_ids": failing_opening,
        "paragraph_skeleton_failed_block_ids": failing_skeleton,
        "dangling_opening_failed_block_ids": failing_dangling,
        "topic_sentence_demoted_block_ids": failing_demoted,
    }


def is_technical_dense_paragraph(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    tech_tokens = _TECH_TOKEN_RE.findall(stripped)
    numbers = _NUMBER_RE.findall(stripped)
    citations = _CITATION_RE.findall(stripped)
    return bool(_TECHNICAL_DENSE_RE.search(stripped)) or len(tech_tokens) >= 3 or (
        len(numbers) >= 3 and len(citations) >= 1
    )


def is_english_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped or _CJK_RE.search(stripped):
        return False
    words = _ENGLISH_WORD_RE.findall(stripped)
    visible = sum(1 for char in stripped if not char.isspace())
    ascii_letters = sum(1 for char in stripped if char.isascii() and char.isalpha())
    if len(words) < 4 and ascii_letters < 24:
        return False
    return visible > 0 and ascii_letters / visible >= 0.65


def _topic_sentence_not_demoted(original_opening: str, revised_sentences: list[str]) -> bool:
    if len(revised_sentences) <= 1:
        return True
    first_similarity = _similarity(original_opening, revised_sentences[0])
    later_similarity = max(_similarity(original_opening, sentence) for sentence in revised_sentences[1:])
    if semantic_topic_anchor_match(original_opening, revised_sentences[0]):
        return True
    return not (later_similarity >= 0.58 and later_similarity > first_similarity + 0.16)


def _body_paragraph_pairs(
    original: str,
    revised: str,
    *,
    guidance: Any | None,
    rewrite_stats: list[Any] | None,
    suffix: str,
) -> list[tuple[int, str, str]]:
    if guidance is None:
        original_paragraphs = _plain_body_paragraphs(original)
        revised_paragraphs = _plain_body_paragraphs(revised)
        return [
            (index + 1, original_text, revised_paragraphs[index] if index < len(revised_paragraphs) else original_text)
            for index, original_text in enumerate(original_paragraphs)
        ]

    policies = list(getattr(guidance, "block_policies", []))
    body_policies = [policy for policy in policies if _is_body_policy_like(policy)]
    stats_by_id = {
        int(getattr(stats, "block_id", 0)): stats
        for stats in (rewrite_stats or [])
        if int(getattr(stats, "block_id", 0))
    }
    if stats_by_id:
        pairs: list[tuple[int, str, str]] = []
        for policy in body_policies:
            block_id = int(policy.block_id)
            policy_original = str(getattr(policy, "original_text", "") or "")
            stats = stats_by_id.get(block_id)
            if stats is None:
                pairs.append((block_id, policy_original, policy_original))
                continue
            stats_original = "".join(getattr(stats, "original_sentences", [])).strip() or policy_original
            stats_revised = "".join(getattr(stats, "rewritten_sentences", [])).strip() or policy_original
            pairs.append((block_id, stats_original, stats_revised))
        return pairs
    try:
        revised_chunks = chunk_text(revised, suffix=suffix, max_chars=DEFAULT_CONFIG.max_chunk_chars)
    except ValueError:
        revised_chunks = []

    revised_body_chunks = [chunk.text for chunk in revised_chunks if _is_body_text_candidate_like(chunk.text)]
    if len(revised_body_chunks) == len(body_policies):
        return [
            (int(policy.block_id), str(getattr(policy, "original_text", "") or ""), revised_body_chunks[index])
            for index, policy in enumerate(body_policies)
        ]

    revised_paragraphs = _plain_body_paragraphs(revised)
    if len(revised_paragraphs) == len(body_policies):
        return [
            (int(policy.block_id), str(getattr(policy, "original_text", "") or ""), revised_paragraphs[index])
            for index, policy in enumerate(body_policies)
        ]

    pairs: list[tuple[int, str, str]] = []
    for policy in body_policies:
        block_id = int(getattr(policy, "block_id", 0))
        original_text = str(getattr(policy, "original_text", "") or "")
        revised_text = original_text
        if 0 < block_id <= len(revised_chunks):
            revised_text = revised_chunks[block_id - 1].text
        pairs.append((block_id, original_text, revised_text))
    return pairs


def _is_body_policy_like(policy: Any) -> bool:
    edit_policy = getattr(policy, "edit_policy", "")
    block_type = getattr(policy, "block_type", "")
    text = str(getattr(policy, "original_text", "") or "")
    return (
        edit_policy in {"light_edit", "rewritable"}
        and block_type in {"narrative", "light_edit_narration"}
        and bool(text.strip())
        and not is_english_block(text)
    )


def _is_body_text_candidate_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped or _looks_structural(stripped) or is_english_block(stripped):
        return False
    if not _CJK_RE.search(stripped):
        return False
    return True


def _plain_body_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                paragraph = "\n".join(current)
                if not _looks_structural(paragraph):
                    paragraphs.append(paragraph)
                current = []
            continue
        if _looks_structural(stripped):
            if current:
                paragraph = "\n".join(current)
                if not _looks_structural(paragraph):
                    paragraphs.append(paragraph)
                current = []
            continue
        current.append(line)
    if current:
        paragraph = "\n".join(current)
        if not _looks_structural(paragraph):
            paragraphs.append(paragraph)
    return paragraphs


def _looks_structural(text: str) -> bool:
    stripped = text.strip()
    return bool(
        not stripped
        or re.match(r"^\s{0,3}#{1,6}\s+", stripped)
        or re.match(r"^\s*!\[[^\]]*]\([^)]+\)\s*$", stripped)
        or re.match(r"^\s*(?:[-*+]\s+|\d+\.\s+)", stripped)
        or re.match(r"^\s*\|.*\|\s*$", stripped)
        or re.match(r"^\s*(?:\$\$|\\\[|\\\(|\\begin\{|\\end\{)", stripped)
    )


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(a=_normalize(left), b=_normalize(right)).ratio()


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub("", _strip_sentence(text))


def _strip_sentence(sentence: str) -> str:
    return sentence.strip().rstrip("。！？?!；;")


def _topic_anchors(sentence: str) -> list[str]:
    stripped = _normalize(sentence)
    anchors: list[str] = []
    for pattern in (
        r"AIGC",
        r"Deepfake",
        r"ViT",
        r"CNN",
        r"本研究",
        r"系统",
        r"模型",
        r"方法",
        r"实验",
        r"检测",
        r"研究",
        r"生成式?图像",
        r"语义分支",
        r"频域分支",
        r"困难真实样本",
    ):
        for match in re.finditer(pattern, stripped, flags=re.I):
            anchors.append(match.group(0))
    for token in _TECH_TOKEN_RE.findall(stripped):
        anchors.append(token)
    return list(dict.fromkeys(_normalize(anchor) for anchor in anchors if anchor))
