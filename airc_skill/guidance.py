from __future__ import annotations

import re
from pathlib import Path

from .chunker import chunk_text
from .config import DEFAULT_CONFIG
from .core_guard import collect_protection_stats, protect_core_content, restore_core_content, snapshot_core_content
from .markdown_guard import protect, restore
from .models import BlockPolicy, GuidanceReport
from .rewriter import split_sentences

_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
_SPECIAL_HEADING_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百0-9]+章.*|摘\s*要\s*$|关\s*键\s*词[:：]?.*$|Abstract\s*$|Keywords[:：]?.*)$"
)
_FORMULA_RE = re.compile(r"\$\$|\\\(|\\\[|\\tag\{")
_CAPTION_RE = re.compile(
    r"^(?:"
    r"(?:图|表)\s*\d+(?:-\d+)?[^\n]*"
    r"|(?:Figure|Table)\s*\d+[^\n]*"
    r"|.*(?:如图|如表|Figure|Table)[^\n]*?(?:所示|如下|展示|流程)[：:]?"
    r")$",
    re.MULTILINE,
)
_IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]+\)")
_LINK_RE = re.compile(r"(?<!!)\[[^\]]+]\([^)]+\)")
_PATH_RE = re.compile(
    r"(?:[A-Za-z]:\\[^\s<>()]+|(?:[\w.-]+/)+[\w./-]+|[\w.-]+\.(?:pth|pt|ckpt|png|jpg|jpeg|pdf))"
)
_CHECKPOINT_RE = re.compile(r"\b(?:checkpoint[s]?/[^\s<>()]+|[\w.-]+\.(?:pth|pt|ckpt))\b")
_CITATION_RE = re.compile(
    r"\[\d+\](?:\[\d+\])*|\([A-Z][A-Za-z\-]+,\s*\d{4}[a-z]?\)|（[A-Z][A-Za-z\-]+,\s*\d{4}[a-z]?）"
)
_REFERENCE_LINE_RE = re.compile(r"^\[\d+\]")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_META_OPENERS = (
    "本研究的主题为",
    "本文讨论的核心问题是",
    "本研究尝试回应",
    "本研究聚焦于",
    "本研究围绕",
)
_TEMPLATE_CONNECTORS = ("同时", "与此同时", "因此", "此外", "由此", "在这种情况下", "综上所述")
_META_SUBJECTS = ("本研究", "本文", "该研究", "该系统")
_COLLOQUIAL_MARKERS = ("这块", "这边", "大家", "我们", "里头", "特别", "真的", "其实", "超", "超级")
_ENUMERATION_RE = re.compile(r"[^，。；]+(?:、[^，。；]+){2,}")
_IMPLICATION_HEAD_RE = re.compile(r"^(?:因此|所以|由此|基于此|在这种情况下|这也意味着)")
_CORE_PROTECTED_PATTERNS = [
    "heading",
    "formula",
    "citation",
    "technical_term",
    "numeric_value",
    "path",
    "checkpoint",
]
_FORMAT_PROTECTED_PATTERNS = [
    "heading_format",
    "english_spacing",
    "placeholder",
    "caption",
    "markdown_symbol",
    "linebreak_whitespace",
    "table",
    "code_block",
]
_NATURALNESS_PRIORITIES = [
    "Prefer reducing repeated subjects before introducing new sentence-open templates.",
    "Delete or absorb mechanical connectors when the logic is already clear.",
    "Keep the original sentence when the available rewrite would sound stiff.",
    "Preserve explicit subjects when removing them would weaken academic reference clarity.",
]


def guide_document_text(
    text: str,
    metadata: dict[str, object] | None = None,
    *,
    suffix: str | None = None,
    source_path: str | Path | None = None,
    max_chars: int | None = None,
) -> GuidanceReport:
    metadata = metadata or {}
    resolved_suffix = suffix or str(metadata.get("suffix", ".txt"))
    resolved_path = source_path or metadata.get("source_path")
    resolved_source = Path(resolved_path).expanduser().resolve() if resolved_path else None
    max_chars = max_chars or DEFAULT_CONFIG.max_chunk_chars

    protection_stats = collect_protection_stats(text, resolved_suffix)
    protected_text = text
    markdown_placeholders: dict[str, str] = {}
    if resolved_suffix == ".md":
        protected_text, markdown_placeholders = protect(text)
    protected_text, core_placeholders = protect_core_content(protected_text, suffix=resolved_suffix)
    chunks = chunk_text(protected_text, suffix=resolved_suffix, max_chars=max_chars)

    block_policies: list[BlockPolicy] = []
    do_not_touch_blocks: list[BlockPolicy] = []
    high_risk_blocks: list[BlockPolicy] = []
    light_edit_blocks: list[BlockPolicy] = []
    rewritable_blocks: list[BlockPolicy] = []
    rewrite_actions_by_block: dict[int, list[str]] = {}
    repeated_subject_blocks: list[int] = []
    meta_dense_blocks: list[int] = []
    template_dense_blocks: list[int] = []

    for index, chunk in enumerate(chunks, start=1):
        visible = restore_core_content(chunk.text, core_placeholders)
        if resolved_suffix == ".md":
            visible = restore(visible, markdown_placeholders)
        policy = _analyze_block(index=index, text=visible, rewritable=chunk.rewritable, suffix=resolved_suffix)
        block_policies.append(policy)
        rewrite_actions_by_block[index] = list(policy.recommended_actions)
        if policy.edit_policy == "do_not_touch":
            do_not_touch_blocks.append(policy)
        elif policy.edit_policy == "high_risk":
            high_risk_blocks.append(policy)
        elif policy.edit_policy == "light_edit":
            light_edit_blocks.append(policy)
        elif policy.edit_policy == "rewritable":
            rewritable_blocks.append(policy)

        if "reduce_repeated_subjects" in policy.recommended_actions:
            repeated_subject_blocks.append(index)
        if "compress_meta_discourse" in policy.recommended_actions:
            meta_dense_blocks.append(index)
        if "soften_template_connectors" in policy.recommended_actions:
            template_dense_blocks.append(index)

    snapshot = snapshot_core_content(text, resolved_suffix)
    core_protected_terms = _deduplicate([*snapshot.technical_terms, *snapshot.paths])[:50]
    agent_notes = _build_agent_notes(block_policies)
    write_gate_preconditions = _build_write_gate_preconditions()

    return GuidanceReport(
        source_path=resolved_source,
        document_risk=_overall_risk_level(block_policies),
        block_policies=block_policies,
        do_not_touch_blocks=do_not_touch_blocks,
        high_risk_blocks=high_risk_blocks,
        light_edit_blocks=light_edit_blocks,
        rewritable_blocks=rewritable_blocks,
        rewrite_actions_by_block=rewrite_actions_by_block,
        core_protected_terms=core_protected_terms,
        core_protected_patterns=list(_CORE_PROTECTED_PATTERNS),
        format_protected_patterns=list(_FORMAT_PROTECTED_PATTERNS),
        naturalness_priorities=list(_NATURALNESS_PRIORITIES),
        agent_notes=agent_notes,
        write_gate_preconditions=write_gate_preconditions,
        format_integrity_status={
            "preflight": "pass",
            "frozen_heading_blocks": protection_stats.frozen_heading_blocks,
            "frozen_formula_blocks": protection_stats.frozen_formula_blocks,
            "frozen_english_blocks": protection_stats.frozen_english_blocks,
            "frozen_placeholder_blocks": protection_stats.frozen_placeholder_blocks,
            "frozen_caption_blocks": protection_stats.frozen_caption_blocks,
        },
        naturalness_review={
            "repeated_subject_blocks": repeated_subject_blocks,
            "meta_dense_blocks": meta_dense_blocks,
            "template_dense_blocks": template_dense_blocks,
            "colloquial_markers_detected": any(marker in text for marker in _COLLOQUIAL_MARKERS),
        },
        write_gate_decision="guidance_only",
    )


def _analyze_block(index: int, text: str, rewritable: bool, suffix: str) -> BlockPolicy:
    stripped = text.strip()
    snapshot = snapshot_core_content(stripped, suffix)
    protected_items = _deduplicate([*snapshot.technical_terms, *snapshot.paths])[:12]
    forbidden_actions = [
        "do_not_change_titles",
        "do_not_change_formulas",
        "do_not_change_terms",
        "do_not_change_citations",
        "do_not_change_numbers_or_paths",
    ]

    if not stripped:
        return BlockPolicy(
            block_id=index,
            block_type="blank",
            risk_level="do_not_touch",
            edit_policy="do_not_touch",
            rewrite_depth="do_not_touch",
            rewrite_intensity="light",
            preview="",
            protected_items=[],
            recommended_actions=["keep_original"],
            required_structural_actions=[],
            required_discourse_actions=[],
            required_minimum_sentence_level_changes=0,
            required_minimum_cluster_changes=0,
            optional_actions=["keep_original"],
            forbidden_actions=forbidden_actions,
            notes=["Blank separator block."],
            original_text="",
            should_rewrite=False,
        )

    block_type = _infer_block_kind(stripped, rewritable)
    repeated_subject = _has_repeated_meta_subject(stripped)
    meta_dense = _meta_density(stripped) >= 2
    template_dense = _template_density(stripped) >= 2
    path_or_checkpoint = bool(_PATH_RE.search(stripped) or _CHECKPOINT_RE.search(stripped))
    citation_dense = len(_CITATION_RE.findall(stripped)) >= 2
    number_dense = len(_NUMBER_RE.findall(stripped)) >= 3
    term_dense = len(snapshot.technical_terms) >= 4

    if block_type in {
        "heading",
        "reference",
        "english_block",
        "caption",
        "placeholder",
        "formula",
        "table",
        "code",
        "list",
        "quote",
    }:
        return BlockPolicy(
            block_id=index,
            block_type=block_type,
            risk_level="do_not_touch",
            edit_policy="do_not_touch",
            rewrite_depth="do_not_touch",
            rewrite_intensity="light",
            preview=_preview(stripped),
            protected_items=protected_items,
            recommended_actions=["skip_rewrite", "format_cleanup_only_if_corrupted"],
            required_structural_actions=[],
            required_discourse_actions=[],
            required_minimum_sentence_level_changes=0,
            required_minimum_cluster_changes=0,
            optional_actions=["skip_rewrite", "format_cleanup_only_if_corrupted"],
            forbidden_actions=forbidden_actions,
            notes=[f"{block_type} block should be preserved verbatim."],
            original_text=stripped,
            should_rewrite=False,
        )

    high_risk = path_or_checkpoint or (citation_dense and number_dense) or (citation_dense and term_dense)
    prefer_light_edit = citation_dense or number_dense or term_dense or "：" in stripped or bool(re.match(r"^[（(]?\d+[）)]", stripped))
    if high_risk:
        notes = ["Mixed technical block with terms, numbers, citations, or deployment references."]
        if path_or_checkpoint:
            notes.append("Contains path or checkpoint reference.")
        actions = ["keep_original", "format_cleanup_only", "keep_explicit_subject_if_needed"]
        return BlockPolicy(
            block_id=index,
            block_type="high_risk_narration",
            risk_level="high_risk",
            edit_policy="high_risk",
            rewrite_depth="high_risk",
            rewrite_intensity="light",
            preview=_preview(stripped),
            protected_items=protected_items,
            recommended_actions=actions,
            required_structural_actions=[],
            required_discourse_actions=[],
            required_minimum_sentence_level_changes=0,
            required_minimum_cluster_changes=0,
            optional_actions=actions,
            forbidden_actions=forbidden_actions,
            notes=notes,
            original_text=stripped,
            should_rewrite=False,
        )

    recommended_actions: list[str] = []
    notes: list[str] = []
    edit_policy = "rewritable"
    block_type = "narrative"
    rewrite_depth = "developmental_rewrite"

    if _looks_like_light_edit_block(stripped) or prefer_light_edit:
        edit_policy = "light_edit"
        block_type = "light_edit_narration"
        rewrite_depth = "light_edit"
        notes.append("Method, system, definition, citation/number-dense, or conclusion block: prefer conservative editing.")
        recommended_actions.extend(
            [
                "reduce_repeated_subjects",
                "compress_meta_discourse",
                "soften_template_connectors",
                "light_clause_reorder",
                "sentence_merge_or_split_light",
            ]
        )
    else:
        notes.append("General narrative block: developmental rewrite is expected.")
        recommended_actions.extend(
            [
                "reduce_repeated_subjects",
                "compress_meta_discourse",
                "sentence_cluster_rewrite",
                "merge_or_split_sentence_cluster",
                "proposition_reorder",
                "conclusion_absorb",
                "enumeration_reframe",
                "transition_absorption",
                "rationale_expansion",
                "soften_template_connectors",
            ]
        )

    if repeated_subject:
        notes.append("Repeated meta subject chain detected.")
        recommended_actions.append("reduce_repeated_subjects")
    if meta_dense:
        notes.append("Meta discourse is dense.")
        recommended_actions.append("compress_meta_discourse")
    if template_dense:
        notes.append("Template connectors are dense.")
        recommended_actions.append("soften_template_connectors")
    if _should_keep_explicit_subject(stripped):
        notes.append("Explicit subject should stay to preserve reference clarity.")
        recommended_actions.append("keep_explicit_subject_if_needed")
    if not repeated_subject and not meta_dense and not template_dense:
        recommended_actions.append("keep_original_if_rewrite_would_be_stiff")

    rewrite_intensity = _determine_rewrite_intensity(
        text=stripped,
        rewrite_depth=rewrite_depth,
        repeated_subject=repeated_subject,
        meta_dense=meta_dense,
        template_dense=template_dense,
    )
    required_structural_actions = _determine_required_structural_actions(
        text=stripped,
        edit_policy=edit_policy,
        repeated_subject=repeated_subject,
    )
    required_discourse_actions = _determine_required_discourse_actions(
        text=stripped,
        rewrite_depth=rewrite_depth,
        repeated_subject=repeated_subject,
        meta_dense=meta_dense,
        template_dense=template_dense,
    )
    minimum_sentence_changes, minimum_cluster_changes = _rewrite_minima(rewrite_depth)
    optional_actions = _deduplicate(
        [
            action
            for action in recommended_actions
            if action not in required_structural_actions and action not in required_discourse_actions
        ]
    )

    return BlockPolicy(
        block_id=index,
        block_type=block_type,
        risk_level=edit_policy,
        edit_policy=edit_policy,
        rewrite_depth=rewrite_depth,
        rewrite_intensity=rewrite_intensity,
        preview=_preview(stripped),
        protected_items=protected_items,
        recommended_actions=_deduplicate(recommended_actions),
        required_structural_actions=required_structural_actions,
        required_discourse_actions=required_discourse_actions,
        required_minimum_sentence_level_changes=minimum_sentence_changes,
        required_minimum_cluster_changes=minimum_cluster_changes,
        optional_actions=optional_actions,
        forbidden_actions=forbidden_actions,
        notes=_deduplicate(notes),
        original_text=stripped,
        should_rewrite=True,
    )


def _build_agent_notes(block_policies: list[BlockPolicy]) -> list[str]:
    notes = [
        "Use block policy as the first gate: do_not_touch blocks must stay verbatim, and high_risk blocks default to keep-original.",
        "When a rewrite only works through a stiff sentence opener, keep the original sentence instead of forcing change.",
        "Preserve explicit subjects when removing them would weaken who performed the action or which system/configuration is being referenced.",
    ]
    for policy in block_policies:
        if policy.edit_policy == "do_not_touch":
            notes.append(
                f"Block {policy.block_id} is {policy.block_type}; preserve it verbatim and only fix formatting if it is already broken."
            )
        elif policy.edit_policy == "high_risk":
            notes.append(
                f"Block {policy.block_id} is high risk; keep the original wording unless only minimal formatting cleanup is needed."
            )
        elif policy.edit_policy == "light_edit":
            notes.append(
                f"Block {policy.block_id} is light_edit with {policy.rewrite_intensity} intensity: deliver at least {policy.required_minimum_sentence_level_changes} sentence-level change, but do not do paragraph-level reorganization."
            )
        elif policy.edit_policy == "rewritable":
            notes.append(
                f"Block {policy.block_id} is developmental_rewrite with {policy.rewrite_intensity} intensity: execute discourse actions {policy.required_discourse_actions or ['sentence_cluster_rewrite']} and satisfy structural obligations {policy.required_structural_actions}."
            )
    return _deduplicate(notes)


def _build_write_gate_preconditions() -> list[str]:
    return [
        "Core content integrity checks must all pass.",
        "Format integrity checks must all pass.",
        "Each rewritable block must satisfy its required_structural_actions.",
        "Each rewritable block must satisfy its required_discourse_actions and rewrite depth minimums.",
        "Light-edit blocks need sentence-level change; developmental-rewrite blocks need cluster-level change.",
        "The rewritten candidate must show effective discourse-level change, not only word substitution.",
        "Severe template risk or repeated-subject risk blocks writing.",
        "At least one high-value structural action must be present before writing.",
    ]


def _determine_required_structural_actions(text: str, edit_policy: str, repeated_subject: bool) -> list[str]:
    if edit_policy not in {"light_edit", "rewritable"}:
        return []

    requirements: list[str] = []
    sentences = split_sentences(text)
    if edit_policy == "rewritable":
        if len(sentences) >= 2:
            requirements.append("pair_fusion")
        elif text.count("，") >= 2:
            requirements.append("clause_reorder")
    if repeated_subject:
        requirements.append("subject_chain_compression")
    if any(_IMPLICATION_HEAD_RE.match(sentence.strip()) for sentence in sentences):
        requirements.append("conclusion_absorb")
    if _ENUMERATION_RE.search(text):
        requirements.append("enumeration_reframe")
    return _deduplicate(requirements)


def _determine_required_discourse_actions(
    text: str,
    rewrite_depth: str,
    repeated_subject: bool,
    meta_dense: bool,
    template_dense: bool,
) -> list[str]:
    if rewrite_depth not in {"light_edit", "developmental_rewrite"}:
        return []

    actions: list[str] = []
    sentences = split_sentences(text)
    if repeated_subject:
        actions.append("subject_chain_compression")
    if meta_dense:
        actions.append("meta_compression")
    if any(_IMPLICATION_HEAD_RE.match(sentence.strip()) for sentence in sentences):
        actions.append("conclusion_absorb")
    if _ENUMERATION_RE.search(text):
        actions.append("enumeration_reframe")
    if template_dense:
        actions.append("transition_absorption")

    if rewrite_depth == "light_edit":
        if not actions:
            actions.append("sentence_level_recast")
        return _deduplicate(actions)

    if len(sentences) >= 2:
        actions.append("sentence_cluster_rewrite")
    if len(sentences) >= 3:
        actions.append("proposition_reorder")
    if re.search(r"(原因|因为|导致|使得|使其|关键在于|更在于|风险)", text):
        actions.append("rationale_expansion")
    return _deduplicate(actions)


def _rewrite_minima(rewrite_depth: str) -> tuple[int, int]:
    if rewrite_depth == "light_edit":
        return 1, 0
    if rewrite_depth == "developmental_rewrite":
        return 2, 1
    return 0, 0


def _determine_rewrite_intensity(
    text: str,
    rewrite_depth: str,
    repeated_subject: bool,
    meta_dense: bool,
    template_dense: bool,
) -> str:
    if rewrite_depth != "developmental_rewrite":
        return "light"
    sentence_count = len(split_sentences(text))
    if sentence_count >= 3 and re.search(r"(背景|意义|风险|分析|挑战|问题|价值|影响|趋势|现状)", text):
        return "high"
    if sentence_count >= 3 and (repeated_subject or meta_dense or template_dense):
        return "high"
    return "medium"


def _infer_block_kind(text: str, rewritable: bool) -> str:
    if _HEADING_RE.search(text) or _SPECIAL_HEADING_RE.match(text.strip()):
        return "heading"
    if _REFERENCE_LINE_RE.match(text.strip()):
        return "reference"
    if _is_english_block(text):
        return "english_block"
    if _CAPTION_RE.search(text):
        return "caption"
    if _IMAGE_RE.search(text) or _LINK_RE.search(text):
        return "placeholder"
    if _FORMULA_RE.search(text):
        return "formula"
    if text.lstrip().startswith(("```", "~~~")):
        return "code"
    if "|" in text and text.count("|") >= 2:
        return "table"
    if text.lstrip().startswith(">"):
        return "quote"
    if re.match(r"^\s*(?:[-*+]\s+|\d+\.\s+|l\s+)", text):
        return "list"
    if not rewritable:
        return "structural"
    return "narrative"


def _looks_like_light_edit_block(text: str) -> bool:
    if any(marker in text for marker in _META_OPENERS) and re.search(
        r"(方法|模型|系统|实现|部署|阈值|实验|流程|模块|架构|定义|总结|展望|结论)",
        text,
    ):
        return True
    return bool(
        re.search(
            r"(本研究|本文).{0,12}(方法|模型|系统|实现|部署|阈值|实验|流程|模块|架构|定义|结论)|"
            r"(方法|系统|实现|部署|结论|定义|架构|模型).{0,10}(如下|主要|包括|采用)",
            text,
        )
    )


def _has_repeated_meta_subject(text: str) -> bool:
    current = ""
    streak = 0
    for sentence in split_sentences(text):
        stripped = sentence.strip()
        subject = next((item for item in _META_SUBJECTS if stripped.startswith(item)), "")
        if subject and subject == current:
            streak += 1
        elif subject:
            current = subject
            streak = 1
        else:
            current = ""
            streak = 0
        if streak >= 2:
            return True
    return False


def _meta_density(text: str) -> int:
    return sum(text.count(marker) for marker in _META_OPENERS)


def _template_density(text: str) -> int:
    return sum(text.count(marker) for marker in _TEMPLATE_CONNECTORS)


def _should_keep_explicit_subject(text: str) -> bool:
    markers = (
        "最终采用",
        "默认运行于",
        "作为本地部署模型",
        "checkpoint",
        "checkpoints/",
        ".pth",
        ".pt",
        ".ckpt",
        "epoch_",
        "Phase",
        "base_only",
        "路径",
    )
    return any(marker in text for marker in markers)


def _is_english_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped or _CJK_RE.search(stripped):
        return False
    english_words = _ENGLISH_WORD_RE.findall(stripped)
    if len(english_words) < 4:
        return False
    ascii_letters = sum(1 for char in stripped if char.isascii() and char.isalpha())
    visible = sum(1 for char in stripped if not char.isspace())
    if visible == 0:
        return False
    return ascii_letters / visible >= 0.45


def _overall_risk_level(blocks: list[BlockPolicy]) -> str:
    levels = {block.edit_policy for block in blocks}
    if {"do_not_touch", "high_risk"} & levels and {"light_edit", "rewritable"} & levels:
        return "mixed"
    if "high_risk" in levels:
        return "high_risk"
    if "light_edit" in levels:
        return "light_edit"
    if "rewritable" in levels:
        return "rewritable"
    return "do_not_touch"


def _preview(text: str, limit: int = 90) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 1]}…"


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


BlockGuidance = BlockPolicy
GuidanceResult = GuidanceReport
