from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from .body_metrics import build_body_inventory
from .chapter_policy import (
    chapter_summary_from_blocks,
    classify_chapter_type,
    clean_heading_title,
    priority_for_chapter,
    quota_for_priority,
    target_intensity_for_priority,
)
from .chunker import chunk_text
from .config import DEFAULT_CONFIG
from .core_guard import collect_protection_stats, protect_core_content, restore_core_content, snapshot_core_content
from .markdown_guard import protect, restore
from .models import BlockPolicy, GuidanceReport
from .natural_revision_profile import ACADEMIC_NATURAL_STUDENTLIKE, COLLOQUIAL_FORBIDDEN_MARKERS, get_natural_revision_profile
from .paragraph_skeleton import analyze_paragraph_skeleton
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
_COLLOQUIAL_MARKERS = COLLOQUIAL_FORBIDDEN_MARKERS
_FUNCTION_WORD_RE = re.compile(r"(进行了|进行|在[^，。；]{2,20}(?:中|过程中|背景下)|的[^，。；]{1,12}的)")
_STIFF_VERB_RE = re.compile(r"(进行|构建|实现|确保|提升|开展|推动)")
_PARALLELISM_RE = re.compile(
    r"(不仅要.+?还要|应从.+?层面|奠定坚实基础|具有重要意义并提供有力支撑|形成完整闭环并实现有效提升)"
)
_DENSE_NOMINAL_RE = re.compile(r"(?:[^，。；]{1,8}的){3,}[^，。；]{1,12}")
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
    *ACADEMIC_NATURAL_STUDENTLIKE.style_principles,
    "Prefer reducing repeated subjects before introducing new sentence-open templates.",
    "Delete or absorb mechanical connectors when the logic is already clear.",
    "Keep the original sentence when the available rewrite would sound stiff.",
    "Preserve explicit subjects when removing them would weaken academic reference clarity.",
    "Add local human revision realism: vary sentence cadence, soften overexplicit transitions, and rebuild paragraph-internal hierarchy without moving the topic sentence.",
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
    target_style = str(metadata.get("target_style", "academic_natural"))
    style_profile = get_natural_revision_profile(target_style)
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
    current_chapter_title = ""
    current_chapter_id = 0

    for index, chunk in enumerate(chunks, start=1):
        visible = restore_core_content(chunk.text, core_placeholders)
        if resolved_suffix == ".md":
            visible = restore(visible, markdown_placeholders)
        if _is_chapter_heading_candidate(visible):
            current_chapter_title = clean_heading_title(visible)
            current_chapter_id += 1
        policy = _analyze_block(
            index=index,
            text=visible,
            rewritable=chunk.rewritable,
            suffix=resolved_suffix,
            chapter_title=current_chapter_title,
            chapter_id=current_chapter_id,
            target_style=style_profile.name,
        )
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

    initial_inventory = build_body_inventory(block_policies)
    block_policies = _apply_document_scale_pressure(block_policies, initial_inventory.document_scale)
    body_inventory = build_body_inventory(block_policies)
    do_not_touch_blocks = [block for block in block_policies if block.edit_policy == "do_not_touch"]
    high_risk_blocks = [block for block in block_policies if block.edit_policy == "high_risk"]
    light_edit_blocks = [block for block in block_policies if block.edit_policy == "light_edit"]
    rewritable_blocks = [block for block in block_policies if block.edit_policy == "rewritable"]
    rewrite_actions_by_block = {block.block_id: list(block.recommended_actions) for block in block_policies}

    snapshot = snapshot_core_content(text, resolved_suffix)
    core_protected_terms = _deduplicate([*snapshot.technical_terms, *snapshot.paths])[:50]
    agent_notes = _build_agent_notes(block_policies, body_inventory.document_scale, style_profile.name)
    write_gate_preconditions = _build_write_gate_preconditions()
    chapter_policy_summary = chapter_summary_from_blocks(block_policies)

    return GuidanceReport(
        source_path=resolved_source,
        document_risk=_overall_risk_level(block_policies),
        document_scale=body_inventory.document_scale,
        body_block_ids=body_inventory.body_block_ids,
        body_blocks_total=body_inventory.body_blocks_total,
        body_sentences_total=body_inventory.body_sentences_total,
        body_characters=body_inventory.body_characters,
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
            "document_scale": body_inventory.document_scale,
            "body_block_ids": body_inventory.body_block_ids,
            "body_blocks_total": body_inventory.body_blocks_total,
            "body_sentences_total": body_inventory.body_sentences_total,
            "body_characters": body_inventory.body_characters,
            "chapter_policy_summary": chapter_policy_summary,
            "target_style": style_profile.name,
            "style_profile": style_profile.name,
        },
        write_gate_decision="guidance_only",
        chapter_policy_summary=chapter_policy_summary,
    )


def _analyze_block(
    index: int,
    text: str,
    rewritable: bool,
    suffix: str,
    chapter_title: str = "",
    chapter_id: int = 0,
    target_style: str = "academic_natural_studentlike",
) -> BlockPolicy:
    stripped = text.strip()
    chapter_type = classify_chapter_type(chapter_title, stripped)
    chapter_priority = priority_for_chapter(chapter_type, stripped)
    chapter_quota = quota_for_priority(chapter_priority).to_dict()
    chapter_intensity = target_intensity_for_priority(chapter_priority)
    paragraph_skeleton = analyze_paragraph_skeleton(stripped) if stripped else None
    l2_profile_enabled = target_style == "zh_academic_l2_mild"

    def finish(policy: BlockPolicy) -> BlockPolicy:
        return replace(
            policy,
            chapter_id=chapter_id,
            chapter_title=chapter_title or "Inferred section",
            chapter_type=chapter_type,
            chapter_rewrite_priority=chapter_priority,
            chapter_rewrite_quota=chapter_quota,
            chapter_rewrite_intensity=chapter_intensity,
            paragraph_sentence_roles=paragraph_skeleton.role_names if paragraph_skeleton is not None else [],
            opening_rewrite_allowed=(
                paragraph_skeleton.opening_rewrite_allowed if paragraph_skeleton is not None else True
            ),
            opening_reorder_allowed=(
                paragraph_skeleton.opening_reorder_allowed if paragraph_skeleton is not None else True
            ),
            topic_sentence_text=paragraph_skeleton.topic_sentence_text if paragraph_skeleton is not None else "",
            high_sensitivity_prose=_is_high_sensitivity_prose_block(
                text=stripped,
                block_type=policy.block_type,
                chapter_type=chapter_type,
                chapter_title=chapter_title,
            ),
        )

    snapshot = snapshot_core_content(stripped, suffix)
    protected_items = _deduplicate([*snapshot.technical_terms, *snapshot.paths])[:12]
    forbidden_actions = [
        "do_not_change_titles",
        "do_not_change_formulas",
        "do_not_change_terms",
        "do_not_change_citations",
        "do_not_change_numbers_or_paths",
        "do_not_start_paragraph_with_dangling_transition",
    ]

    if not stripped:
        return finish(BlockPolicy(
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
        ))

    block_type = _infer_block_kind(stripped, rewritable)
    repeated_subject = _has_repeated_meta_subject(stripped)
    meta_dense = _meta_density(stripped) >= 2
    template_dense = _template_density(stripped) >= 2
    function_word_dense = _function_word_density(stripped) >= 2
    stiff_verb_dense = _stiff_verb_density(stripped) >= 2
    parallelism = _has_template_parallelism(stripped)
    dense_nominal = _has_dense_nominal_phrase(stripped)
    rhythm_flat = _has_flat_sentence_rhythm(stripped)
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
        return finish(BlockPolicy(
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
        ))

    high_risk = path_or_checkpoint or (citation_dense and number_dense) or (citation_dense and term_dense)
    prefer_light_edit = citation_dense or number_dense or term_dense or "：" in stripped or bool(re.match(r"^[（(]?\d+[）)]", stripped))
    if high_risk:
        notes = ["Mixed technical block with terms, numbers, citations, or deployment references."]
        if path_or_checkpoint:
            notes.append("Contains path or checkpoint reference.")
        actions = ["keep_original", "format_cleanup_only", "keep_explicit_subject_if_needed"]
        if term_dense or path_or_checkpoint:
            actions.append("keep_original_if_technical_density_is_high")
        return finish(BlockPolicy(
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
        ))

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
                "compress_subject_chain",
                "compress_meta_discourse",
                "partial_keep",
                "uneven_rewrite_distribution",
                "soften_template_connectors",
                "weaken_template_connectors",
                "reduce_function_word_overuse",
                "rewrite_dense_nominal_phrases",
                "rebuild_sentence_rhythm",
                "preserve_explicit_subject_if_clarity_needed",
                "light_clause_reorder",
                "sentence_merge_or_split_light",
                *_local_revision_realism_actions("light_edit"),
            ]
        )
    else:
        notes.append("General narrative block: developmental rewrite is expected.")
        recommended_actions.extend(
            [
                "reduce_repeated_subjects",
                "compress_subject_chain",
                "compress_meta_discourse",
                "sentence_cluster_merge",
                "sentence_cluster_split",
                "discourse_reordering",
                "narrative_path_rewrite",
                "conclusion_absorption",
                "uneven_rewrite_distribution",
                "sentence_cluster_rewrite",
                "merge_or_split_sentence_cluster",
                "proposition_reorder",
                "conclusion_absorb",
                "enumeration_reframe",
                "transition_absorption",
                "rationale_expansion",
                "soften_template_connectors",
                "weaken_template_connectors",
                "reduce_function_word_overuse",
                "rewrite_dense_nominal_phrases",
                "rebuild_sentence_rhythm",
                "break_parallelism",
                "preserve_explicit_subject_if_clarity_needed",
                *_local_revision_realism_actions("developmental_rewrite"),
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
        recommended_actions.append("weaken_template_connectors")
    if function_word_dense or stiff_verb_dense:
        notes.append("Function words or empty academic verbs are overused; reduce them only when meaning stays stable.")
        recommended_actions.append("reduce_function_word_overuse")
    if dense_nominal:
        notes.append("Dense nominal phrases detected; simplify phrase structure rather than expanding content.")
        recommended_actions.append("rewrite_dense_nominal_phrases")
    if rhythm_flat:
        notes.append("Sentence lengths are too even; rebuild rhythm with safe merge/split decisions.")
        recommended_actions.append("rebuild_sentence_rhythm")
    if parallelism:
        notes.append("Template-like parallelism or slogan-style phrasing detected.")
        recommended_actions.append("break_parallelism")
    if _should_keep_explicit_subject(stripped):
        notes.append("Explicit subject should stay to preserve reference clarity.")
        recommended_actions.append("keep_explicit_subject_if_needed")
        recommended_actions.append("preserve_explicit_subject_if_clarity_needed")
    if term_dense:
        notes.append("Technical term density is high; simplify sentence structure rather than expanding or generalizing terms.")
        recommended_actions.append("keep_original_if_technical_density_is_high")
    if not repeated_subject and not meta_dense and not template_dense:
        recommended_actions.append("keep_original_if_rewrite_would_be_stiff")
    if paragraph_skeleton is not None and paragraph_skeleton.topic_sentence_text:
        notes.append(
            "Paragraph opening is a valid topic sentence: light recast is allowed, but its opening function and position are locked."
        )
        recommended_actions.append("preserve_topic_sentence_at_opening")
        forbidden_actions.extend(
            [
                "do_not_move_topic_sentence_from_paragraph_start",
                "do_not_absorb_topic_sentence_into_later_sentence",
            ]
        )
    if _is_high_sensitivity_prose_block(
        text=stripped,
        block_type=block_type,
        chapter_type=chapter_type,
        chapter_title=chapter_title,
    ):
        notes.append(
            "High-sensitivity prose mode: preserve sentence completeness and paragraph readability even if this reduces rewrite scale."
        )
        recommended_actions.append("readability_repair_pass")
        recommended_actions.append("sentence_completeness_repair")
        recommended_actions.extend(
            [
                "de_template_academic_cliche",
                "retain_some_plain_sentences",
                "soft_keep_for_human_revision_feel",
            ]
        )
        forbidden_actions.extend(
            [
                "do_not_leave_fragment_like_support_sentences",
                "do_not_leave_dangling_conclusion_sentences",
                "do_not_uniformly_polish_every_sentence",
            ]
        )
    semantic_role_actions = _semantic_role_integrity_actions(
        text=stripped,
        block_type=block_type,
        chapter_type=chapter_type,
        high_sensitivity_prose=_is_high_sensitivity_prose_block(
            text=stripped,
            block_type=block_type,
            chapter_type=chapter_type,
            chapter_title=chapter_title,
        ),
    )
    if semantic_role_actions:
        notes.append("Preserve semantic role: keep core, mechanism, enumeration, and summary sentences in their original discourse role.")
        recommended_actions.extend(semantic_role_actions)
        forbidden_actions.extend(
            [
                "do_not_replace_concrete_subject_with_abstract_scaffolding",
                "do_not_turn_enumeration_items_into_appendix_like_support",
            ]
        )
    authorial_intent_actions = _authorial_intent_actions(
        text=stripped,
        block_type=block_type,
        chapter_type=chapter_type,
        high_sensitivity_prose=_is_high_sensitivity_prose_block(
            text=stripped,
            block_type=block_type,
            chapter_type=chapter_type,
            chapter_title=chapter_title,
        ),
    )
    if authorial_intent_actions:
        notes.append(
            "Preserve authorial intent: keep mechanism sentences direct, avoid appendix-like support phrasing, and retain source stance markers when they carry the argument."
        )
        recommended_actions.extend(authorial_intent_actions)
        forbidden_actions.extend(
            [
                "do_not_weaken_mechanism_assertions_into_appendix_like_support",
                "do_not_remove_source_contrast_or_choice_markers_without_replacement",
            ]
        )
    evidence_fidelity_actions = _evidence_fidelity_actions(
        text=stripped,
        block_type=block_type,
        chapter_type=chapter_type,
        high_sensitivity_prose=_is_high_sensitivity_prose_block(
            text=stripped,
            block_type=block_type,
            chapter_type=chapter_type,
            chapter_title=chapter_title,
        ),
    )
    if evidence_fidelity_actions:
        notes.append(
            "Preserve source evidence scope and thesis register: do not add outside background, commentary, metaphor, or unjustified first-person claims."
        )
        recommended_actions.extend(evidence_fidelity_actions)
        forbidden_actions.extend(
            [
                "do_not_add_external_background_or_statistics",
                "do_not_turn_thesis_prose_into_commentary_or_storytelling",
                "do_not_switch_to_first_person_without_source_support",
            ]
        )
    sentence_naturalization_actions = _academic_sentence_naturalization_actions(
        text=stripped,
        block_type=block_type,
        chapter_type=chapter_type,
        high_sensitivity_prose=_is_high_sensitivity_prose_block(
            text=stripped,
            block_type=block_type,
            chapter_type=chapter_type,
            chapter_title=chapter_title,
        ),
    )
    if sentence_naturalization_actions:
        notes.append(
            "Naturalize academic sentence shape: reduce project-style openings, repeated explicit subjects, slogan-like goals, and over-structured parallel syntax without weakening the source claim."
        )
        recommended_actions.extend(sentence_naturalization_actions)
        forbidden_actions.extend(
            [
                "do_not_replace_direct_thesis_sentences_with_project_report_openings",
                "do_not_use_slogan_like_goal_phrasing_for_ordinary_thesis_prose",
            ]
        )
    if l2_profile_enabled and edit_policy in {"light_edit", "rewritable"} and not term_dense and not path_or_checkpoint:
        notes.append(
            "Optional zh_academic_l2_mild profile: keep prose academic and grammatical, but allow mild explanatory wording, slightly more function words, and less native-like compression."
        )
        recommended_actions.extend(
            [
                "expand_compact_academic_clause",
                "increase_function_word_density_mildly",
                "soften_native_like_concision",
                "allow_explanatory_rephrasing",
                "inject_mild_l2_texture",
                "avoid_too_fluent_native_polish",
            ]
        )
        forbidden_actions.extend(
            [
                "do_not_make_l2_profile_colloquial",
                "do_not_create_ungrammatical_l2_errors",
                "do_not_add_external_facts_for_l2_texture",
            ]
        )

    edit_policy, block_type, rewrite_depth, recommended_actions, notes = _apply_chapter_rewrite_policy(
        text=stripped,
        edit_policy=edit_policy,
        block_type=block_type,
        rewrite_depth=rewrite_depth,
        recommended_actions=recommended_actions,
        notes=notes,
        chapter_type=chapter_type,
        chapter_priority=chapter_priority,
        has_explicit_chapter=chapter_id > 0,
        citation_dense=citation_dense,
        number_dense=number_dense,
        term_dense=term_dense,
        path_or_checkpoint=path_or_checkpoint,
    )
    rewrite_intensity = _determine_rewrite_intensity(
        text=stripped,
        rewrite_depth=rewrite_depth,
        repeated_subject=repeated_subject,
        meta_dense=meta_dense,
        template_dense=template_dense,
        function_word_dense=function_word_dense or stiff_verb_dense,
        parallelism=parallelism,
        dense_nominal=dense_nominal,
        rhythm_flat=rhythm_flat,
    )
    if chapter_priority == "high" and rewrite_depth == "developmental_rewrite" and chapter_id > 0:
        rewrite_intensity = "high"
    elif chapter_priority == "medium" and rewrite_depth == "developmental_rewrite":
        rewrite_intensity = "medium"
    elif chapter_priority == "conservative":
        rewrite_intensity = "light"
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
        function_word_dense=function_word_dense or stiff_verb_dense,
        parallelism=parallelism,
        dense_nominal=dense_nominal,
        rhythm_flat=rhythm_flat,
    )
    minimum_sentence_changes, minimum_cluster_changes = _rewrite_minima(rewrite_depth)
    revision_pattern = _determine_revision_pattern(
        index=index,
        text=stripped,
        edit_policy=edit_policy,
        rewrite_depth=rewrite_depth,
        repeated_subject=repeated_subject,
        template_dense=template_dense,
        rhythm_flat=rhythm_flat,
        parallelism=parallelism,
    )
    optional_actions = _deduplicate(
        [
            action
            for action in recommended_actions
            if action not in required_structural_actions and action not in required_discourse_actions
        ]
    )

    return finish(BlockPolicy(
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
        revision_pattern=revision_pattern,
    ))


def _is_chapter_heading_candidate(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    first_line = next((line.strip() for line in stripped.splitlines() if line.strip()), "")
    if _HEADING_RE.match(first_line):
        return True
    if re.match(r"^第[一二三四五六七八九十百0-9]+章", first_line):
        return True
    if re.match(r"^\d+(?:\.\d+)*\s+.{2,}$", first_line):
        return True
    return False


def _apply_chapter_rewrite_policy(
    *,
    text: str,
    edit_policy: str,
    block_type: str,
    rewrite_depth: str,
    recommended_actions: list[str],
    notes: list[str],
    chapter_type: str,
    chapter_priority: str,
    has_explicit_chapter: bool,
    citation_dense: bool,
    number_dense: bool,
    term_dense: bool,
    path_or_checkpoint: bool,
) -> tuple[str, str, str, list[str], list[str]]:
    actions = list(recommended_actions)
    updated_notes = list(notes)
    technical_dense = citation_dense or number_dense or term_dense or path_or_checkpoint
    hard_technical_dense = path_or_checkpoint

    if chapter_priority == "high" and has_explicit_chapter and not hard_technical_dense:
        edit_policy = "rewritable"
        block_type = "narrative"
        rewrite_depth = "developmental_rewrite"
        actions = _deduplicate(
            [
                *actions,
                "chapter_high_priority_rewrite",
                "sentence_cluster_rewrite",
                "narrative_flow_rebuilder",
                "meta_compression",
                "subject_chain_compression",
                "sentence_cluster_merge",
                "discourse_reordering",
                "narrative_path_rewrite",
                "conclusion_absorption",
            ]
        )
        updated_notes.append(
            f"Chapter-aware policy: {chapter_type} is high priority; rewrite body narration more aggressively."
        )
        if term_dense or citation_dense or number_dense:
            updated_notes.append(
                "Chapter-aware policy: preserve technical terms, citations, and numbers exactly while still rebuilding the surrounding narrative."
            )
    elif chapter_priority == "medium":
        if edit_policy != "high_risk" and not path_or_checkpoint:
            edit_policy = edit_policy if edit_policy == "light_edit" else ("rewritable" if not technical_dense else "light_edit")
            block_type = "narrative" if edit_policy == "rewritable" else "light_edit_narration"
            rewrite_depth = "developmental_rewrite" if edit_policy == "rewritable" else "light_edit"
        actions = _deduplicate(
            [
                *actions,
                "chapter_medium_priority_rewrite",
                "sentence_cluster_rewrite",
                "weaken_template_connectors",
                "compress_subject_chain",
            ]
        )
        updated_notes.append(
            f"Chapter-aware policy: {chapter_type} is medium priority; improve flow without drifting technical detail."
        )
        if technical_dense and edit_policy == "rewritable":
            updated_notes.append(
                "Chapter-aware policy: this medium-priority technical paragraph needs developmental flow changes around protected terms, not term substitution."
            )
    elif chapter_priority == "conservative":
        edit_policy = "light_edit"
        block_type = "light_edit_narration"
        rewrite_depth = "light_edit"
        heavy_actions = {
            "chapter_high_priority_rewrite",
            "sentence_cluster_rewrite",
            "sentence_cluster_merge",
            "sentence_cluster_split",
            "narrative_flow_rebuilder",
            "discourse_reordering",
            "narrative_path_rewrite",
            "proposition_reorder",
            "conclusion_absorption",
            "merge_or_split_sentence_cluster",
        }
        actions = [action for action in actions if action not in heavy_actions]
        actions = _deduplicate(
            [
                *actions,
                "chapter_conservative_light_edit",
                "partial_keep",
                "compress_subject_chain",
                "compress_meta_discourse",
                "weaken_template_connectors",
                "keep_original_if_technical_density_is_high",
            ]
        )
        updated_notes.append(
            f"Chapter-aware policy: {chapter_type} is conservative priority; avoid broad sentence-cluster reconstruction."
        )

    return edit_policy, block_type, rewrite_depth, actions, _deduplicate(updated_notes)


def _build_agent_notes(block_policies: list[BlockPolicy], document_scale: str, target_style: str = "academic_natural_studentlike") -> list[str]:
    notes = [
        f"Use the {target_style} profile.",
        "AIRC is not a light polish tool: rewrite ordinary body prose as a human academic editor would rebuild it.",
        "Change sentence groups and the route of explanation; do not merely replace words or sentence openers.",
        "Treat function words such as 的 / 了 / 在……中 and empty verbs such as 进行 / 构建 / 实现 / 提升 as naturalness signals, not mechanical replacement targets.",
        "Break template-like parallelism by merging, reordering, or recasting one sentence; do not replace it with another slogan.",
        "Use block policy as the first gate: do_not_touch blocks must stay verbatim, and high_risk blocks default to keep-original.",
        "When a rewrite only works through a stiff sentence opener, keep the original sentence instead of forcing change.",
        "Preserve explicit subjects when removing them would weaken who performed the action or which system/configuration is being referenced.",
        "Apply chapter-aware rewrite policy: background/significance/review/analysis/conclusion/future-work sections need stronger reconstruction, while formulas, metrics, experiment setup, deployment, and technical-dense method content stay conservative.",
        "Preserve paragraph skeletons: a valid topic sentence may be lightly rewritten, but it must stay at the paragraph opening and must not become a dangling transition.",
        "Add local human revision realism inside each editable paragraph: soften overly explicit sentence transitions, vary cadence, and let support sentences carry different weights.",
    ]
    if target_style == "zh_academic_l2_mild":
        notes.extend(
            [
                "zh_academic_l2_mild means restrained academic prose with mild non-native Chinese texture: slightly more explanatory, less compressed, and a little less native-polished.",
                "For this profile, allow mild 的 / 了 / 来 / 进行 / 能够 / 通过……来 density, but do not create grammar errors, colloquial expressions, or unsupported facts.",
                "Do not apply this profile to formula-dense, path-heavy, checkpoint, citation-heavy, or technical-definition blocks.",
            ]
        )
    if document_scale in {"long", "very_long"}:
        notes.append(
            f"Document scale is {document_scale}: raise rewrite intensity, rewrite a large share of body blocks, and do not fall back to conservative rewriting."
        )
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
                f"Block {policy.block_id} is light_edit with {policy.rewrite_intensity} intensity: apply revision patterns {policy.revision_pattern or ['partial_keep']} and deliver at least {policy.required_minimum_sentence_level_changes} sentence-level change, but do not do paragraph-level reorganization."
            )
        elif policy.edit_policy == "rewritable":
            notes.append(
                f"Block {policy.block_id} is developmental_rewrite with {policy.rewrite_intensity} intensity: apply revision patterns {policy.revision_pattern or ['reframe']} and execute discourse actions {policy.required_discourse_actions or ['sentence_cluster_rewrite']}."
            )
        if policy.should_rewrite:
            notes.append(
                f"Block {policy.block_id} belongs to chapter '{policy.chapter_title}' ({policy.chapter_type}, {policy.chapter_rewrite_priority} priority); quota={policy.chapter_rewrite_quota}."
            )
            if not policy.opening_reorder_allowed:
                notes.append(
                    f"Block {policy.block_id} has a locked topic sentence opening; rewrite support/evidence sentences more than the first sentence and do not reorder the opener."
                )
            if policy.high_sensitivity_prose:
                notes.append(
                    f"Block {policy.block_id} is high-sensitivity prose; sentence completeness and paragraph readability override extra rewrite scale."
                )
                notes.append(
                    f"Block {policy.block_id} should avoid one-pass polishing: keep at least one plain support sentence when the source is already natural, reduce academic cliché density, and do not rewrite every sentence into the same texture."
                )
    return _deduplicate(notes)


def _build_write_gate_preconditions() -> list[str]:
    return [
        "Core content integrity checks must all pass.",
        "Format integrity checks must all pass.",
        "Body-only rewrite coverage must meet the scale-aware quota; headings, images, captions, formulas, references, paths, and technical blocks do not count.",
        "Body changed-block ratio must meet the scale-aware quota for long and very_long documents.",
        "Body discourse-change and cluster-rewrite scores must meet their scale-aware thresholds.",
        "Human revision checks must pass for long/very_long documents: cluster rewrite exists and rewrite distribution is non-uniform.",
        "Chapter rewrite quota must pass: high-priority narrative chapters cannot remain barely changed, and conservative technical chapters cannot be over-rewritten.",
        "Paragraph skeleton checks must pass: valid topic sentences stay at the opening and paragraph openings must not be dangling transition phrases.",
        "Local revision realism checks must pass for long documents: transitions are natural, paragraph-internal discourse is not flat, sentence uniformity is reduced, and revision realism is present.",
        "Author-texture checks should pass for rewritten prose: stylistic uniformity stays controlled, support-sentence texture varies, paragraph voice is not identical across sections, and high-sensitivity prose does not accumulate academic cliché phrasing.",
        "Evidence-fidelity checks should pass for rewritten prose: do not add unsupported domain background, metaphor, external commentary, or unjustified first-person claims.",
        "Sentence completeness and paragraph readability checks must pass: support and conclusion sentences cannot be dangling fragments.",
        "Each rewritable block must satisfy its required_structural_actions.",
        "Each rewritable block must satisfy its required_discourse_actions and rewrite depth minimums.",
        "Light-edit blocks need sentence-level change; developmental-rewrite blocks need cluster-level change.",
        "The rewritten candidate must show effective discourse-level change, not only word substitution.",
        "Natural revision checklist must pass: academic tone, reduced mechanical connectors, reduced repeated subjects, varied rhythm, no colloquial markers.",
        "Severe template risk or repeated-subject risk blocks writing.",
        "At least one high-value structural action must be present before writing.",
    ]


def _local_revision_realism_actions(rewrite_depth: str) -> list[str]:
    """Return local human-revision actions appropriate for the rewrite depth."""

    if rewrite_depth == "light_edit":
        return [
            "soften_overexplicit_transition",
            "light_partial_retain_with_local_rephrase",
            "paragraph_readability_smoothing",
            "de_template_academic_cliche",
            "soft_keep_for_human_revision_feel",
        ]
    if rewrite_depth == "developmental_rewrite":
        return [
            "soften_overexplicit_transition",
            "reduce_sentence_uniformity",
            "introduce_local_hierarchy",
            "reshape_supporting_sentence",
            "weaken_overfinished_sentence",
            "convert_flat_parallel_flow",
            "light_partial_retain_with_local_rephrase",
            "paragraph_readability_smoothing",
            "vary_support_sentence_texture",
            "de_template_academic_cliche",
            "retain_some_plain_sentences",
            "avoid_overpolished_supporting_sentence",
            "introduce_mild_authorial_asymmetry",
            "deuniform_paragraph_texture",
            "rephrase_summary_like_sentence_without_cliche",
            "soft_keep_for_human_revision_feel",
        ]
    return []


def _semantic_role_integrity_actions(
    *,
    text: str,
    block_type: str,
    chapter_type: str,
    high_sensitivity_prose: bool,
) -> list[str]:
    """Return semantic-role-preservation actions for sensitive prose and explicit enumerations."""

    actions: list[str] = []
    if high_sensitivity_prose:
        actions.extend(
            [
                "preserve_semantic_role_of_core_sentence",
                "remove_generated_scaffolding_phrase",
                "replace_abstracted_subject_with_concrete_referent",
                "avoid_huanbaokuo_style_expansion",
            ]
        )
    if _ENUMERATION_RE.search(text) or re.search(r"^(?:[（(]?\d+[）)]|第一|第二|第三)", text):
        actions.extend(
            [
                "preserve_enumeration_item_role",
                "repair_enumeration_flow",
                "avoid_huanbaokuo_style_expansion",
            ]
        )
    if chapter_type in {"mechanism_explanation", "method_design", "training_strategy", "system_workflow"} or re.search(
        r"(?:模型|系统|模块|分支|接口|机制|路径|损失函数|课程学习|判别机制).*(?:采用|通过|将|把|用于|负责|"
        r"生成|输出|融合|提取|建模|约束|对外提供|保留|剥离|实施|构成|形成)",
        text,
    ):
        actions.extend(
            [
                "restore_mechanism_sentence_from_support_like_rewrite",
                "prevent_appendix_like_supporting_sentence",
                "replace_abstracted_subject_with_concrete_referent",
            ]
        )
    if block_type in {"innovation_points", "chapter_summary"}:
        actions.extend(
            [
                "preserve_enumeration_item_role",
                "preserve_semantic_role_of_core_sentence",
            ]
        )
    return _deduplicate(actions)


def _authorial_intent_actions(
    *,
    text: str,
    block_type: str,
    chapter_type: str,
    high_sensitivity_prose: bool,
) -> list[str]:
    """Return authorial-intent actions for direct assertion, concrete subjects, and source stance."""

    actions: list[str] = []
    if high_sensitivity_prose:
        actions.extend(
            [
                "upgrade_appendix_like_sentence_to_assertion",
                "replace_weak_modal_verbs",
                "restore_authorial_choice_expression",
                "reduce_overuse_of_passive_explanations",
            ]
        )
    if chapter_type in {"mechanism_explanation", "method_design", "training_strategy", "system_workflow"} or re.search(
        r"(?:模型|系统|模块|分支|接口|机制|路径|课程学习|判别机制|训练).*(?:用于|可以|能够|有助于|通过)",
        text,
    ):
        actions.extend(
            [
                "upgrade_appendix_like_sentence_to_assertion",
                "strengthen_mechanism_verb",
                "replace_weak_modal_verbs",
                "promote_support_sentence_to_core_if_needed",
            ]
        )
    if re.search(r"(?:相比之下|并非|而是|更关键的是|关键在于|核心在于|选择.+而非.+)", text):
        actions.extend(
            [
                "restore_authorial_choice_expression",
                "reduce_overuse_of_passive_explanations",
            ]
        )
    if block_type in {"innovation_points", "chapter_summary", "conclusion"}:
        actions.extend(
            [
                "restore_authorial_choice_expression",
                "promote_support_sentence_to_core_if_needed",
            ]
        )
    return _deduplicate(actions)


def _evidence_fidelity_actions(
    *,
    text: str,
    block_type: str,
    chapter_type: str,
    high_sensitivity_prose: bool,
) -> list[str]:
    """Return evidence-boundary and thesis-register actions for sensitive academic prose."""

    actions: list[str] = []
    if high_sensitivity_prose:
        actions.extend(
            [
                "remove_unsupported_expansion",
                "remove_external_domain_commentary",
                "remove_metaphoric_storytelling",
                "restore_thesis_register",
                "replace_we_with_original_subject_style",
                "downgrade_overclaimed_judgment",
                "preserve_original_evidence_scope",
            ]
        )
    if block_type in {"innovation_points", "chapter_summary", "conclusion"}:
        actions.extend(
            [
                "restore_thesis_register",
                "preserve_original_evidence_scope",
            ]
        )
    if chapter_type in {
        "mechanism_explanation",
        "result_analysis",
        "conclusion",
        "future_work",
        "training_strategy",
    } or re.search(r"(?:模型|系统|模块|分支|接口|机制|课程学习|判别机制).*(?:通过|将|把|负责|实现|约束)", text):
        actions.extend(
            [
                "restore_mechanism_sentence_to_academic_statement",
                "replace_we_with_original_subject_style",
                "downgrade_overclaimed_judgment",
            ]
        )
    if re.search(r"(?:主流观点|业界通常认为|普遍认为|超过八成|超过半数|绝大多数|如同|仿佛|终于摆脱)", text):
        actions.extend(
            [
                "remove_unsupported_expansion",
                "remove_external_domain_commentary",
                "remove_metaphoric_storytelling",
            ]
        )
    return _deduplicate(actions)


def _academic_sentence_naturalization_actions(
    *,
    text: str,
    block_type: str,
    chapter_type: str,
    high_sensitivity_prose: bool,
) -> list[str]:
    """Return actions that make academic sentences less bureaucratic and less rewriter-shaped."""

    actions: list[str] = []
    if high_sensitivity_prose:
        actions.extend(
            [
                "remove_bureaucratic_opening",
                "compress_explicit_subject_chain",
                "flatten_overstructured_parallelism",
                "advance_main_clause",
                "remove_slogan_like_goal_phrase",
                "convert_project_style_opening_to_academic_statement",
                "enforce_direct_statement",
                "remove_academic_wrapping",
                "reduce_connectors",
                "convert_passive_to_active",
                "split_overlong_sentence",
                "diversify_subject",
            ]
        )
    if re.search(r"(?:本研究面向.+需求|本研究的主题为|在方法上|围绕这一目标|在.+目标下|在完成.+后|在.+层面|具体而言|从.+角度|就.+而言|在.+过程中)", text):
        actions.extend(
            [
                "remove_bureaucratic_opening",
                "convert_project_style_opening_to_academic_statement",
                "decompress_overpacked_modifier_prefix",
                "enforce_direct_statement",
                "move_main_clause_forward",
            ]
        )
    if re.search(r"(?:旨在构建|旨在|以实现.+落地|从而实现|形成完整闭环|具有显著.+价值|提供清晰方法论路径|提供.+基础|形成.+体系|提供.+路径)", text):
        actions.extend(
            [
                "remove_slogan_like_goal_phrase",
                "replace_theme_is_with_direct_topic_statement",
                "remove_academic_wrapping",
            ]
        )
    if re.search(r"(?:并不是.+而是|不仅.+(?:同时|还)|在结构上.+在决策层面|第一.+进而第二)", text):
        actions.extend(["flatten_overstructured_parallelism", "denormalize_parallel_structure"])
    if re.search(r"(?:并且|从而|进而|同时)", text):
        actions.append("reduce_connectors")
    if re.search(r"(?:被视为|属于|得到体现)", text):
        actions.append("convert_passive_to_active")
    if re.search(r"(?:进行(?:评估|分析|验证|控制|处理)|实现(?:控制|部署|融合|落地)|提供(?:支撑|支持))", text):
        actions.append("remove_academic_wrapping")
    if len(re.sub(r"\s+", "", text)) > 110:
        actions.append("split_overlong_sentence")
    if chapter_type in {"background", "significance", "literature_review", "conclusion", "future_work"}:
        actions.extend(
            [
                "compress_explicit_subject_chain",
                "replace_theme_is_with_direct_topic_statement",
                "diversify_subject",
            ]
        )
    return _deduplicate(actions)


def _is_high_sensitivity_prose_block(
    *,
    text: str,
    block_type: str,
    chapter_type: str,
    chapter_title: str,
) -> bool:
    """Identify prose blocks where readability should override extra rewrite scale."""

    if block_type in {
        "abstract_zh",
        "research_significance",
        "innovation_points",
        "conclusion",
        "future_work",
        "chapter_summary",
    }:
        return True
    title = re.sub(r"\s+", "", chapter_title or "")
    if re.search(r"(摘要|研究意义|创新|总结|结论|展望|本章小结)", title):
        return True
    if chapter_type in {"significance", "conclusion", "future_work"}:
        return True
    if chapter_type in {"literature_review", "result_analysis"} and len(split_sentences(text)) >= 3:
        return True
    return bool(re.search(r"(研究意义|创新点|总结而言|综上|未来工作|本文完成|本研究的核心价值)", text))


def _apply_document_scale_pressure(block_policies: list[BlockPolicy], document_scale: str) -> list[BlockPolicy]:
    if document_scale not in {"long", "very_long"}:
        return block_policies

    pressured: list[BlockPolicy] = []
    for block in block_policies:
        if not block.should_rewrite or block.edit_policy not in {"light_edit", "rewritable"}:
            pressured.append(block)
            continue

        notes = list(block.notes)
        recommended = list(block.recommended_actions)
        structural = list(block.required_structural_actions)
        discourse = list(block.required_discourse_actions)
        edit_policy = block.edit_policy
        block_type = block.block_type
        rewrite_depth = block.rewrite_depth
        min_sentence = block.required_minimum_sentence_level_changes
        min_cluster = block.required_minimum_cluster_changes
        intensity = "high" if block.rewrite_depth == "developmental_rewrite" else "medium"
        if block.chapter_rewrite_priority == "medium" and block.edit_policy == "light_edit":
            edit_policy = "rewritable"
            block_type = "narrative"
            rewrite_depth = "developmental_rewrite"
            intensity = "medium"
            recommended = _deduplicate(
                [
                    *recommended,
                    "chapter_medium_long_document_developmental_rewrite",
                    "sentence_cluster_rewrite",
                    "narrative_path_rewrite",
                    "keep_original_if_technical_density_is_high",
                ]
            )
            notes.append(
                f"Document-scale pressure ({document_scale}): medium-priority chapter block is upgraded from light edit to controlled developmental rewrite."
            )
        if block.chapter_rewrite_priority == "medium" and rewrite_depth == "developmental_rewrite":
            intensity = "medium"

        if block.chapter_rewrite_priority == "conservative":
            intensity = "light"
            recommended = _deduplicate(
                [
                    *recommended,
                    "chapter_conservative_light_edit",
                    "partial_keep",
                    "compress_subject_chain",
                    "compress_meta_discourse",
                    "weaken_template_connectors",
                    "keep_original_if_technical_density_is_high",
                ]
            )
            discourse = [
                action
                for action in discourse
                if action
                not in {
                    "sentence_cluster_rewrite",
                    "sentence_cluster_merge",
                    "sentence_cluster_split",
                    "proposition_reorder",
                    "discourse_reordering",
                    "narrative_path_rewrite",
                    "conclusion_absorption",
                }
            ]
            structural = [
                action
                for action in structural
                if action not in {"pair_fusion", "sentence_cluster_merge", "sentence_cluster_split"}
            ]
            min_cluster = 0
            notes.append("Chapter-aware override: conservative technical chapter remains light even in a long document.")
        elif rewrite_depth == "developmental_rewrite":
            recommended = _deduplicate(
                [
                    *recommended,
                    "sentence_cluster_rewrite",
                    "narrative_flow_rebuilder",
                    "sentence_cluster_merge",
                    "discourse_reordering",
                    "narrative_path_rewrite",
                    "uneven_rewrite_distribution",
                    "proposition_reorder",
                    "transition_absorption",
                    *_local_revision_realism_actions("developmental_rewrite"),
                ]
            )
            discourse = _deduplicate(
                [
                    *discourse,
                    "sentence_cluster_rewrite",
                    "sentence_cluster_merge",
                    "discourse_reordering",
                    "narrative_path_rewrite",
                    "uneven_rewrite_distribution",
                    "proposition_reorder",
                ]
            )
            if len(split_sentences(block.original_text)) >= 2:
                structural = _deduplicate([*structural, "pair_fusion"])
            min_sentence = max(min_sentence, DEFAULT_CONFIG.developmental_min_sentence_level_changes)
            if block.chapter_rewrite_priority == "medium":
                min_cluster = max(min_cluster, 0)
            else:
                min_cluster = max(min_cluster, DEFAULT_CONFIG.developmental_min_cluster_changes)
            notes.append(
                f"Document-scale pressure ({document_scale}): developmental body block must be rewritten, not lightly polished."
            )
        else:
            recommended = _deduplicate(
                [
                    *recommended,
                    "sentence_level_recast",
                    "light_clause_reorder",
                    *_local_revision_realism_actions("light_edit"),
                ]
            )
            discourse = _deduplicate([*discourse, "sentence_level_recast"])
            min_sentence = max(min_sentence, DEFAULT_CONFIG.light_edit_min_sentence_level_changes)
            notes.append(
                f"Document-scale pressure ({document_scale}): light-edit body block still needs visible sentence-level recasting."
            )

        pressured.append(
            BlockPolicy(
                block_id=block.block_id,
                block_type=block_type,
                risk_level=block.risk_level,
                edit_policy=edit_policy,
                rewrite_depth=rewrite_depth,
                rewrite_intensity=intensity,
                preview=block.preview,
                protected_items=list(block.protected_items),
                recommended_actions=recommended,
                required_structural_actions=structural,
                required_discourse_actions=discourse,
                required_minimum_sentence_level_changes=min_sentence,
                required_minimum_cluster_changes=min_cluster,
                optional_actions=[
                    action for action in recommended if action not in structural and action not in discourse
                ],
                forbidden_actions=list(block.forbidden_actions),
                notes=_deduplicate(notes),
                original_text=block.original_text,
                should_rewrite=block.should_rewrite,
                revision_pattern=list(block.revision_pattern),
                chapter_id=block.chapter_id,
                chapter_title=block.chapter_title,
                chapter_type=block.chapter_type,
                chapter_rewrite_priority=block.chapter_rewrite_priority,
                chapter_rewrite_quota=dict(block.chapter_rewrite_quota),
                chapter_rewrite_intensity=intensity if block.should_rewrite else block.chapter_rewrite_intensity,
                paragraph_sentence_roles=list(block.paragraph_sentence_roles),
                opening_rewrite_allowed=block.opening_rewrite_allowed,
                opening_reorder_allowed=block.opening_reorder_allowed,
                topic_sentence_text=block.topic_sentence_text,
                high_sensitivity_prose=block.high_sensitivity_prose,
            )
        )

    return pressured


def _determine_required_structural_actions(text: str, edit_policy: str, repeated_subject: bool) -> list[str]:
    if edit_policy not in {"light_edit", "rewritable"}:
        return []

    requirements: list[str] = []
    sentences = split_sentences(text)
    if edit_policy == "rewritable":
        if len(sentences) >= 2:
            requirements.append("pair_fusion")
            requirements.append("sentence_cluster_merge")
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
    function_word_dense: bool,
    parallelism: bool,
    dense_nominal: bool,
    rhythm_flat: bool,
) -> list[str]:
    if rewrite_depth not in {"light_edit", "developmental_rewrite"}:
        return []

    actions: list[str] = []
    sentences = split_sentences(text)
    if repeated_subject:
        actions.extend(["subject_chain_compression", "compress_subject_chain"])
    if meta_dense:
        actions.append("meta_compression")
    if any(_IMPLICATION_HEAD_RE.match(sentence.strip()) for sentence in sentences):
        actions.append("conclusion_absorb")
    if _ENUMERATION_RE.search(text):
        actions.append("enumeration_reframe")
    if template_dense:
        actions.extend(["transition_absorption", "weaken_template_connectors"])
    if function_word_dense:
        actions.append("reduce_function_word_overuse")
    if dense_nominal:
        actions.append("rewrite_dense_nominal_phrases")
    if rhythm_flat:
        actions.append("rebuild_sentence_rhythm")
    if parallelism:
        actions.append("break_parallelism")

    if rewrite_depth == "light_edit":
        if not actions:
            actions.append("sentence_level_recast")
        actions.append("uneven_rewrite_distribution")
        return _deduplicate(actions)

    if len(sentences) >= 2:
        actions.append("sentence_cluster_rewrite")
        actions.append("sentence_cluster_merge")
    if len(sentences) >= 3:
        actions.append("proposition_reorder")
        actions.append("discourse_reordering")
        actions.append("narrative_path_rewrite")
    if re.search(r"(原因|因为|导致|使得|使其|关键在于|更在于|风险)", text):
        actions.append("rationale_expansion")
    return _deduplicate(actions)


def _determine_revision_pattern(
    *,
    index: int,
    text: str,
    edit_policy: str,
    rewrite_depth: str,
    repeated_subject: bool,
    template_dense: bool,
    rhythm_flat: bool,
    parallelism: bool,
) -> list[str]:
    if edit_policy == "light_edit":
        patterns = ["partial_keep"]
        if template_dense:
            patterns.append("soften")
        elif rhythm_flat:
            patterns.append("split")
        else:
            patterns.append("compress")
        return patterns[:2]

    sentences = split_sentences(text)
    options: list[str] = []
    if repeated_subject:
        options.append("compress")
    if len(sentences) >= 3 and index % 3 == 0:
        options.append("reorder")
    if len(sentences) >= 2 and index % 2 == 0:
        options.append("merge")
    if len(sentences) >= 4 and rhythm_flat:
        options.append("split")
    if template_dense:
        options.append("soften")
    if parallelism:
        options.append("reframe")
    if not options:
        options.append("reframe" if rewrite_depth == "developmental_rewrite" else "partial_keep")
    if index % 5 == 0:
        options.append("partial_keep")
    elif index % 7 == 0:
        options.append("rewrite_all")
    elif len(sentences) <= 2:
        options.append("expand")
    return _deduplicate(options)[:2]


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
    function_word_dense: bool = False,
    parallelism: bool = False,
    dense_nominal: bool = False,
    rhythm_flat: bool = False,
) -> str:
    if rewrite_depth != "developmental_rewrite":
        return "light"
    sentence_count = len(split_sentences(text))
    if sentence_count >= 3 and re.search(r"(背景|意义|风险|分析|挑战|问题|价值|影响|趋势|现状)", text):
        return "high"
    if sentence_count >= 3 and (
        repeated_subject or meta_dense or template_dense or function_word_dense or parallelism or dense_nominal or rhythm_flat
    ):
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


def _function_word_density(text: str) -> int:
    return len(_FUNCTION_WORD_RE.findall(text))


def _stiff_verb_density(text: str) -> int:
    return len(_STIFF_VERB_RE.findall(text))


def _has_template_parallelism(text: str) -> bool:
    if _PARALLELISM_RE.search(text):
        return True
    sentences = split_sentences(text)
    if len(sentences) < 3:
        return False
    openers = [sentence.strip()[:5] for sentence in sentences if sentence.strip()]
    return len(openers) >= 3 and len(set(openers[:3])) == 1


def _has_dense_nominal_phrase(text: str) -> bool:
    return bool(_DENSE_NOMINAL_RE.search(text)) or text.count("的") >= 5


def _has_flat_sentence_rhythm(text: str) -> bool:
    sentences = [sentence for sentence in split_sentences(text) if sentence.strip()]
    if len(sentences) < 3:
        return False
    lengths = [len(re.sub(r"\s+", "", sentence)) for sentence in sentences]
    return max(lengths) - min(lengths) <= 10


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
    ascii_letters = sum(1 for char in stripped if char.isascii() and char.isalpha())
    visible = sum(1 for char in stripped if not char.isspace())
    if visible == 0:
        return False
    if len(english_words) < 4 and ascii_letters < 24:
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
