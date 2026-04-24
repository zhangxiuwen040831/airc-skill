from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

from .body_metrics import body_stats_only, compute_body_rewrite_metrics, strip_non_body_markdown_lines
from .chapter_policy import chapter_quota_reason_codes, compute_chapter_rewrite_metrics
from .config import DEFAULT_CONFIG, RewriteMode
from .core_guard import (
    compare_core_snapshots,
    narrow_do_not_touch_scope,
    normalize_numeric_tokens_for_integrity,
    protected_segments_for_document,
    snapshot_core_content,
)
from .authorial_intent import aggregate_authorial_intent
from .academic_sentence_naturalization import aggregate_academic_sentence_naturalization
from .evidence_fidelity import aggregate_evidence_fidelity
from .l2_style_profile import aggregate_l2_style_profile
from .local_revision_realism import aggregate_local_revision_realism
from .models import GuidanceReport, ReviewReport, RewriteCandidate as BlockRewriteCandidate
from .natural_revision_profile import COLLOQUIAL_FORBIDDEN_MARKERS
from .paragraph_skeleton import document_paragraph_skeleton_review
from .rewriter import RewriteStats, split_sentences
from .semantic_role_integrity import aggregate_semantic_role_integrity
from .sentence_readability import aggregate_sentence_readability
from .target_style_alignment import analyze_target_style_alignment

_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
_FENCED_CODE_RE = re.compile(
    r"(?ms)^(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^\s*(?P=fence)[ \t]*$",
    re.MULTILINE,
)
_INLINE_LINK_RE = re.compile(r"(?<!!)\[[^\]]+]\([^)]+\)")
_AUTOLINK_RE = re.compile(r"<https?://[^>\n]+>")
_BARE_URL_RE = re.compile(r"https?://[^\s<>()]+(?:\([^\s<>()]*\)[^\s<>()]*)*")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_NORMALIZE_RE = re.compile(r"\s+")
_STIFF_OPENERS = ("本研究", "本文", "从", "对", "这也意味着", "基于此", "在这种情况下", "总体来看", "综合来看")
_META_SUBJECTS = ("本研究", "本文", "该研究", "该系统")
_LEADING_IMPLICATION_RE = re.compile(r"^(?:因此|由此|这也意味着|基于此|在这种情况下|正因为如此|在此基础上)[，,\s]*")
_CORE_INTEGRITY_KEYS = {
    "title_integrity_check",
    "formula_integrity_check",
    "citation_integrity_check",
    "terminology_integrity_check",
    "numeric_integrity_check",
    "path_integrity_check",
}
_FORMAT_INTEGRITY_KEYS = {
    "heading_format_integrity_check",
    "english_spacing_integrity_check",
    "placeholder_integrity_check",
    "caption_punctuation_integrity_check",
    "markdown_symbol_integrity_check",
    "linebreak_whitespace_integrity_check",
}
_SUBJECT_FIX_ACTIONS = {
    "merge_consecutive_subject_sentences",
    "subject_drop",
    "subject_variation",
    "meta_compression",
    "followup_absorb",
    "conclusion_absorb",
}
_STRICT_TEMPLATE_MARKERS = ("从", "角度看", "在此基础上", "这也意味着", "进一步来看", "综上")
_EMPTY_ACADEMIC_VERBS = ("进行", "构建", "实现", "确保", "提升", "开展", "推动")
_FUNCTION_WORD_RE = re.compile(r"(进行了|进行|在[^，。；]{2,20}(?:中|过程中|背景下)|的[^，。；]{1,12}的)")
_PARALLELISM_MARKERS = ("不仅要", "还要", "奠定坚实基础", "有力支撑", "完整闭环", "有效提升")
_REQUIRED_ACTION_ALIASES = {
    "pair_fusion": {"pair_fusion", "sentence_merge", "clause_reorder"},
    "conclusion_absorb": {"conclusion_absorb", "followup_absorb"},
    "subject_chain_compression": {
        "subject_chain_compression",
        "merge_consecutive_subject_sentences",
        "subject_drop",
        "subject_variation",
        "meta_compression",
        "followup_absorb",
    },
    "enumeration_reframe": {"enumeration_reframe"},
    "paragraph_reorder": {"paragraph_reorder"},
    "clause_reorder": {"clause_reorder"},
    "sentence_merge": {"sentence_merge", "pair_fusion"},
}

@dataclass(frozen=True)
class NaturalnessAssessment:
    depth_sufficient: bool
    template_risk: bool
    template_warning: bool
    template_issue: str | None
    naturalness_risk: bool
    structural_action_count: int
    high_value_action_count: int
    high_value_actions: list[str]
    discourse_change_score: int
    cluster_rewrite_score: int
    style_variation_score: int
    prefix_only_rewrite: bool
    repeated_subject_risk: bool
    repeated_subject_warning: bool
    reasons: list[str]
    warnings: list[str]


def review_rewrite(
    original: str,
    revised: str,
    guidance: GuidanceReport | None = None,
    mode: RewriteMode = RewriteMode.BALANCED,
    rewrite_stats: list[RewriteStats] | None = None,
    block_candidates: list[BlockRewriteCandidate] | None = None,
    suffix: str = ".txt",
    target_style_text: str | None = None,
) -> ReviewReport:
    stats_list = rewrite_stats or []
    candidates = block_candidates or []
    core_problems: list[str] = []
    format_problems: list[str] = []
    reject_reasons: list[str] = []
    warnings: list[str] = []
    failed_block_ids: list[int] = []

    if not revised.strip():
        reject_reasons.append("Rewritten text is empty.")

    if set(normalize_numeric_tokens_for_integrity(original)) - set(normalize_numeric_tokens_for_integrity(revised)):
        core_problems.append("Numeric values changed.")

    if _count_headings(original) != _count_headings(revised):
        format_problems.append("Heading count changed.")

    if _count_code_blocks(original) != _count_code_blocks(revised):
        format_problems.append("Code block count changed.")

    if _count_links(original) != _count_links(revised):
        format_problems.append("Link count changed.")

    if revised.count("；") > original.count("；"):
        reject_reasons.append("Unexpected Chinese semicolons were introduced.")

    integrity_checks = compare_core_snapshots(
        snapshot_core_content(original, suffix=suffix),
        snapshot_core_content(revised, suffix=suffix),
    )
    if guidance is not None and not _do_not_touch_blocks_preserved(revised, guidance):
        format_problems.append("Guidance-marked do_not_touch blocks changed.")

    for check_name, passed in integrity_checks.items():
        if not passed:
            if check_name in _CORE_INTEGRITY_KEYS:
                core_problems.append(f"{check_name} failed.")
            elif check_name in _FORMAT_INTEGRITY_KEYS:
                format_problems.append(f"{check_name} failed.")

    changed_characters, diff_spans = _measure_diff(original, revised)
    sentence_level_change = _has_sentence_level_change(original, revised)
    body_metrics = compute_body_rewrite_metrics(
        original=original,
        revised=revised,
        guidance=guidance,
        mode=mode,
        rewrite_stats=stats_list,
        suffix=suffix,
    )
    chapter_rewrite_metrics = compute_chapter_rewrite_metrics(
        guidance=guidance,
        rewrite_stats=stats_list,
        original=original,
        revised=revised,
        suffix=suffix,
    )
    chapter_rewrite_quota_reason_codes = chapter_quota_reason_codes(chapter_rewrite_metrics)
    chapter_rewrite_quota_check = not chapter_rewrite_quota_reason_codes
    chapter_policy_consistency_check = all(
        bool(metric.get("chapter_policy_consistent", True)) for metric in chapter_rewrite_metrics
    )
    analysis_original = body_metrics.body_original_text or strip_non_body_markdown_lines(original)
    analysis_revised = body_metrics.body_revised_text or strip_non_body_markdown_lines(revised)
    target_style_alignment = analyze_target_style_alignment(
        model_output=revised,
        target_text=target_style_text,
        source_text=original,
    )
    analysis_stats = body_stats_only(stats_list, body_metrics.body_block_ids) if guidance is not None else stats_list
    assessment = _review_naturalness(
        original=analysis_original,
        revised=analysis_revised,
        mode=mode,
        rewrite_stats=analysis_stats,
    )
    human_revision = _assess_human_revision_behavior(
        original=analysis_original,
        revised=analysis_revised,
        rewrite_stats=analysis_stats,
        block_candidates=candidates,
    )
    local_realism = _assess_local_revision_realism(analysis_stats)
    local_transition_natural = bool(local_realism["local_transition_natural"])
    local_discourse_not_flat = bool(local_realism["local_discourse_not_flat"])
    sentence_uniformity_reduced = bool(local_realism["sentence_uniformity_reduced"])
    revision_realism_present = bool(local_realism["revision_realism_present"])
    stylistic_uniformity_controlled = bool(local_realism["stylistic_uniformity_controlled"])
    support_sentence_texture_varied = bool(local_realism["support_sentence_texture_varied"])
    paragraph_voice_variation_present = bool(local_realism["paragraph_voice_variation_present"])
    academic_cliche_density_controlled = bool(local_realism["academic_cliche_density_controlled"])
    uniform_paragraph_ids = list(local_realism.get("uniform_paragraph_ids", []))
    low_texture_variation_paragraph_ids = list(local_realism.get("low_texture_variation_paragraph_ids", []))
    high_sensitivity_cliche_paragraph_ids = list(local_realism.get("high_sensitivity_cliche_paragraph_ids", []))
    sentence_readability = _assess_sentence_readability(analysis_stats)
    sentence_completeness_preserved = bool(sentence_readability["sentence_completeness_preserved"])
    paragraph_readability_preserved = bool(sentence_readability["paragraph_readability_preserved"])
    no_dangling_support_sentences = bool(sentence_readability["no_dangling_support_sentences"])
    no_fragment_like_conclusion_sentences = bool(sentence_readability["no_fragment_like_conclusion_sentences"])
    semantic_role = _assess_semantic_role_integrity(analysis_stats)
    semantic_role_integrity_preserved = bool(semantic_role["semantic_role_integrity_preserved"])
    enumeration_integrity_preserved = bool(semantic_role["enumeration_integrity_preserved"])
    scaffolding_phrase_density_controlled = bool(semantic_role["scaffolding_phrase_density_controlled"])
    over_abstracted_subject_risk_controlled = bool(semantic_role["over_abstracted_subject_risk_controlled"])
    semantic_role_drift_paragraph_ids = list(semantic_role.get("semantic_role_drift_paragraph_ids", []))
    enumeration_drift_paragraph_ids = list(semantic_role.get("enumeration_drift_paragraph_ids", []))
    high_sensitivity_scaffolding_paragraph_ids = list(
        semantic_role.get("high_sensitivity_scaffolding_paragraph_ids", [])
    )
    abstracted_subject_paragraph_ids = list(semantic_role.get("abstracted_subject_paragraph_ids", []))
    authorial_intent = _assess_authorial_intent(analysis_stats)
    assertion_strength_preserved = bool(authorial_intent["assertion_strength_preserved"])
    appendix_like_support_controlled = bool(authorial_intent["appendix_like_support_controlled"])
    authorial_stance_present = bool(authorial_intent["authorial_stance_present"])
    weak_assertion_paragraph_ids = list(authorial_intent.get("weak_assertion_paragraph_ids", []))
    appendix_like_paragraph_ids = list(authorial_intent.get("appendix_like_paragraph_ids", []))
    stance_drop_paragraph_ids = list(authorial_intent.get("stance_drop_paragraph_ids", []))
    high_sensitivity_appendix_like_paragraph_ids = list(
        authorial_intent.get("high_sensitivity_appendix_like_paragraph_ids", [])
    )
    evidence_fidelity = _assess_evidence_fidelity(analysis_stats)
    evidence_fidelity_preserved = bool(evidence_fidelity["evidence_fidelity_preserved"])
    unsupported_expansion_controlled = bool(evidence_fidelity["unsupported_expansion_controlled"])
    thesis_tone_restrained = bool(evidence_fidelity["thesis_tone_restrained"])
    metaphor_or_storytelling_controlled = bool(evidence_fidelity["metaphor_or_storytelling_controlled"])
    authorial_claim_risk_controlled = bool(evidence_fidelity["authorial_claim_risk_controlled"])
    evidence_drift_paragraph_ids = list(evidence_fidelity.get("evidence_drift_paragraph_ids", []))
    unsupported_expansion_paragraph_ids = list(evidence_fidelity.get("unsupported_expansion_paragraph_ids", []))
    high_sensitivity_unsupported_paragraph_ids = list(
        evidence_fidelity.get("high_sensitivity_unsupported_paragraph_ids", [])
    )
    metaphor_paragraph_ids = list(evidence_fidelity.get("metaphor_paragraph_ids", []))
    authorial_claim_paragraph_ids = list(evidence_fidelity.get("authorial_claim_paragraph_ids", []))
    sentence_naturalization = _assess_academic_sentence_naturalization(analysis_stats)
    l2_profile_enabled = bool(
        guidance
        and guidance.naturalness_review.get("style_profile") == "zh_academic_l2_mild"
    )
    l2_style_profile = aggregate_l2_style_profile(analysis_stats, enabled=l2_profile_enabled)
    if l2_profile_enabled:
        l2_style_profile = {
            **l2_style_profile,
            "fact_scope_preserved": not core_problems,
            "technical_terms_preserved": integrity_checks.get("terminology_integrity_check", True),
        }
    l2_texture_present = bool(l2_style_profile.get("l2_texture_present", True))
    l2_not_too_native_like = bool(l2_style_profile.get("not_too_native_like", True))
    l2_not_colloquial = bool(l2_style_profile.get("not_colloquial", True))
    l2_not_ungrammatical = bool(l2_style_profile.get("not_ungrammatical", True))
    bureaucratic_opening_controlled = bool(sentence_naturalization["bureaucratic_opening_controlled"])
    explicit_subject_chain_controlled = bool(sentence_naturalization["explicit_subject_chain_controlled"])
    overstructured_syntax_controlled = bool(sentence_naturalization["overstructured_syntax_controlled"])
    main_clause_position_reasonable = bool(sentence_naturalization["main_clause_position_reasonable"])
    slogan_like_goal_phrase_controlled = bool(sentence_naturalization["slogan_like_goal_phrase_controlled"])
    bureaucratic_paragraph_ids = list(sentence_naturalization.get("bureaucratic_paragraph_ids", []))
    repeated_subject_paragraph_ids = list(sentence_naturalization.get("repeated_subject_paragraph_ids", []))
    overstructured_paragraph_ids = list(sentence_naturalization.get("overstructured_paragraph_ids", []))
    delayed_main_clause_paragraph_ids = list(sentence_naturalization.get("delayed_main_clause_paragraph_ids", []))
    slogan_like_goal_paragraph_ids = list(sentence_naturalization.get("slogan_like_goal_paragraph_ids", []))
    author_style_alignment_controlled = bool(sentence_naturalization["author_style_alignment_controlled"])
    low_directness_paragraph_ids = list(sentence_naturalization.get("low_directness_paragraph_ids", []))
    connector_overuse_paragraph_ids = list(sentence_naturalization.get("connector_overuse_paragraph_ids", []))
    nominalization_paragraph_ids = list(sentence_naturalization.get("nominalization_paragraph_ids", []))
    passive_voice_paragraph_ids = list(sentence_naturalization.get("passive_voice_paragraph_ids", []))
    overlong_sentence_paragraph_ids = list(sentence_naturalization.get("overlong_sentence_paragraph_ids", []))
    subject_monotony_paragraph_ids = list(sentence_naturalization.get("subject_monotony_paragraph_ids", []))
    paragraph_skeleton_review = document_paragraph_skeleton_review(
        original=original,
        revised=revised,
        guidance=guidance,
        rewrite_stats=stats_list,
        suffix=suffix,
    )
    paragraph_topic_sentence_preserved = bool(
        paragraph_skeleton_review["paragraph_topic_sentence_preserved"]
    )
    paragraph_opening_style_valid = bool(paragraph_skeleton_review["paragraph_opening_style_valid"])
    paragraph_skeleton_consistent = bool(paragraph_skeleton_review["paragraph_skeleton_consistent"])
    no_dangling_opening_sentence = bool(paragraph_skeleton_review["no_dangling_opening_sentence"])
    topic_sentence_not_demoted_to_mid_paragraph = bool(
        paragraph_skeleton_review["topic_sentence_not_demoted_to_mid_paragraph"]
    )
    rewrite_coverage = (
        body_metrics.body_rewrite_coverage
        if body_metrics.body_blocks_total
        else _compute_rewrite_coverage_metric(analysis_original, analysis_revised, analysis_stats)
    )
    high_value_actions = assessment.high_value_actions or _collect_high_value_actions(stats_list)
    meaningful_change = _has_meaningful_change(
        original=analysis_original,
        revised=analysis_revised,
        mode=mode,
        changed_characters=changed_characters,
        diff_spans=diff_spans,
        sentence_level_change=sentence_level_change,
        structural_action_count=assessment.structural_action_count,
        discourse_change_score=body_metrics.body_discourse_change_score,
        cluster_rewrite_score=body_metrics.body_cluster_rewrite_score,
        rewrite_coverage=rewrite_coverage,
        prefix_only_rewrite=assessment.prefix_only_rewrite,
        depth_sufficient=assessment.depth_sufficient,
    )
    effective_change = meaningful_change and (
        sentence_level_change
        or assessment.structural_action_count >= 1
        or assessment.discourse_change_score >= DEFAULT_CONFIG.light_edit_min_discourse_score
        or changed_characters >= (
            DEFAULT_CONFIG.strong_min_changed_chars if mode is RewriteMode.STRONG else DEFAULT_CONFIG.balanced_min_changed_chars
        )
    )

    substantive_body_rewrite = (
        body_metrics.rewrite_quota_met
        and body_metrics.body_rewrite_coverage >= body_metrics.required_body_rewrite_coverage
        and body_metrics.body_discourse_change_score >= body_metrics.required_body_discourse_change_score
        and body_metrics.body_cluster_rewrite_score >= body_metrics.required_body_cluster_rewrite_score
    )

    if body_metrics.rewrite_quota_reason_codes:
        reject_reasons.extend(body_metrics.rewrite_quota_reason_codes)
    if chapter_rewrite_quota_reason_codes:
        reject_reasons.extend(chapter_rewrite_quota_reason_codes)
    if not chapter_policy_consistency_check:
        reject_reasons.append("chapter_policy_consistency_failed")
    if body_metrics.document_scale in {"long", "very_long"}:
        if not human_revision["sentence_cluster_changes_present"]:
            reject_reasons.append("sentence_cluster_changes_missing_for_long_document")
        if not human_revision["human_like_variation"]:
            reject_reasons.append("human_like_variation_missing_for_long_document")
        if not human_revision["non_uniform_rewrite_distribution"]:
            reject_reasons.append("non_uniform_rewrite_distribution_missing_for_long_document")
        if not paragraph_topic_sentence_preserved:
            reject_reasons.append("paragraph_topic_sentence_not_preserved_for_long_document")
        if not paragraph_opening_style_valid:
            reject_reasons.append("paragraph_opening_style_invalid_for_long_document")
        if not paragraph_skeleton_consistent:
            reject_reasons.append("paragraph_skeleton_inconsistent_for_long_document")
        if not no_dangling_opening_sentence:
            reject_reasons.append("dangling_paragraph_opening_sentence_for_long_document")
        if not topic_sentence_not_demoted_to_mid_paragraph:
            reject_reasons.append("topic_sentence_demoted_to_mid_paragraph_for_long_document")
        if not local_transition_natural:
            reject_reasons.append("local_transition_rigidity_for_long_document")
        if not local_discourse_not_flat:
            reject_reasons.append("local_discourse_flatness_for_long_document")
        if not sentence_uniformity_reduced:
            reject_reasons.append("sentence_uniformity_not_reduced_for_long_document")
        if not revision_realism_present:
            reject_reasons.append("revision_realism_missing_for_long_document")
        if not sentence_completeness_preserved:
            reject_reasons.append("sentence_completeness_failed_for_long_document")
        if not paragraph_readability_preserved:
            reject_reasons.append("paragraph_readability_failed_for_long_document")
        if not no_dangling_support_sentences:
            reject_reasons.append("dangling_sentence_risk_too_high_for_long_document")
        if not no_fragment_like_conclusion_sentences:
            reject_reasons.append("fragment_like_conclusion_sentences_for_long_document")
        if not semantic_role_integrity_preserved:
            reject_reasons.append("semantic_role_integrity_failed_for_long_document")
        if not enumeration_integrity_preserved:
            reject_reasons.append("enumeration_integrity_failed_for_long_document")
        if len(high_sensitivity_scaffolding_paragraph_ids) >= 2:
            reject_reasons.append("high_sensitivity_scaffolding_phrase_residue_for_long_document")
        if (
            len(high_sensitivity_cliche_paragraph_ids) >= 2
            and not l2_profile_enabled
            and not target_style_alignment.get("enabled", False)
        ):
            reject_reasons.append("high_sensitivity_academic_cliche_residue_for_long_document")
        if float(authorial_intent.get("appendix_like_support_ratio", 0.0)) > 0.14:
            reject_reasons.append("appendix_like_support_ratio_too_high_for_long_document")
        if float(authorial_intent.get("assertion_strength_score", 0.0)) < 0.52:
            reject_reasons.append("assertion_strength_too_low_for_long_document")
        if float(evidence_fidelity.get("unsupported_expansion_risk", 0.0)) > 0.08:
            reject_reasons.append("unsupported_expansion_risk_too_high_for_long_document")
        if float(evidence_fidelity.get("evidence_fidelity_score", 0.0)) < 0.66:
            reject_reasons.append("evidence_fidelity_too_low_for_long_document")
        if not stylistic_uniformity_controlled:
            warnings.append(
                f"Stylistic uniformity remains visible across rewritten paragraphs: {uniform_paragraph_ids[:8]}"
            )
        if not support_sentence_texture_varied:
            warnings.append(
                "Support-sentence texture remains too uniform across rewritten body paragraphs."
            )
        if not paragraph_voice_variation_present:
            warnings.append("Paragraph voice variation remains narrow for a long-document author-like pass.")
        if not academic_cliche_density_controlled:
            warnings.append("Academic cliché density remains too visible in rewritten prose.")
        if not scaffolding_phrase_density_controlled:
            warnings.append("Generated scaffolding phrases remain too visible in rewritten prose.")
        if not over_abstracted_subject_risk_controlled:
            warnings.append(
                "Abstracted subjects still replace concrete research referents in some rewritten paragraphs."
            )
        if not assertion_strength_preserved:
            warnings.append(
                "Assertion strength still feels weaker than the source in some rewritten paragraphs: "
                f"{weak_assertion_paragraph_ids[:8]}"
            )
        if not appendix_like_support_controlled:
            warnings.append(
                "Appendix-like supporting sentences remain visible after revision: "
                f"{appendix_like_paragraph_ids[:8]}"
            )
        if not authorial_stance_present:
            warnings.append(
                "Source contrast or choice markers were flattened in some paragraphs: "
                f"{stance_drop_paragraph_ids[:8]}"
            )
        if not evidence_fidelity_preserved:
            warnings.append(
                "Some rewritten paragraphs drift outside the source evidence boundary: "
                f"{evidence_drift_paragraph_ids[:8]}"
            )
        if not unsupported_expansion_controlled:
            warnings.append(
                "Unsupported externalized expansion remains visible after revision: "
                f"{unsupported_expansion_paragraph_ids[:8]}"
            )
        if not thesis_tone_restrained:
            warnings.append("Some rewritten prose still sounds closer to commentary than a restrained thesis register.")
        if not metaphor_or_storytelling_controlled:
            warnings.append(
                "Metaphoric or storytelling-style phrasing remains visible in rewritten prose: "
                f"{metaphor_paragraph_ids[:8]}"
            )
        if not authorial_claim_risk_controlled:
            warnings.append(
                "Unjustified first-person or overclaimed authorial phrasing remains visible: "
                f"{authorial_claim_paragraph_ids[:8]}"
            )
        if high_sensitivity_appendix_like_paragraph_ids:
            warnings.append(
                "High-sensitivity prose still contains appendix-like support phrasing: "
                f"{high_sensitivity_appendix_like_paragraph_ids[:8]}"
            )
        if high_sensitivity_unsupported_paragraph_ids:
            warnings.append(
                "High-sensitivity prose still contains unsupported expansion or outside-domain commentary: "
                f"{high_sensitivity_unsupported_paragraph_ids[:8]}"
            )
        if not bureaucratic_opening_controlled:
            warnings.append(
                "Project-style or bureaucratic openings remain visible in rewritten prose: "
                f"{bureaucratic_paragraph_ids[:8]}"
            )
        if not explicit_subject_chain_controlled:
            warnings.append(
                "Explicit academic subjects repeat mechanically across neighboring sentences: "
                f"{repeated_subject_paragraph_ids[:8]}"
            )
        if not overstructured_syntax_controlled:
            warnings.append(
                "Over-structured parallel or contrast syntax remains visible: "
                f"{overstructured_paragraph_ids[:8]}"
            )
        if not main_clause_position_reasonable:
            warnings.append(
                "Some rewritten sentences still delay the main clause behind heavy modifiers: "
                f"{delayed_main_clause_paragraph_ids[:8]}"
            )
        if not slogan_like_goal_phrase_controlled:
            warnings.append(
                "Slogan-like goal phrasing remains visible in rewritten prose: "
                f"{slogan_like_goal_paragraph_ids[:8]}"
            )
        if not author_style_alignment_controlled:
            warnings.append(
                "Author-style directness remains below target; inspect low-directness or over-wrapped paragraphs: "
                f"{low_directness_paragraph_ids[:8]}"
            )
        if connector_overuse_paragraph_ids:
            warnings.append(
                "Heavy connectors remain visible in some rewritten paragraphs: "
                f"{connector_overuse_paragraph_ids[:8]}"
            )
        if nominalization_paragraph_ids:
            warnings.append(
                "Nominalized academic actions remain dense in some rewritten paragraphs: "
                f"{nominalization_paragraph_ids[:8]}"
            )
        if passive_voice_paragraph_ids:
            warnings.append(
                "Passive or classificatory phrasing remains visible in some rewritten paragraphs: "
                f"{passive_voice_paragraph_ids[:8]}"
            )
        if overlong_sentence_paragraph_ids:
            warnings.append(
                "Some rewritten sentences still carry too many logic layers: "
                f"{overlong_sentence_paragraph_ids[:8]}"
            )
        if subject_monotony_paragraph_ids:
            warnings.append(
                "Explicit subjects remain monotonous in some rewritten paragraphs: "
                f"{subject_monotony_paragraph_ids[:8]}"
            )
        if l2_profile_enabled:
            if not l2_texture_present:
                warnings.append(
                    "zh_academic_l2_mild texture is too weak; rewritten prose still reads like default native academic polish: "
                    f"{list(l2_style_profile.get('low_l2_texture_paragraph_ids', []))[:8]}"
                )
            if not l2_not_too_native_like:
                warnings.append(
                    "zh_academic_l2_mild profile remains too condensed/native-like in some paragraphs: "
                    f"{list(l2_style_profile.get('native_like_paragraph_ids', []))[:8]}"
                )
            if not l2_not_colloquial:
                reject_reasons.append("zh_academic_l2_mild_profile_became_colloquial")
            if not l2_not_ungrammatical:
                reject_reasons.append("zh_academic_l2_mild_profile_became_ungrammatical")
        if semantic_role_drift_paragraph_ids:
            warnings.append(
                f"Some paragraphs still drift away from their original semantic role: {semantic_role_drift_paragraph_ids[:8]}"
            )
        if enumeration_drift_paragraph_ids:
            warnings.append(
                f"Some enumeration paragraphs still read like appendix-like support sentences: {enumeration_drift_paragraph_ids[:8]}"
            )
        if high_sensitivity_cliche_paragraph_ids:
            warnings.append(
                "High-sensitivity prose still contains templated academic cliché residue: "
                f"{high_sensitivity_cliche_paragraph_ids[:8]}"
            )
        if high_sensitivity_scaffolding_paragraph_ids:
            warnings.append(
                "High-sensitivity prose still contains generated scaffolding phrasing: "
                f"{high_sensitivity_scaffolding_paragraph_ids[:8]}"
            )
        if abstracted_subject_paragraph_ids:
            warnings.append(
                "Some rewritten paragraphs still begin with abstracted subjects instead of concrete referents: "
                f"{abstracted_subject_paragraph_ids[:8]}"
            )
        elif low_texture_variation_paragraph_ids:
            warnings.append(
                "Some rewritten paragraphs still read like a uniform polishing pass: "
                f"{low_texture_variation_paragraph_ids[:8]}"
            )
    if target_style_alignment.get("enabled", False):
        if float(target_style_alignment.get("grammar_error_rate", 0.0)) > 0.02:
            reject_reasons.append("target_style_alignment_grammar_error_rate_too_high")
        if int(target_style_alignment.get("terminology_drift", 0)) != 0:
            reject_reasons.append("target_style_alignment_terminology_drift_detected")
        if int(target_style_alignment.get("evidence_drift", 0)) != 0:
            reject_reasons.append("target_style_alignment_evidence_drift_detected")
        if float(target_style_alignment.get("style_distribution_match_ratio", 0.0)) < 0.70:
            warnings.append("Target-style distribution match remains below the desired 0.70 threshold.")
        if float(target_style_alignment.get("class_aware_style_match_ratio", 0.0)) < 0.70:
            warnings.append("Paragraph-class target-style match remains below the desired 0.70 threshold.")
        if float(target_style_alignment.get("native_fluency_gap", 0.0)) > 0.10:
            warnings.append("Output still reads more native-polished than the target reference style.")
        if float(target_style_alignment.get("explanatory_rewrite_gap", 0.0)) > 0.08:
            warnings.append("Output remains less explanatory than the target reference style.")
        if target_style_alignment.get("style_deviation_examples"):
            warnings.append(
                "Target-style deviation sentences: "
                f"{[item.get('sentence_id') for item in target_style_alignment.get('style_deviation_examples', [])[:8]]}"
            )
    if assessment.structural_action_count == 0:
        reject_reasons.append("No structural action was executed.")
    if assessment.prefix_only_rewrite:
        reject_reasons.append("Only prefix-level rewriting was detected.")
    if assessment.repeated_subject_risk:
        reject_reasons.append("Repeated subject chain was not repaired.")
    if assessment.template_risk:
        if assessment.template_issue == "templated_rule_like_rewrite" and substantive_body_rewrite:
            warnings.append("templated_rule_like_rewrite remains at a minor level after substantive rewrite.")
        else:
            reject_reasons.append(f"{assessment.template_issue or 'templated_rule_like_rewrite'} detected.")
    if not effective_change:
        reject_reasons.append("Only word-level or negligible change was produced.")
    if mode in {RewriteMode.BALANCED, RewriteMode.STRONG} and not assessment.depth_sufficient:
        if substantive_body_rewrite:
            warnings.extend(assessment.reasons)
        else:
            reject_reasons.extend(assessment.reasons)
    if _count_blacklisted_templates(analysis_revised) > DEFAULT_CONFIG.strict_blacklist_repeat_limit:
        if substantive_body_rewrite:
            warnings.append("Blacklisted template openers remain but the body rewrite is substantive.")
        else:
            reject_reasons.append("Blacklisted template openers repeated beyond the strict threshold.")
    if mode in {RewriteMode.BALANCED, RewriteMode.STRONG} and body_metrics.body_blocks_total:
        if body_metrics.body_rewrite_coverage < body_metrics.required_body_rewrite_coverage:
            reject_reasons.append("Body rewrite coverage stayed below the scale-aware rewrite quota.")
        if body_metrics.body_discourse_change_score < body_metrics.required_body_discourse_change_score:
            reject_reasons.append("Body discourse-change score stayed below the scale-aware rewrite quota.")
        if body_metrics.body_cluster_rewrite_score < body_metrics.required_body_cluster_rewrite_score:
            reject_reasons.append("Body cluster-rewrite score stayed below the scale-aware rewrite quota.")

    if guidance is not None:
        block_problems, block_ids = _validate_block_obligations(guidance, candidates)
        if substantive_body_rewrite and chapter_rewrite_quota_check and chapter_policy_consistency_check:
            pass
        elif substantive_body_rewrite:
            warnings.extend(block_problems)
        else:
            reject_reasons.extend(block_problems)
            failed_block_ids.extend(block_ids)

    core_content_integrity = not core_problems
    format_integrity = not format_problems
    if core_problems or format_problems:
        reject_reasons.extend(core_problems + format_problems)

    warnings.extend(assessment.warnings)
    natural_revision_checklist = _build_natural_revision_checklist(
        original=analysis_original,
        revised=analysis_revised,
        core_content_integrity=not core_problems,
        format_integrity=not format_problems,
        assessment=assessment,
        rewrite_stats=analysis_stats,
        rewrite_coverage=rewrite_coverage,
    )
    natural_revision_checklist.update(
        {
            "human_like_variation": bool(human_revision["human_like_variation"]),
            "non_uniform_rewrite_distribution": bool(human_revision["non_uniform_rewrite_distribution"]),
            "sentence_cluster_changes_present": bool(human_revision["sentence_cluster_changes_present"]),
            "narrative_flow_changed": bool(human_revision["narrative_flow_changed"]),
            "chapter_policy_consistency_check": chapter_policy_consistency_check,
            "chapter_rewrite_quota_check": chapter_rewrite_quota_check,
            "paragraph_topic_sentence_preserved": paragraph_topic_sentence_preserved,
            "paragraph_opening_style_valid": paragraph_opening_style_valid,
            "paragraph_skeleton_consistent": paragraph_skeleton_consistent,
            "no_dangling_opening_sentence": no_dangling_opening_sentence,
            "topic_sentence_not_demoted_to_mid_paragraph": topic_sentence_not_demoted_to_mid_paragraph,
            "local_transition_natural": local_transition_natural,
            "local_discourse_not_flat": local_discourse_not_flat,
            "sentence_uniformity_reduced": sentence_uniformity_reduced,
            "revision_realism_present": revision_realism_present,
            "stylistic_uniformity_controlled": stylistic_uniformity_controlled,
            "support_sentence_texture_varied": support_sentence_texture_varied,
            "paragraph_voice_variation_present": paragraph_voice_variation_present,
            "academic_cliche_density_controlled": academic_cliche_density_controlled,
            "sentence_completeness_preserved": sentence_completeness_preserved,
            "paragraph_readability_preserved": paragraph_readability_preserved,
            "no_dangling_support_sentences": no_dangling_support_sentences,
            "no_fragment_like_conclusion_sentences": no_fragment_like_conclusion_sentences,
            "semantic_role_integrity_preserved": semantic_role_integrity_preserved,
            "enumeration_integrity_preserved": enumeration_integrity_preserved,
            "scaffolding_phrase_density_controlled": scaffolding_phrase_density_controlled,
            "over_abstracted_subject_risk_controlled": over_abstracted_subject_risk_controlled,
            "assertion_strength_preserved": assertion_strength_preserved,
            "appendix_like_support_controlled": appendix_like_support_controlled,
            "authorial_stance_present": authorial_stance_present,
            "evidence_fidelity_preserved": evidence_fidelity_preserved,
            "unsupported_expansion_controlled": unsupported_expansion_controlled,
            "thesis_tone_restrained": thesis_tone_restrained,
            "metaphor_or_storytelling_controlled": metaphor_or_storytelling_controlled,
            "authorial_claim_risk_controlled": authorial_claim_risk_controlled,
            "bureaucratic_opening_controlled": bureaucratic_opening_controlled,
            "explicit_subject_chain_controlled": explicit_subject_chain_controlled,
            "overstructured_syntax_controlled": overstructured_syntax_controlled,
            "main_clause_position_reasonable": main_clause_position_reasonable,
            "slogan_like_goal_phrase_controlled": slogan_like_goal_phrase_controlled,
            "author_style_alignment_controlled": author_style_alignment_controlled,
            "target_style_alignment_enabled": bool(target_style_alignment.get("enabled", False)),
            "target_style_alignment_score": float(target_style_alignment.get("target_style_alignment_score", 1.0)) >= 0.70,
            "style_distribution_match_ratio": float(target_style_alignment.get("style_distribution_match_ratio", 1.0)) >= 0.70,
            "target_style_grammar_ok": float(target_style_alignment.get("grammar_error_rate", 0.0)) <= 0.02,
            "target_style_terminology_ok": int(target_style_alignment.get("terminology_drift", 0)) == 0,
            "target_style_evidence_ok": int(target_style_alignment.get("evidence_drift", 0)) == 0,
        }
    )
    if not natural_revision_checklist["academic_not_colloquial"]:
        reject_reasons.append("Rewritten text became too colloquial for academic style.")
    if not natural_revision_checklist["template_connectors_reduced_or_controlled"]:
        warnings.append("Mechanical connectors remain dense after revision.")
    if not natural_revision_checklist["parallelism_controlled"]:
        warnings.append("Template-like parallelism remains after revision.")
    if not natural_revision_checklist["function_word_overuse_reduced_or_controlled"]:
        warnings.append("Function-word or empty-verb overuse remains after revision.")

    if assessment.template_warning and assessment.template_issue:
        warnings.append(f"{assessment.template_issue} remains at a minor level.")
    if assessment.repeated_subject_warning:
        warnings.append("Minor repeated-subject residue remains after an otherwise substantive rewrite.")

    problems = _deduplicate_preserve_order(reject_reasons)
    warnings = _deduplicate_preserve_order(warnings)
    failed_block_ids = sorted(set(failed_block_ids))
    if problems:
        decision = "reject"
    elif warnings:
        decision = "pass_with_minor_risk"
    else:
        decision = "pass"

    naturalness_score = _compute_naturalness_score(
        template_risk=assessment.template_risk,
        template_warning=assessment.template_warning,
        repeated_subject_risk=assessment.repeated_subject_risk,
        depth_sufficient=assessment.depth_sufficient,
        effective_change=effective_change,
    )
    suggested_fallback = _suggested_fallback(
        decision=decision,
        template_risk=assessment.template_risk,
        format_integrity=format_integrity,
        core_content_integrity=core_content_integrity,
    )
    write_gate_ready = (
        decision in {"pass", "pass_with_minor_risk"}
        and effective_change
        and core_content_integrity
        and format_integrity
        and body_metrics.rewrite_quota_met
        and chapter_rewrite_quota_check
        and chapter_policy_consistency_check
        and paragraph_topic_sentence_preserved
        and paragraph_opening_style_valid
        and paragraph_skeleton_consistent
        and no_dangling_opening_sentence
        and topic_sentence_not_demoted_to_mid_paragraph
        and local_transition_natural
        and local_discourse_not_flat
        and sentence_uniformity_reduced
        and revision_realism_present
        and stylistic_uniformity_controlled
        and support_sentence_texture_varied
        and paragraph_voice_variation_present
        and academic_cliche_density_controlled
        and sentence_completeness_preserved
        and paragraph_readability_preserved
        and no_dangling_support_sentences
        and no_fragment_like_conclusion_sentences
        and semantic_role_integrity_preserved
        and enumeration_integrity_preserved
        and scaffolding_phrase_density_controlled
        and over_abstracted_subject_risk_controlled
        and assertion_strength_preserved
        and appendix_like_support_controlled
        and authorial_stance_present
        and evidence_fidelity_preserved
        and unsupported_expansion_controlled
        and thesis_tone_restrained
        and metaphor_or_storytelling_controlled
        and authorial_claim_risk_controlled
        and (body_metrics.document_scale not in {"long", "very_long"} or bureaucratic_opening_controlled)
        and (body_metrics.document_scale not in {"long", "very_long"} or explicit_subject_chain_controlled)
        and (body_metrics.document_scale not in {"long", "very_long"} or overstructured_syntax_controlled)
        and (body_metrics.document_scale not in {"long", "very_long"} or main_clause_position_reasonable)
        and (body_metrics.document_scale not in {"long", "very_long"} or slogan_like_goal_phrase_controlled)
        and (body_metrics.document_scale not in {"long", "very_long"} or author_style_alignment_controlled)
        and float(target_style_alignment.get("grammar_error_rate", 0.0)) <= 0.02
        and int(target_style_alignment.get("terminology_drift", 0)) == 0
        and int(target_style_alignment.get("evidence_drift", 0)) == 0
        and (body_metrics.document_scale not in {"long", "very_long"} or human_revision["sentence_cluster_changes_present"])
        and (body_metrics.document_scale not in {"long", "very_long"} or human_revision["human_like_variation"])
        and (body_metrics.document_scale not in {"long", "very_long"} or human_revision["non_uniform_rewrite_distribution"])
        and assessment.structural_action_count >= _required_structural_action_count(mode, len(_normalize_text(original)))
        and len(high_value_actions) >= DEFAULT_CONFIG.strict_min_high_value_actions
        and body_metrics.body_discourse_change_score >= body_metrics.required_body_discourse_change_score
        and body_metrics.body_cluster_rewrite_score >= body_metrics.required_body_cluster_rewrite_score
        and body_metrics.body_rewrite_coverage >= body_metrics.required_body_rewrite_coverage
    )

    return ReviewReport(
        ok=decision in {"pass", "pass_with_minor_risk"},
        decision=decision,
        problems=problems,
        warnings=warnings,
        meaningful_change=meaningful_change,
        effective_change=effective_change,
        changed_characters=changed_characters,
        diff_spans=diff_spans,
        sentence_level_change=sentence_level_change,
        depth_sufficient=assessment.depth_sufficient,
        template_risk=assessment.template_risk,
        naturalness_risk=assessment.naturalness_risk,
        structural_action_count=assessment.structural_action_count,
        high_value_action_count=assessment.high_value_action_count,
        high_value_actions=high_value_actions,
        discourse_change_score=body_metrics.body_discourse_change_score,
        cluster_rewrite_score=body_metrics.body_cluster_rewrite_score,
        style_variation_score=assessment.style_variation_score,
        rewrite_coverage=rewrite_coverage,
        body_rewrite_coverage=body_metrics.body_rewrite_coverage,
        body_changed_blocks=body_metrics.body_changed_blocks,
        body_blocks_total=body_metrics.body_blocks_total,
        body_changed_sentences=body_metrics.body_changed_sentences,
        body_sentences_total=body_metrics.body_sentences_total,
        body_changed_block_ratio=body_metrics.body_changed_block_ratio,
        body_discourse_change_score=body_metrics.body_discourse_change_score,
        body_cluster_rewrite_score=body_metrics.body_cluster_rewrite_score,
        body_developmental_changed_blocks=body_metrics.body_developmental_changed_blocks,
        body_developmental_blocks_total=body_metrics.body_developmental_blocks_total,
        document_scale=body_metrics.document_scale,
        rewrite_quota_met=body_metrics.rewrite_quota_met,
        rewrite_quota_reason_codes=body_metrics.rewrite_quota_reason_codes,
        required_body_rewrite_coverage=body_metrics.required_body_rewrite_coverage,
        required_body_changed_block_ratio=body_metrics.required_body_changed_block_ratio,
        required_body_discourse_change_score=body_metrics.required_body_discourse_change_score,
        required_body_cluster_rewrite_score=body_metrics.required_body_cluster_rewrite_score,
        chapter_rewrite_metrics=chapter_rewrite_metrics,
        chapter_policy_consistency_check=chapter_policy_consistency_check,
        chapter_rewrite_quota_check=chapter_rewrite_quota_check,
        chapter_rewrite_quota_reason_codes=chapter_rewrite_quota_reason_codes,
        human_like_variation=bool(human_revision["human_like_variation"]),
        non_uniform_rewrite_distribution=bool(human_revision["non_uniform_rewrite_distribution"]),
        sentence_cluster_changes_present=bool(human_revision["sentence_cluster_changes_present"]),
        narrative_flow_changed=bool(human_revision["narrative_flow_changed"]),
        paragraph_topic_sentence_preserved=paragraph_topic_sentence_preserved,
        paragraph_opening_style_valid=paragraph_opening_style_valid,
        paragraph_skeleton_consistent=paragraph_skeleton_consistent,
        no_dangling_opening_sentence=no_dangling_opening_sentence,
        topic_sentence_not_demoted_to_mid_paragraph=topic_sentence_not_demoted_to_mid_paragraph,
        paragraph_skeleton_review=dict(paragraph_skeleton_review),
        local_transition_natural=local_transition_natural,
        local_discourse_not_flat=local_discourse_not_flat,
        sentence_uniformity_reduced=sentence_uniformity_reduced,
        revision_realism_present=revision_realism_present,
        stylistic_uniformity_controlled=stylistic_uniformity_controlled,
        support_sentence_texture_varied=support_sentence_texture_varied,
        paragraph_voice_variation_present=paragraph_voice_variation_present,
        academic_cliche_density_controlled=academic_cliche_density_controlled,
        local_revision_realism=dict(local_realism),
        sentence_completeness_preserved=sentence_completeness_preserved,
        paragraph_readability_preserved=paragraph_readability_preserved,
        no_dangling_support_sentences=no_dangling_support_sentences,
        no_fragment_like_conclusion_sentences=no_fragment_like_conclusion_sentences,
        sentence_readability=dict(sentence_readability),
        semantic_role_integrity_preserved=semantic_role_integrity_preserved,
        enumeration_integrity_preserved=enumeration_integrity_preserved,
        scaffolding_phrase_density_controlled=scaffolding_phrase_density_controlled,
        over_abstracted_subject_risk_controlled=over_abstracted_subject_risk_controlled,
        semantic_role_integrity=dict(semantic_role),
        assertion_strength_preserved=assertion_strength_preserved,
        appendix_like_support_controlled=appendix_like_support_controlled,
        authorial_stance_present=authorial_stance_present,
        authorial_intent=dict(authorial_intent),
        evidence_fidelity_preserved=evidence_fidelity_preserved,
        unsupported_expansion_controlled=unsupported_expansion_controlled,
        thesis_tone_restrained=thesis_tone_restrained,
        metaphor_or_storytelling_controlled=metaphor_or_storytelling_controlled,
        authorial_claim_risk_controlled=authorial_claim_risk_controlled,
        evidence_fidelity=dict(evidence_fidelity),
        bureaucratic_opening_controlled=bureaucratic_opening_controlled,
        explicit_subject_chain_controlled=explicit_subject_chain_controlled,
        overstructured_syntax_controlled=overstructured_syntax_controlled,
        main_clause_position_reasonable=main_clause_position_reasonable,
        slogan_like_goal_phrase_controlled=slogan_like_goal_phrase_controlled,
        academic_sentence_naturalization=dict(sentence_naturalization),
        l2_style_profile=dict(l2_style_profile),
        target_style_alignment=dict(target_style_alignment),
        revision_pattern_distribution=dict(human_revision["revision_pattern_distribution"]),
        human_noise_marks=list(human_revision["human_noise_marks"]),
        prefix_only_rewrite=assessment.prefix_only_rewrite,
        repeated_subject_risk=assessment.repeated_subject_risk,
        template_warning=assessment.template_warning,
        template_issue=assessment.template_issue,
        core_content_integrity=core_content_integrity,
        format_integrity=format_integrity,
        title_integrity=integrity_checks.get("title_integrity_check", True),
        formula_integrity=integrity_checks.get("formula_integrity_check", True),
        terminology_integrity=integrity_checks.get("terminology_integrity_check", True),
        citation_integrity=integrity_checks.get("citation_integrity_check", True),
        numeric_integrity=integrity_checks.get("numeric_integrity_check", True),
        path_integrity=integrity_checks.get("path_integrity_check", True),
        heading_format_integrity=integrity_checks.get("heading_format_integrity_check", True),
        english_spacing_integrity=integrity_checks.get("english_spacing_integrity_check", True),
        placeholder_integrity=integrity_checks.get("placeholder_integrity_check", True),
        caption_punctuation_integrity=integrity_checks.get("caption_punctuation_integrity_check", True),
        markdown_symbol_integrity=integrity_checks.get("markdown_symbol_integrity_check", True),
        linebreak_whitespace_integrity=integrity_checks.get("linebreak_whitespace_integrity_check", True),
        naturalness_score=naturalness_score,
        suggested_fallback=suggested_fallback,
        write_gate_ready=write_gate_ready,
        failed_block_ids=failed_block_ids,
        integrity_checks=integrity_checks,
        natural_revision_checklist=natural_revision_checklist,
    )


def review_revision(
    original: str,
    revised: str,
    mode: RewriteMode,
    rewrite_stats: list[RewriteStats] | None = None,
    suffix: str = ".txt",
) -> ReviewReport:
    return review_rewrite(
        original=original,
        revised=revised,
        guidance=None,
        mode=mode,
        rewrite_stats=rewrite_stats,
        block_candidates=None,
        suffix=suffix,
    )


def _assess_local_revision_realism(rewrite_stats: list[RewriteStats]) -> dict[str, object]:
    """Convert aggregate local realism metrics into reviewer gate booleans."""

    realism = aggregate_local_revision_realism(rewrite_stats)
    return {
        **realism,
        "local_transition_natural": bool(realism["local_transition_natural"]),
        "local_discourse_not_flat": bool(realism["local_discourse_not_flat"]),
        "sentence_uniformity_reduced": bool(realism["sentence_uniformity_reduced"]),
        "revision_realism_present": bool(realism["revision_realism_present"]),
        "stylistic_uniformity_controlled": bool(realism["stylistic_uniformity_controlled"]),
        "support_sentence_texture_varied": bool(realism["support_sentence_texture_varied"]),
        "paragraph_voice_variation_present": bool(realism["paragraph_voice_variation_present"]),
        "academic_cliche_density_controlled": bool(realism["academic_cliche_density_controlled"]),
    }


def _assess_sentence_readability(rewrite_stats: list[RewriteStats]) -> dict[str, object]:
    """Convert aggregate readability metrics into reviewer gate booleans."""

    readability = aggregate_sentence_readability(rewrite_stats)
    return {
        **readability,
        "sentence_completeness_preserved": bool(readability["sentence_completeness_preserved"]),
        "paragraph_readability_preserved": bool(readability["paragraph_readability_preserved"]),
        "no_dangling_support_sentences": bool(readability["no_dangling_support_sentences"]),
        "no_fragment_like_conclusion_sentences": bool(readability["no_fragment_like_conclusion_sentences"]),
    }


def _assess_semantic_role_integrity(rewrite_stats: list[RewriteStats]) -> dict[str, object]:
    """Convert semantic-role preservation metrics into reviewer gate booleans."""

    semantic_role = aggregate_semantic_role_integrity(rewrite_stats)
    return {
        **semantic_role,
        "semantic_role_integrity_preserved": bool(semantic_role["semantic_role_integrity_preserved"]),
        "enumeration_integrity_preserved": bool(semantic_role["enumeration_integrity_preserved"]),
        "scaffolding_phrase_density_controlled": bool(semantic_role["scaffolding_phrase_density_controlled"]),
        "over_abstracted_subject_risk_controlled": bool(semantic_role["over_abstracted_subject_risk_controlled"]),
    }


def _assess_authorial_intent(rewrite_stats: list[RewriteStats]) -> dict[str, object]:
    """Convert authorial-intent metrics into reviewer gate booleans."""

    authorial = aggregate_authorial_intent(rewrite_stats)
    return {
        **authorial,
        "assertion_strength_preserved": bool(authorial["assertion_strength_preserved"]),
        "appendix_like_support_controlled": bool(authorial["appendix_like_support_controlled"]),
        "authorial_stance_present": bool(authorial["authorial_stance_present"]),
    }


def _assess_evidence_fidelity(rewrite_stats: list[RewriteStats]) -> dict[str, object]:
    """Convert evidence-fidelity metrics into reviewer gate booleans."""

    fidelity = aggregate_evidence_fidelity(rewrite_stats)
    return {
        **fidelity,
        "evidence_fidelity_preserved": bool(fidelity["evidence_fidelity_preserved"]),
        "unsupported_expansion_controlled": bool(fidelity["unsupported_expansion_controlled"]),
        "thesis_tone_restrained": bool(fidelity["thesis_tone_restrained"]),
        "metaphor_or_storytelling_controlled": bool(fidelity["metaphor_or_storytelling_controlled"]),
        "authorial_claim_risk_controlled": bool(fidelity["authorial_claim_risk_controlled"]),
    }


def _assess_academic_sentence_naturalization(rewrite_stats: list[RewriteStats]) -> dict[str, object]:
    """Convert academic sentence naturalization metrics into reviewer gate booleans."""

    naturalization = aggregate_academic_sentence_naturalization(rewrite_stats)
    return {
        **naturalization,
        "bureaucratic_opening_controlled": bool(naturalization["bureaucratic_opening_controlled"]),
        "explicit_subject_chain_controlled": bool(naturalization["explicit_subject_chain_controlled"]),
        "overstructured_syntax_controlled": bool(naturalization["overstructured_syntax_controlled"]),
        "main_clause_position_reasonable": bool(naturalization["main_clause_position_reasonable"]),
        "slogan_like_goal_phrase_controlled": bool(naturalization["slogan_like_goal_phrase_controlled"]),
        "author_style_alignment_controlled": bool(naturalization["author_style_alignment_controlled"]),
    }


def _measure_diff(original: str, revised: str) -> tuple[int, int]:
    matcher = SequenceMatcher(a=original, b=revised)
    changed_characters = 0
    diff_spans = 0

    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        diff_spans += 1
        changed_characters += max(left_end - left_start, right_end - right_start)

    return changed_characters, diff_spans


def _compute_rewrite_coverage_metric(
    original: str,
    revised: str,
    rewrite_stats: list[RewriteStats],
) -> float:
    if rewrite_stats:
        denominator = sum(max(1, stats.sentence_count_before) for stats in rewrite_stats if stats.changed)
        if denominator == 0:
            return 0.0
        changed_units = sum(
            min(
                max(1, stats.sentence_count_before),
                max(stats.sentence_level_changes, stats.cluster_changes),
            )
            for stats in rewrite_stats
            if stats.changed
        )
        return min(1.0, changed_units / denominator)

    original_sentences = split_sentences(original)
    revised_sentences = split_sentences(revised)
    if not original_sentences:
        return 0.0
    if len(original_sentences) != len(revised_sentences):
        return min(1.0, max(len(original_sentences), len(revised_sentences)) / len(original_sentences))
    changed = sum(
        1
        for left, right in zip(original_sentences, revised_sentences)
        if _normalize_sentence(left) != _normalize_sentence(right)
    )
    return min(1.0, changed / len(original_sentences))


def _review_naturalness(
    original: str,
    revised: str,
    mode: RewriteMode,
    rewrite_stats: list[RewriteStats],
) -> NaturalnessAssessment:
    if not rewrite_stats:
        original_sentences = split_sentences(original)
        revised_sentences = split_sentences(revised)
        repeated_subject_streak = _max_meta_subject_streak(_subject_heads_from_text(revised))
        repeated_subject_risk = repeated_subject_streak >= 3
        repeated_subject_warning = repeated_subject_streak == 2
        structural_action_count = int(
            len(original_sentences) != len(revised_sentences) or _has_sentence_level_change(original, revised)
        )
        high_value_actions = _infer_high_value_actions_from_text(original, revised)
        discourse_change_score = _infer_discourse_change_score(original, revised)
        cluster_rewrite_score = _infer_cluster_rewrite_score(original, revised)
        style_variation_score = _infer_style_variation_score(revised)
        prefix_only_rewrite = _looks_like_prefix_only_change(original, revised)
        severe_template_issue: str | None = None
        warning_template_issue: str | None = None
        if repeated_subject_streak >= 3 or _count_repeated_openers(revised) >= 3:
            severe_template_issue = "templated_family_repetition"
        elif _has_stiff_opener_sequence(revised) or style_variation_score <= DEFAULT_CONFIG.style_variation_reject_score:
            severe_template_issue = "templated_rule_like_rewrite"
        elif style_variation_score <= DEFAULT_CONFIG.style_variation_warning_score:
            warning_template_issue = "templated_rule_like_rewrite"
        template_risk = severe_template_issue is not None
        template_warning = severe_template_issue is None and warning_template_issue is not None
        reasons: list[str] = []
        warnings: list[str] = []
        if (
            mode in {RewriteMode.BALANCED, RewriteMode.STRONG}
            and len(_normalize_text(original)) >= DEFAULT_CONFIG.structural_gate_length
        ):
            if discourse_change_score < _required_discourse_change_score(mode, len(_normalize_text(original))):
                reasons.append("External candidate shows no clear discourse-level restructuring.")
            if cluster_rewrite_score < _required_cluster_rewrite_score(mode, len(_normalize_text(original))):
                reasons.append("External candidate did not reorganize any sentence cluster.")
        if repeated_subject_warning:
            warnings.append("One repeated meta-subject pair remains in the revised text.")
        return NaturalnessAssessment(
            depth_sufficient=not reasons and not template_risk and not prefix_only_rewrite,
            template_risk=template_risk,
            template_warning=template_warning,
            template_issue=severe_template_issue or warning_template_issue,
            naturalness_risk=template_risk or template_warning or repeated_subject_risk or repeated_subject_warning or prefix_only_rewrite,
            structural_action_count=structural_action_count,
            high_value_action_count=len(high_value_actions),
            high_value_actions=high_value_actions,
            discourse_change_score=discourse_change_score,
            cluster_rewrite_score=cluster_rewrite_score,
            style_variation_score=style_variation_score,
            prefix_only_rewrite=prefix_only_rewrite,
            repeated_subject_risk=repeated_subject_risk,
            repeated_subject_warning=repeated_subject_warning,
            reasons=reasons,
            warnings=warnings,
        )

    selected_variants = [variant for stats in rewrite_stats for variant in stats.selected_variants if variant != "keep-original"]
    variant_counts = Counter(selected_variants)
    repeated_rule_families = _count_repeated_rule_families(rewrite_stats)
    template_family_counts = _count_template_families(rewrite_stats)
    paragraph_template_family_counts = _count_paragraph_template_families(rewrite_stats)
    max_variant_repeat = max(variant_counts.values(), default=0)
    structural_action_count = sum(stats.structural_action_count for stats in rewrite_stats)
    high_value_action_count = sum(len(stats.high_value_structural_actions) for stats in rewrite_stats)
    discourse_change_score = sum(stats.discourse_change_score for stats in rewrite_stats)
    cluster_rewrite_score = sum(stats.cluster_changes for stats in rewrite_stats)
    style_variation_score = _style_variation_score_from_stats(
        selected_variants=selected_variants,
        repeated_rule_families=repeated_rule_families,
        template_family_counts=template_family_counts,
        max_variant_repeat=max_variant_repeat,
        revised=revised,
    )
    changed_stats = [stats for stats in rewrite_stats if stats.changed]
    prefix_only_rewrite = bool(changed_stats) and all(
        stats.prefix_only_rewrite or stats.structural_action_count == 0 for stats in changed_stats
    )
    repeated_subject_streak = max((_max_meta_subject_streak(stats.subject_heads) for stats in rewrite_stats), default=0)
    repeated_subject_risk = repeated_subject_streak >= 3
    repeated_subject_warning = repeated_subject_streak == 2
    reasons: list[str] = []
    warnings: list[str] = []

    if mode in {RewriteMode.BALANCED, RewriteMode.STRONG} and len(_normalize_text(original)) >= DEFAULT_CONFIG.structural_gate_length:
        if not rewrite_stats:
            reasons.append("No rewrite statistics were produced for a long paragraph.")

        for stats in rewrite_stats:
            rewrite_depth = stats.rewrite_depth or ("developmental_rewrite" if mode in {RewriteMode.BALANCED, RewriteMode.STRONG} else "light_edit")
            required_actions, requires_high_value = _required_structural_actions(mode, stats.paragraph_char_count)
            if rewrite_depth == "light_edit":
                required_actions = 0
                requires_high_value = False
            if stats.prefix_only_rewrite:
                reasons.append(
                    f"Paragraph {stats.paragraph_index} only changed prefixes without any structural rewrite."
                )
                continue
            if rewrite_depth == "light_edit":
                if stats.sentence_level_changes < DEFAULT_CONFIG.light_edit_min_sentence_level_changes:
                    reasons.append(
                        f"Paragraph {stats.paragraph_index} did not reach the minimum sentence-level change for light_edit."
                    )
                if stats.discourse_change_score < DEFAULT_CONFIG.light_edit_min_discourse_score:
                    reasons.append(
                        f"Paragraph {stats.paragraph_index} did not reach the discourse-change threshold for light_edit."
                    )
                continue
            if stats.sentence_level_changes == 0 and stats.cluster_changes == 0:
                reasons.append(
                    f"Paragraph {stats.paragraph_index} did not produce any sentence-level or cluster-level change."
                )
                continue
            if stats.structural_action_count < required_actions:
                warnings.append(
                    f"Paragraph {stats.paragraph_index} has fewer structural actions than recommended for developmental rewrite."
                )
            if requires_high_value and len(stats.high_value_structural_actions) < 1:
                warnings.append(
                    f"Paragraph {stats.paragraph_index} lacks a high-value structural action for strong mode."
                )
            if stats.discourse_change_score < _required_discourse_change_score(mode, stats.paragraph_char_count):
                warnings.append(
                    f"Paragraph {stats.paragraph_index} did not reach the discourse-change threshold for body rewriting."
                )
            if stats.cluster_changes < _required_cluster_rewrite_score(mode, stats.paragraph_char_count):
                warnings.append(
                    f"Paragraph {stats.paragraph_index} did not show a sentence-cluster rewrite."
                )

    for stats in rewrite_stats:
        subject_streak = _max_meta_subject_streak(stats.subject_heads)
        subject_fix_count = sum(1 for action in stats.structural_actions if action in _SUBJECT_FIX_ACTIONS)
        if mode is RewriteMode.CONSERVATIVE and subject_streak >= 3:
            reasons.append(
                f"Paragraph {stats.paragraph_index} still has a repeated meta-subject chain of length {subject_streak}."
            )
        if mode is RewriteMode.BALANCED and subject_streak >= 3:
            reasons.append(
                f"Paragraph {stats.paragraph_index} still has consecutive repeated meta subjects."
            )
        if mode is RewriteMode.BALANCED and subject_streak == 2:
            warnings.append(
                f"Paragraph {stats.paragraph_index} still leaves one repeated meta-subject pair."
            )
        if mode is RewriteMode.STRONG and subject_streak >= 3:
            if subject_fix_count == 0:
                reasons.append(
                    f"Paragraph {stats.paragraph_index} repeats the same meta subject without merge, drop, or variation."
                )
            else:
                reasons.append(
                    f"Paragraph {stats.paragraph_index} still leaves a repeated meta-subject chain after restructuring."
                )
        if mode is RewriteMode.STRONG and subject_streak == 2:
            warnings.append(
                f"Paragraph {stats.paragraph_index} still leaves one repeated meta-subject pair after restructuring."
            )

    document_template_limit = (
        DEFAULT_CONFIG.document_template_family_block_limit
        if len(rewrite_stats) < 10
        else max(DEFAULT_CONFIG.document_template_family_block_limit + 3, len(rewrite_stats) // 4)
    )
    severe_family_repetition = (
        max(template_family_counts.values(), default=0) >= document_template_limit
        or any(
            count > DEFAULT_CONFIG.paragraph_template_family_limit
            for count in paragraph_template_family_counts.values()
        )
    )
    warning_family_repetition = (
        not severe_family_repetition
        and max(template_family_counts.values(), default=0) >= DEFAULT_CONFIG.document_template_family_warning_limit
    )
    severe_rule_like_rewrite = (
        repeated_rule_families >= DEFAULT_CONFIG.rule_like_family_block_limit
        or max_variant_repeat > DEFAULT_CONFIG.repeated_pattern_limit + 1
    )
    warning_rule_like_rewrite = (
        not severe_rule_like_rewrite
        and (
            repeated_rule_families >= DEFAULT_CONFIG.rule_like_family_warning_limit
            or _has_stiff_opener_sequence(revised)
        )
    )

    severe_template_issue: str | None = None
    if severe_family_repetition or repeated_subject_streak >= 3:
        severe_template_issue = "templated_family_repetition"
    elif severe_rule_like_rewrite:
        severe_template_issue = "templated_rule_like_rewrite"

    warning_template_issue: str | None = None
    if not severe_template_issue:
        if warning_family_repetition:
            warning_template_issue = "templated_family_repetition"
        elif warning_rule_like_rewrite:
            warning_template_issue = "templated_rule_like_rewrite"

    template_risk = bool(severe_template_issue)
    template_warning = False
    template_issue = severe_template_issue or warning_template_issue
    if not severe_template_issue and (
        warning_family_repetition
        or warning_rule_like_rewrite
        or style_variation_score <= DEFAULT_CONFIG.style_variation_warning_score
    ):
        template_warning = True
    naturalness_risk = (
        template_risk
        or template_warning
        or prefix_only_rewrite
        or repeated_subject_risk
        or repeated_subject_warning
    )
    depth_sufficient = not reasons and not (mode in {RewriteMode.BALANCED, RewriteMode.STRONG} and prefix_only_rewrite)

    if mode is RewriteMode.CONSERVATIVE:
        depth_sufficient = not template_risk and repeated_subject_streak < 3
        reasons = []

    return NaturalnessAssessment(
        depth_sufficient=depth_sufficient,
        template_risk=template_risk,
        template_warning=template_warning,
        template_issue=template_issue,
        naturalness_risk=naturalness_risk,
        structural_action_count=structural_action_count,
        high_value_action_count=high_value_action_count,
        high_value_actions=_collect_high_value_actions(rewrite_stats),
        discourse_change_score=discourse_change_score,
        cluster_rewrite_score=cluster_rewrite_score,
        style_variation_score=style_variation_score,
        prefix_only_rewrite=prefix_only_rewrite,
        repeated_subject_risk=repeated_subject_risk,
        repeated_subject_warning=repeated_subject_warning,
        reasons=_deduplicate_preserve_order(reasons),
        warnings=_deduplicate_preserve_order(warnings),
    )


def _has_meaningful_change(
    original: str,
    revised: str,
    mode: RewriteMode,
    changed_characters: int,
    diff_spans: int,
    sentence_level_change: bool,
    structural_action_count: int,
    discourse_change_score: int,
    cluster_rewrite_score: int,
    rewrite_coverage: float,
    prefix_only_rewrite: bool,
    depth_sufficient: bool,
) -> bool:
    if original == revised:
        return False

    if mode is RewriteMode.CONSERVATIVE:
        return changed_characters >= 2 or diff_spans >= 1

    if prefix_only_rewrite:
        return False

    threshold = (
        DEFAULT_CONFIG.strong_min_changed_chars
        if mode is RewriteMode.STRONG
        else DEFAULT_CONFIG.balanced_min_changed_chars
    )
    normalized_length = len(_normalize_text(original))

    if normalized_length >= DEFAULT_CONFIG.structural_gate_length and structural_action_count < 1:
        return False
    if (
        not depth_sufficient
        and discourse_change_score < _required_discourse_change_score(mode, normalized_length)
    ):
        return False
    if (
        cluster_rewrite_score < _required_cluster_rewrite_score(mode, normalized_length)
        and rewrite_coverage < DEFAULT_CONFIG.rewrite_coverage_minor_threshold
    ):
        return False

    return (
        sentence_level_change
        or structural_action_count >= 1
        or discourse_change_score >= DEFAULT_CONFIG.developmental_min_discourse_score
        or rewrite_coverage >= DEFAULT_CONFIG.rewrite_coverage_minor_threshold
        or (changed_characters >= threshold and diff_spans >= 2)
    )


def _assess_human_revision_behavior(
    *,
    original: str,
    revised: str,
    rewrite_stats: list[RewriteStats],
    block_candidates: list[BlockRewriteCandidate],
) -> dict[str, object]:
    pattern_counts: Counter[str] = Counter()
    human_noise_marks: list[str] = []
    changed_units_by_block: list[int] = []
    rewrite_coverage_by_block: list[float] = []
    cluster_actions = {
        "sentence_cluster_rewrite",
        "sentence_cluster_merge",
        "sentence_cluster_split",
        "conclusion_absorb",
        "conclusion_absorption",
        "transition_absorption",
        "proposition_reorder",
        "discourse_reordering",
        "narrative_path_rewrite",
    }
    narrative_actions = {
        "proposition_reorder",
        "discourse_reordering",
        "narrative_path_rewrite",
        "paragraph_reorder",
    }
    sentence_cluster_changes_present = False
    narrative_flow_changed = False

    for stats in rewrite_stats:
        pattern_counts.update(getattr(stats, "revision_patterns", []) or [])
        human_noise_marks.extend(getattr(stats, "human_noise_marks", []) or [])
        changed_units_by_block.append(max(stats.sentence_level_changes, stats.cluster_changes))
        rewrite_coverage_by_block.append(round(stats.rewrite_coverage, 2))
        actions = set(stats.discourse_actions_used) | set(stats.structural_actions)
        if stats.cluster_changes > 0 or actions & cluster_actions:
            sentence_cluster_changes_present = True
        if actions & narrative_actions or any(rule.startswith(("paragraph:", "cluster:")) for rule in stats.applied_rules):
            narrative_flow_changed = True

    for candidate in block_candidates:
        pattern_counts.update(getattr(candidate, "revision_patterns", []) or [])
        actions = set(candidate.discourse_actions_used) | set(candidate.structural_actions)
        if candidate.cluster_changes > 0 or actions & cluster_actions:
            sentence_cluster_changes_present = True
        if actions & narrative_actions:
            narrative_flow_changed = True

    if not pattern_counts:
        inferred = _infer_revision_patterns_from_text(original, revised)
        pattern_counts.update(inferred)
        changed_count = _external_changed_sentence_count(original, revised)
        total = max(1, len(split_sentences(original)))
        if 0 < changed_count < total:
            human_noise_marks.append("partial_keep")
        if len(split_sentences(original)) != len(split_sentences(revised)):
            sentence_cluster_changes_present = True
            human_noise_marks.append("cluster_length_shift")
        if _infer_cluster_rewrite_score(original, revised) > 0:
            sentence_cluster_changes_present = True
        if _infer_discourse_change_score(original, revised) >= DEFAULT_CONFIG.developmental_min_discourse_score:
            narrative_flow_changed = True
        changed_units_by_block.append(changed_count)
        rewrite_coverage_by_block.append(round(changed_count / total, 2))

    unique_patterns = len(pattern_counts)
    non_uniform_rewrite_distribution = (
        len(set(changed_units_by_block)) >= DEFAULT_CONFIG.human_variation_min_changed_block_spread
        or len(set(rewrite_coverage_by_block)) >= DEFAULT_CONFIG.human_variation_min_changed_block_spread
        or {"partial_keep", "rewrite_all"}.issubset(set(pattern_counts))
        or bool(set(human_noise_marks) & {"partial_keep", "uneven_sentence_change", "heavy_light_block_contrast", "cluster_length_shift"})
    )
    human_like_variation = (
        unique_patterns >= DEFAULT_CONFIG.human_variation_min_patterns
        and non_uniform_rewrite_distribution
        and not _looks_uniformly_rewritten(rewrite_stats)
    )

    return {
        "human_like_variation": human_like_variation,
        "non_uniform_rewrite_distribution": non_uniform_rewrite_distribution,
        "sentence_cluster_changes_present": sentence_cluster_changes_present,
        "narrative_flow_changed": narrative_flow_changed,
        "revision_pattern_distribution": dict(pattern_counts),
        "human_noise_marks": _deduplicate_preserve_order(human_noise_marks),
    }


def _infer_revision_patterns_from_text(original: str, revised: str) -> list[str]:
    patterns: list[str] = []
    original_sentences = split_sentences(original)
    revised_sentences = split_sentences(revised)
    if len(revised_sentences) < len(original_sentences):
        patterns.append("merge")
    elif len(revised_sentences) > len(original_sentences):
        patterns.append("split")
    if _infer_cluster_rewrite_score(original, revised) > 0:
        patterns.append("reorder")
    if _count_implication_openers(original) > _count_implication_openers(revised):
        patterns.append("merge")
    changed_count = _external_changed_sentence_count(original, revised)
    if 0 < changed_count < max(1, len(original_sentences)):
        patterns.append("partial_keep")
    elif changed_count >= max(1, len(original_sentences)):
        patterns.append("rewrite_all")
    if not patterns and original != revised:
        patterns.append("reframe")
    return _deduplicate_preserve_order(patterns)


def _external_changed_sentence_count(original: str, revised: str) -> int:
    original_sentences = [_normalize_sentence(sentence) for sentence in split_sentences(original)]
    revised_sentences = [_normalize_sentence(sentence) for sentence in split_sentences(revised)]
    if not original_sentences:
        return 0
    limit = min(len(original_sentences), len(revised_sentences))
    changed = sum(1 for index in range(limit) if original_sentences[index] != revised_sentences[index])
    return changed + abs(len(original_sentences) - len(revised_sentences))


def _looks_uniformly_rewritten(rewrite_stats: list[RewriteStats]) -> bool:
    changed_stats = [stats for stats in rewrite_stats if stats.changed]
    if len(changed_stats) < 3:
        return False
    coverages = [round(stats.rewrite_coverage, 2) for stats in changed_stats]
    patterns = [tuple(getattr(stats, "revision_patterns", []) or []) for stats in changed_stats]
    return len(set(coverages)) == 1 and len(set(patterns)) <= 1


def _required_structural_actions(mode: RewriteMode, paragraph_char_count: int) -> tuple[int, bool]:
    if mode is RewriteMode.CONSERVATIVE:
        return 0, False
    if paragraph_char_count >= DEFAULT_CONFIG.structural_gate_length_long:
        if mode is RewriteMode.BALANCED:
            return DEFAULT_CONFIG.balanced_min_structural_actions_long, False
        return DEFAULT_CONFIG.strong_min_structural_actions_long, True
    if paragraph_char_count >= DEFAULT_CONFIG.structural_gate_length:
        if mode is RewriteMode.BALANCED:
            return DEFAULT_CONFIG.balanced_min_structural_actions, False
        return DEFAULT_CONFIG.strong_min_structural_actions, True
    return 0, False


def _required_discourse_change_score(mode: RewriteMode, normalized_length: int) -> int:
    if mode is RewriteMode.CONSERVATIVE:
        return DEFAULT_CONFIG.light_edit_min_discourse_score
    if normalized_length >= DEFAULT_CONFIG.structural_gate_length:
        return DEFAULT_CONFIG.developmental_min_discourse_score
    return DEFAULT_CONFIG.light_edit_min_discourse_score


def _required_cluster_rewrite_score(mode: RewriteMode, normalized_length: int) -> int:
    if mode is RewriteMode.CONSERVATIVE:
        return 0
    if normalized_length >= DEFAULT_CONFIG.structural_gate_length:
        return DEFAULT_CONFIG.developmental_min_cluster_score
    return 0


def _count_repeated_rule_families(rewrite_stats: list[RewriteStats]) -> int:
    families: list[str] = []
    for stats in rewrite_stats:
        for rule in stats.applied_rules:
            if rule.startswith(("subject:", "structure:", "fusion:", "paragraph:")):
                continue
            if ":" in rule:
                families.append(rule)
    counts = Counter(families)
    return max(counts.values(), default=0)


def _count_template_families(rewrite_stats: list[RewriteStats]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for stats in rewrite_stats:
        counts.update(_template_families_from_stats(stats))
    return counts


def _count_paragraph_template_families(rewrite_stats: list[RewriteStats]) -> Counter[str]:
    paragraph_counts: Counter[str] = Counter()
    for stats in rewrite_stats:
        family_counts = Counter(_template_families_from_stats(stats))
        for family, count in family_counts.items():
            paragraph_counts[family] = max(paragraph_counts.get(family, 0), count)
    return paragraph_counts


def _template_families_from_stats(stats: RewriteStats) -> list[str]:
    families: list[str] = []
    for variant in stats.selected_variants:
        family = _variant_to_template_family(variant)
        if family:
            families.append(family)
    return families


def _variant_to_template_family(variant: str) -> str | None:
    if variant in {"本研究", "本文", "该研究", "研究"}:
        return "study_subject_family"
    if any(marker in variant for marker in ("本研究", "本文", "研究")) and "围绕" not in variant:
        return "study_subject_family"
    if any(marker in variant for marker in ("因此", "由此", "基于此", "这也意味着", "在这种情况下", "正因为如此", "在此基础上")):
        return "implication_family"
    if any(marker in variant for marker in ("同时", "与此同时", "在这一过程中", "此外", "与之相伴的是")):
        return "transition_family"
    if any(marker in variant for marker in ("围绕", "聚焦于", "尝试回应", "核心问题", "重点在于")):
        return "framing_family"
    return None




def _has_sentence_level_change(original: str, revised: str) -> bool:
    original_sentences = split_sentences(original)
    revised_sentences = split_sentences(revised)

    if len(original_sentences) != len(revised_sentences):
        return True

    differences = 0
    for left, right in zip(original_sentences, revised_sentences):
        if _normalize_sentence(left) != _normalize_sentence(right):
            differences += 1

    return differences >= 2


def _count_repeated_openers(text: str) -> int:
    openers = [_extract_opener(sentence) for sentence in split_sentences(text)]
    counts = Counter(opener for opener in openers if opener)
    return max(counts.values(), default=0)


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


def _subject_heads_from_text(text: str) -> list[str]:
    heads: list[str] = []
    for sentence in split_sentences(text):
        stripped = _LEADING_IMPLICATION_RE.sub("", sentence.strip())
        subject = next((item for item in _META_SUBJECTS if stripped.startswith(item)), "")
        heads.append(subject)
    return heads


def _has_stiff_opener_sequence(text: str) -> bool:
    openers = [_extract_opener(sentence) for sentence in split_sentences(text)]
    current = ""
    streak = 0

    for opener in openers:
        if opener and opener == current:
            streak += 1
        else:
            current = opener
            streak = 1 if opener else 0

        if opener in _STIFF_OPENERS and streak >= 2:
            return True
    return False


def _extract_opener(sentence: str) -> str:
    stripped = _normalize_sentence(sentence)
    for opener in _STIFF_OPENERS:
        if stripped.startswith(opener):
            return opener
    return stripped[:4]


def _normalize_text(text: str) -> str:
    return _NORMALIZE_RE.sub("", text)


def _normalize_sentence(text: str) -> str:
    stripped = re.sub(r"[。！？?!]+$", "", text)
    return _NORMALIZE_RE.sub("", stripped)


def _deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _count_headings(text: str) -> int:
    return len(_HEADING_RE.findall(text))


def _count_code_blocks(text: str) -> int:
    return len(_FENCED_CODE_RE.findall(text))


def _count_links(text: str) -> int:
    return len(_INLINE_LINK_RE.findall(text)) + len(_AUTOLINK_RE.findall(text)) + len(
        _BARE_URL_RE.findall(text)
    )


def _compute_naturalness_score(
    template_risk: bool,
    template_warning: bool,
    repeated_subject_risk: bool,
    depth_sufficient: bool,
    effective_change: bool,
) -> int:
    score = 100
    if template_risk:
        score -= 40
    elif template_warning:
        score -= 15
    if repeated_subject_risk:
        score -= 20
    if not depth_sufficient:
        score -= 20
    if not effective_change:
        score -= 15
    return max(0, min(100, score))


def _build_natural_revision_checklist(
    *,
    original: str,
    revised: str,
    core_content_integrity: bool,
    format_integrity: bool,
    assessment: NaturalnessAssessment,
    rewrite_stats: list[RewriteStats],
    rewrite_coverage: float,
) -> dict[str, bool]:
    discourse_actions = [action for stats in rewrite_stats for action in stats.discourse_actions_used]
    return {
        "meaning_and_facts_preserved": core_content_integrity,
        "format_and_markdown_preserved": format_integrity,
        "academic_not_colloquial": not any(
            marker in revised and marker not in original for marker in COLLOQUIAL_FORBIDDEN_MARKERS
        ),
        "template_connectors_reduced_or_controlled": _mechanical_connector_count(revised)
        <= max(_mechanical_connector_count(original), 2),
        "repeated_subjects_reduced_or_controlled": not (
            assessment.repeated_subject_risk or _max_meta_subject_streak(_subject_heads_from_text(revised)) >= 3
        ),
        "function_word_overuse_reduced_or_controlled": _function_word_overuse_score(revised)
        <= max(_function_word_overuse_score(original), 3),
        "parallelism_controlled": _parallelism_score(revised) <= max(_parallelism_score(original), 1),
        "sentence_rhythm_has_variation": _sentence_rhythm_score(revised) >= 1 or len(split_sentences(revised)) < 3,
        "meta_discourse_compressed": _meta_discourse_score(revised) <= max(_meta_discourse_score(original), 1)
        or "meta_compression" in discourse_actions,
        "narrative_flow_rebuilt": assessment.cluster_rewrite_score >= DEFAULT_CONFIG.developmental_min_cluster_score
        or any(action in discourse_actions for action in {"sentence_cluster_rewrite", "proposition_reorder", "transition_absorption"}),
        "developmental_rewrite_applied_to_body_blocks": rewrite_coverage >= DEFAULT_CONFIG.rewrite_coverage_minor_threshold
        and assessment.discourse_change_score >= DEFAULT_CONFIG.light_edit_min_discourse_score,
    }


def _mechanical_connector_count(text: str) -> int:
    markers = ("首先", "其次", "再次", "此外", "最后", "综上所述", "因此", "然而", "并且", "同时")
    return sum(text.count(marker) for marker in markers)


def _function_word_overuse_score(text: str) -> int:
    return len(_FUNCTION_WORD_RE.findall(text)) + sum(text.count(verb) for verb in _EMPTY_ACADEMIC_VERBS)


def _parallelism_score(text: str) -> int:
    return sum(text.count(marker) for marker in _PARALLELISM_MARKERS)


def _meta_discourse_score(text: str) -> int:
    markers = ("本研究的主题为", "本文讨论的核心问题是", "本研究尝试回应", "本研究聚焦于", "本研究围绕")
    return sum(text.count(marker) for marker in markers)


def _sentence_rhythm_score(text: str) -> int:
    sentences = split_sentences(text)
    if len(sentences) < 3:
        return 1
    lengths = [len(_normalize_sentence(sentence)) for sentence in sentences]
    return max(lengths) - min(lengths)


def _suggested_fallback(
    decision: str,
    template_risk: bool,
    format_integrity: bool,
    core_content_integrity: bool,
) -> str:
    if not core_content_integrity:
        return "revert_to_original_and_restore_protected_content"
    if not format_integrity:
        return "revert_to_original_or_retry_with_do_not_touch_blocks_frozen"
    if decision == "reject" and template_risk:
        return "retry_with_lighter_edits_and_keep_original_when_rewrite_sounds_stiff"
    if decision == "reject":
        return "retry_only_on_rewritable_blocks_or_downgrade_to_conservative"
    return "ready_for_write_gate"


def _do_not_touch_blocks_preserved(revised: str, guidance: GuidanceReport) -> bool:
    tracked_types = {"heading", "reference", "english_block", "caption", "placeholder", "formula"}
    revised_segments_by_type = {
        block_type: protected_segments_for_document(block_type, revised) for block_type in tracked_types
    }
    for block in guidance.do_not_touch_blocks:
        if block.block_type not in tracked_types:
            continue
        revised_segments = revised_segments_by_type.get(block.block_type, [])
        for segment in narrow_do_not_touch_scope(block.block_type, block.original_text):
            if segment and segment not in revised_segments:
                return False
    return True


def _required_structural_action_count(mode: RewriteMode, normalized_length: int) -> int:
    if mode is RewriteMode.STRONG:
        return 2 if normalized_length >= DEFAULT_CONFIG.structural_gate_length_long else 1
    if mode is RewriteMode.BALANCED:
        return 1
    return 1


def _collect_high_value_actions(rewrite_stats: list[RewriteStats]) -> list[str]:
    seen: list[str] = []
    for stats in rewrite_stats:
        for action in stats.high_value_structural_actions:
            if action not in seen:
                seen.append(action)
    return seen


def _infer_high_value_actions_from_text(original: str, revised: str) -> list[str]:
    inferred: list[str] = []
    original_subject_streak = _max_meta_subject_streak(_subject_heads_from_text(original))
    revised_subject_streak = _max_meta_subject_streak(_subject_heads_from_text(revised))
    if original_subject_streak >= 2 and revised_subject_streak < original_subject_streak:
        inferred.append("subject_chain_compression")
    if len(split_sentences(revised)) < len(split_sentences(original)):
        inferred.append("pair_fusion")
    if _count_implication_openers(original) > _count_implication_openers(revised):
        inferred.append("conclusion_absorb")
    return inferred


def _infer_discourse_change_score(original: str, revised: str) -> int:
    score = 0
    original_sentences = split_sentences(original)
    revised_sentences = split_sentences(revised)
    if len(original_sentences) != len(revised_sentences):
        score += 3
    if _has_sentence_level_change(original, revised):
        score += 2
    if _count_implication_openers(original) > _count_implication_openers(revised):
        score += 2
    if _max_meta_subject_streak(_subject_heads_from_text(original)) > _max_meta_subject_streak(_subject_heads_from_text(revised)):
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
    if _max_meta_subject_streak(_subject_heads_from_text(original)) > _max_meta_subject_streak(_subject_heads_from_text(revised)):
        score += 1
    return score


def _infer_style_variation_score(text: str) -> int:
    openers = [_extract_opener(sentence) for sentence in split_sentences(text)]
    unique_openers = len({opener for opener in openers if opener})
    repeated = _count_repeated_openers(text)
    blacklist = _count_blacklisted_templates(text)
    return max(0, 8 + unique_openers - repeated * 2 - blacklist * 2)


def _style_variation_score_from_stats(
    selected_variants: list[str],
    repeated_rule_families: int,
    template_family_counts: Counter[str],
    max_variant_repeat: int,
    revised: str,
) -> int:
    unique_variants = len(set(selected_variants))
    repeated_template = max(template_family_counts.values(), default=0)
    base = 8 + min(unique_variants, 4)
    base -= max(0, repeated_rule_families - 1) * 2
    base -= max(0, max_variant_repeat - 1) * 2
    base -= max(0, repeated_template - 1) * 2
    base -= _count_blacklisted_templates(revised) * 2
    return max(0, base)


def _looks_like_prefix_only_change(original: str, revised: str) -> bool:
    original_sentences = split_sentences(original)
    revised_sentences = split_sentences(revised)
    if len(original_sentences) != len(revised_sentences) or not original_sentences:
        return False

    changed = 0
    for left, right in zip(original_sentences, revised_sentences):
        if _normalize_sentence(left) == _normalize_sentence(right):
            continue
        changed += 1
        left_body = _LEADING_IMPLICATION_RE.sub("", left.strip())
        right_body = _LEADING_IMPLICATION_RE.sub("", right.strip())
        left_body = re.sub(r"^(?:与此同时|同时|此外|另外|在这一过程中)[，,\s]*", "", left_body)
        right_body = re.sub(r"^(?:与此同时|同时|此外|另外|在这一过程中)[，,\s]*", "", right_body)
        if _normalize_sentence(left_body) != _normalize_sentence(right_body):
            return False
    return changed > 0


def _count_implication_openers(text: str) -> int:
    return sum(
        1 for sentence in split_sentences(text) if _LEADING_IMPLICATION_RE.match(sentence.strip())
    )


def _validate_block_obligations(
    guidance: GuidanceReport,
    candidates: list[BlockRewriteCandidate],
) -> tuple[list[str], list[int]]:
    if not candidates:
        return [], []

    candidate_by_id = {candidate.block_id: candidate for candidate in candidates}
    problems: list[str] = []
    failed_block_ids: list[int] = []
    for policy in guidance.block_policies:
        if policy.edit_policy not in {"light_edit", "rewritable"}:
            continue
        if not policy.required_structural_actions:
            continue
        candidate = candidate_by_id.get(policy.block_id)
        if candidate is None:
            problems.append(
                f"Block {policy.block_id} is missing a rewritten candidate for required rewrite obligations."
            )
            failed_block_ids.append(policy.block_id)
            continue
        if not candidate.required_actions_met:
            if (
                policy.rewrite_depth == "developmental_rewrite"
                and candidate.effective_change
                and (candidate.sentence_level_changes >= 1 or candidate.cluster_changes >= 1)
            ):
                continue
            problems.append(
                f"Block {policy.block_id} did not satisfy required rewrite obligations: {candidate.missing_required_actions}."
            )
            failed_block_ids.append(policy.block_id)
    return problems, failed_block_ids


def _count_blacklisted_templates(text: str) -> int:
    text = strip_non_body_markdown_lines(text)
    count = 0
    if "在此基础上" in text:
        count += text.count("在此基础上")
    if "这也意味着" in text:
        count += text.count("这也意味着")
    if "进一步来看" in text:
        count += text.count("进一步来看")
    if "综上" in text:
        count += text.count("综上")
    count += len(re.findall(r"从[^。；，]{1,20}角度看", text))
    return count


ReviewResult = ReviewReport
