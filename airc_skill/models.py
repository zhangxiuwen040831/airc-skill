from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .config import RewriteMode

if TYPE_CHECKING:
    from .reviewer import ReviewReport


@dataclass(frozen=True)
class BlockPolicy:
    block_id: int
    block_type: str
    risk_level: str
    edit_policy: str
    rewrite_depth: str
    rewrite_intensity: str
    preview: str
    protected_items: list[str]
    recommended_actions: list[str]
    required_structural_actions: list[str]
    required_discourse_actions: list[str]
    required_minimum_sentence_level_changes: int
    required_minimum_cluster_changes: int
    optional_actions: list[str]
    forbidden_actions: list[str]
    notes: list[str]
    original_text: str = ""
    should_rewrite: bool = False
    revision_pattern: list[str] = field(default_factory=list)
    chapter_id: int = 0
    chapter_title: str = ""
    chapter_type: str = "unknown"
    chapter_rewrite_priority: str = "medium"
    chapter_rewrite_quota: dict[str, object] = field(default_factory=dict)
    chapter_rewrite_intensity: str = "medium"
    paragraph_sentence_roles: list[str] = field(default_factory=list)
    opening_rewrite_allowed: bool = True
    opening_reorder_allowed: bool = True
    topic_sentence_text: str = ""
    high_sensitivity_prose: bool = False

    @property
    def block_index(self) -> int:
        return self.block_id

    @property
    def block_kind(self) -> str:
        return self.block_type

    @property
    def prohibited_actions(self) -> list[str]:
        return self.forbidden_actions

    @property
    def reasons(self) -> list[str]:
        return self.notes

    @property
    def protected_terms(self) -> list[str]:
        return self.protected_items


@dataclass(frozen=True)
class GuidanceReport:
    source_path: Path | None
    document_risk: str
    document_scale: str
    body_block_ids: list[int]
    body_blocks_total: int
    body_sentences_total: int
    body_characters: int
    block_policies: list[BlockPolicy]
    do_not_touch_blocks: list[BlockPolicy]
    high_risk_blocks: list[BlockPolicy]
    light_edit_blocks: list[BlockPolicy]
    rewritable_blocks: list[BlockPolicy]
    rewrite_actions_by_block: dict[int, list[str]]
    core_protected_terms: list[str]
    core_protected_patterns: list[str]
    format_protected_patterns: list[str]
    naturalness_priorities: list[str]
    agent_notes: list[str]
    write_gate_preconditions: list[str]
    format_integrity_status: dict[str, object]
    naturalness_review: dict[str, object]
    write_gate_decision: str
    chapter_policy_summary: list[dict[str, object]] = field(default_factory=list)

    @property
    def block_guidance(self) -> list[BlockPolicy]:
        return self.block_policies

    @property
    def rewrite_candidate_blocks(self) -> list[BlockPolicy]:
        return [*self.light_edit_blocks, *self.rewritable_blocks]

    @property
    def rewrite_actions(self) -> dict[int, list[str]]:
        return self.rewrite_actions_by_block

    @property
    def risk_level(self) -> str:
        return self.document_risk

    def with_review(self, review: "ReviewReport", output_written: bool) -> "GuidanceReport":
        format_status = dict(self.format_integrity_status)
        format_status.update(
            {
                "post_review": "pass" if review.format_integrity else "fail",
                "heading_format_integrity_check": review.integrity_checks.get("heading_format_integrity_check", True),
                "english_spacing_integrity_check": review.english_spacing_integrity,
                "placeholder_integrity_check": review.placeholder_integrity,
                "caption_punctuation_integrity_check": review.caption_punctuation_integrity,
                "linebreak_whitespace_integrity_check": review.integrity_checks.get(
                    "linebreak_whitespace_integrity_check", True
                ),
                "output_written": output_written,
            }
        )
        naturalness = dict(self.naturalness_review)
        naturalness.update(
            {
                "reviewer_decision": review.decision,
                "template_risk": review.template_risk,
                "naturalness_risk": review.naturalness_risk,
                "repeated_subject_risk": review.repeated_subject_risk,
                "template_warning": review.template_warning,
                "naturalness_score": review.naturalness_score,
                "discourse_change_score": review.discourse_change_score,
                "cluster_rewrite_score": review.cluster_rewrite_score,
                "style_variation_score": review.style_variation_score,
                "rewrite_coverage": review.rewrite_coverage,
                "body_rewrite_coverage": review.body_rewrite_coverage,
                "body_changed_blocks": review.body_changed_blocks,
                "body_blocks_total": review.body_blocks_total,
                "body_changed_sentences": review.body_changed_sentences,
                "body_sentences_total": review.body_sentences_total,
                "body_discourse_change_score": review.body_discourse_change_score,
                "body_cluster_rewrite_score": review.body_cluster_rewrite_score,
                "document_scale": review.document_scale,
                "rewrite_quota_met": review.rewrite_quota_met,
                "rewrite_quota_reason_codes": review.rewrite_quota_reason_codes,
                "chapter_rewrite_metrics": review.chapter_rewrite_metrics,
                "chapter_policy_consistency_check": review.chapter_policy_consistency_check,
                "chapter_rewrite_quota_check": review.chapter_rewrite_quota_check,
                "chapter_rewrite_quota_reason_codes": review.chapter_rewrite_quota_reason_codes,
                "human_like_variation": review.human_like_variation,
                "non_uniform_rewrite_distribution": review.non_uniform_rewrite_distribution,
                "sentence_cluster_changes_present": review.sentence_cluster_changes_present,
                "narrative_flow_changed": review.narrative_flow_changed,
                "paragraph_topic_sentence_preserved": review.paragraph_topic_sentence_preserved,
                "paragraph_opening_style_valid": review.paragraph_opening_style_valid,
                "paragraph_skeleton_consistent": review.paragraph_skeleton_consistent,
                "no_dangling_opening_sentence": review.no_dangling_opening_sentence,
                "topic_sentence_not_demoted_to_mid_paragraph": review.topic_sentence_not_demoted_to_mid_paragraph,
                "paragraph_skeleton_review": review.paragraph_skeleton_review,
                "local_transition_natural": review.local_transition_natural,
                "local_discourse_not_flat": review.local_discourse_not_flat,
                "sentence_uniformity_reduced": review.sentence_uniformity_reduced,
                "revision_realism_present": review.revision_realism_present,
                "stylistic_uniformity_controlled": review.stylistic_uniformity_controlled,
                "support_sentence_texture_varied": review.support_sentence_texture_varied,
                "paragraph_voice_variation_present": review.paragraph_voice_variation_present,
                "academic_cliche_density_controlled": review.academic_cliche_density_controlled,
                "local_revision_realism": review.local_revision_realism,
                "sentence_completeness_preserved": review.sentence_completeness_preserved,
                "paragraph_readability_preserved": review.paragraph_readability_preserved,
                "no_dangling_support_sentences": review.no_dangling_support_sentences,
                "no_fragment_like_conclusion_sentences": review.no_fragment_like_conclusion_sentences,
                "sentence_readability": review.sentence_readability,
                "semantic_role_integrity_preserved": review.semantic_role_integrity_preserved,
                "enumeration_integrity_preserved": review.enumeration_integrity_preserved,
                "scaffolding_phrase_density_controlled": review.scaffolding_phrase_density_controlled,
                "over_abstracted_subject_risk_controlled": review.over_abstracted_subject_risk_controlled,
                "semantic_role_integrity": review.semantic_role_integrity,
                "assertion_strength_preserved": review.assertion_strength_preserved,
                "appendix_like_support_controlled": review.appendix_like_support_controlled,
                "authorial_stance_present": review.authorial_stance_present,
                "authorial_intent": review.authorial_intent,
                "evidence_fidelity_preserved": review.evidence_fidelity_preserved,
                "unsupported_expansion_controlled": review.unsupported_expansion_controlled,
                "thesis_tone_restrained": review.thesis_tone_restrained,
                "metaphor_or_storytelling_controlled": review.metaphor_or_storytelling_controlled,
                "authorial_claim_risk_controlled": review.authorial_claim_risk_controlled,
                "evidence_fidelity": review.evidence_fidelity,
                "bureaucratic_opening_controlled": review.bureaucratic_opening_controlled,
                "explicit_subject_chain_controlled": review.explicit_subject_chain_controlled,
                "overstructured_syntax_controlled": review.overstructured_syntax_controlled,
                "main_clause_position_reasonable": review.main_clause_position_reasonable,
                "slogan_like_goal_phrase_controlled": review.slogan_like_goal_phrase_controlled,
                "academic_sentence_naturalization": review.academic_sentence_naturalization,
                "l2_style_profile": review.l2_style_profile,
                "target_style_alignment": review.target_style_alignment,
                "target_style_alignment_score": review.target_style_alignment.get("target_style_alignment_score", 1.0),
                "revision_pattern_distribution": review.revision_pattern_distribution,
                "human_noise_marks": review.human_noise_marks,
            }
        )
        return replace(
            self,
            format_integrity_status=format_status,
            naturalness_review=naturalness,
            write_gate_decision=review.decision,
        )


@dataclass(frozen=True)
class RewriteCandidate:
    block_id: int
    original_text: str
    rewritten_text: str
    actions_used: list[str]
    discourse_actions_used: list[str]
    protected_items_respected: bool
    structural_actions: list[str]
    high_value_actions: list[str]
    template_family_usage: dict[str, int]
    subject_chain_actions: list[str]
    effective_change: bool
    sentence_level_changes: int
    cluster_changes: int
    revision_patterns: list[str]
    required_actions_met: bool
    missing_required_actions: list[str]
    notes: list[str]
    mode_used: str


@dataclass(frozen=True)
class RewriteExecutionReport:
    rewritten_text: str
    block_candidates: list[RewriteCandidate]
    rewrite_stats: list[object]
    mode_requested: RewriteMode | str
    mode_used: RewriteMode | str
    effective_change: bool
    changed_block_ids: list[int]
    candidate_count: int
    selected_candidate_reason: str
    convenience_mode: bool
    block_failures: list[str] = field(default_factory=list)
    reviewed: bool = False


@dataclass(frozen=True)
class ReviewReport:
    ok: bool
    decision: str
    problems: list[str]
    warnings: list[str]
    meaningful_change: bool
    effective_change: bool
    changed_characters: int
    diff_spans: int
    sentence_level_change: bool
    depth_sufficient: bool
    template_risk: bool
    naturalness_risk: bool
    structural_action_count: int
    high_value_action_count: int
    high_value_actions: list[str]
    discourse_change_score: int
    cluster_rewrite_score: int
    style_variation_score: int
    rewrite_coverage: float
    body_rewrite_coverage: float
    body_changed_blocks: int
    body_blocks_total: int
    body_changed_sentences: int
    body_sentences_total: int
    body_changed_block_ratio: float
    body_discourse_change_score: int
    body_cluster_rewrite_score: int
    body_developmental_changed_blocks: int
    body_developmental_blocks_total: int
    document_scale: str
    rewrite_quota_met: bool
    rewrite_quota_reason_codes: list[str]
    required_body_rewrite_coverage: float
    required_body_changed_block_ratio: float
    required_body_discourse_change_score: int
    required_body_cluster_rewrite_score: int
    chapter_rewrite_metrics: list[dict[str, object]]
    chapter_policy_consistency_check: bool
    chapter_rewrite_quota_check: bool
    chapter_rewrite_quota_reason_codes: list[str]
    human_like_variation: bool
    non_uniform_rewrite_distribution: bool
    sentence_cluster_changes_present: bool
    narrative_flow_changed: bool
    paragraph_topic_sentence_preserved: bool
    paragraph_opening_style_valid: bool
    paragraph_skeleton_consistent: bool
    no_dangling_opening_sentence: bool
    topic_sentence_not_demoted_to_mid_paragraph: bool
    paragraph_skeleton_review: dict[str, object]
    local_transition_natural: bool
    local_discourse_not_flat: bool
    sentence_uniformity_reduced: bool
    revision_realism_present: bool
    stylistic_uniformity_controlled: bool
    support_sentence_texture_varied: bool
    paragraph_voice_variation_present: bool
    academic_cliche_density_controlled: bool
    local_revision_realism: dict[str, object]
    sentence_completeness_preserved: bool
    paragraph_readability_preserved: bool
    no_dangling_support_sentences: bool
    no_fragment_like_conclusion_sentences: bool
    sentence_readability: dict[str, object]
    semantic_role_integrity_preserved: bool
    enumeration_integrity_preserved: bool
    scaffolding_phrase_density_controlled: bool
    over_abstracted_subject_risk_controlled: bool
    semantic_role_integrity: dict[str, object]
    assertion_strength_preserved: bool
    appendix_like_support_controlled: bool
    authorial_stance_present: bool
    authorial_intent: dict[str, object]
    evidence_fidelity_preserved: bool
    unsupported_expansion_controlled: bool
    thesis_tone_restrained: bool
    metaphor_or_storytelling_controlled: bool
    authorial_claim_risk_controlled: bool
    evidence_fidelity: dict[str, object]
    bureaucratic_opening_controlled: bool
    explicit_subject_chain_controlled: bool
    overstructured_syntax_controlled: bool
    main_clause_position_reasonable: bool
    slogan_like_goal_phrase_controlled: bool
    academic_sentence_naturalization: dict[str, object]
    l2_style_profile: dict[str, object]
    target_style_alignment: dict[str, object]
    revision_pattern_distribution: dict[str, int]
    human_noise_marks: list[str]
    prefix_only_rewrite: bool
    repeated_subject_risk: bool
    template_warning: bool
    template_issue: str | None
    core_content_integrity: bool
    format_integrity: bool
    title_integrity: bool
    formula_integrity: bool
    terminology_integrity: bool
    citation_integrity: bool
    numeric_integrity: bool
    path_integrity: bool
    heading_format_integrity: bool
    english_spacing_integrity: bool
    placeholder_integrity: bool
    caption_punctuation_integrity: bool
    markdown_symbol_integrity: bool
    linebreak_whitespace_integrity: bool
    naturalness_score: int
    suggested_fallback: str
    write_gate_ready: bool
    failed_block_ids: list[int]
    integrity_checks: dict[str, bool]
    natural_revision_checklist: dict[str, bool] = field(default_factory=dict)

    @property
    def core_content_integrity_ok(self) -> bool:
        return self.core_content_integrity

    @property
    def format_integrity_ok(self) -> bool:
        return self.format_integrity


@dataclass(frozen=True)
class WriteGateDecision:
    write_allowed: bool
    decision: str
    reason_codes: list[str]
    warnings: list[str]
    selected_candidate_reason: str
