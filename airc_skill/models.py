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
