from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, SKILL_PRESETS, RewriteMode, get_skill_preset
from .models import GuidanceReport, RewriteExecutionReport, ReviewReport, WriteGateDecision
from .natural_revision_profile import ACADEMIC_NATURAL_STUDENTLIKE, get_natural_revision_profile
from .revision_doctrine import doctrine_for_style


@dataclass(frozen=True)
class SkillInputSchema:
    source_path: str
    output_path: str | None = None
    mode: str = "balanced"
    preset: str = "academic_natural"
    rewrite_scope: str = "full"
    preservation_level: str = "strict"
    language: str = "mixed"
    target_style: str = "academic_natural"
    target_style_file: str | None = None
    max_retry_passes: int = DEFAULT_CONFIG.max_rewrite_passes
    emit_agent_context: bool = False
    emit_json_report: bool = True

    @property
    def text_path(self) -> str:
        return self.source_path

    @classmethod
    def from_path(
        cls,
        source_path: str | Path,
        preset: str = "academic_natural",
        mode: str | None = None,
        target_style: str | None = None,
        rewrite_scope: str | None = None,
        preservation_level: str | None = None,
        output_path: str | Path | None = None,
        language: str = "mixed",
        target_style_file: str | Path | None = None,
        max_retry_passes: int | None = None,
        emit_agent_context: bool = False,
        emit_json_report: bool = True,
    ) -> "SkillInputSchema":
        preset_config = get_skill_preset(preset)
        return cls(
            source_path=str(source_path),
            output_path=str(output_path) if output_path else None,
            mode=mode or preset_config.mode.value,
            preset=preset_config.name,
            rewrite_scope=rewrite_scope or preset_config.rewrite_scope,
            preservation_level=preservation_level or preset_config.preservation_level,
            language=language,
            target_style=target_style or preset_config.target_style,
            target_style_file=str(target_style_file) if target_style_file else None,
            max_retry_passes=max_retry_passes if max_retry_passes is not None else preset_config.max_retry_passes,
            emit_agent_context=emit_agent_context,
            emit_json_report=emit_json_report,
        )

    def resolved_mode(self) -> RewriteMode:
        if self.mode == "custom":
            return get_skill_preset(self.preset).mode
        return RewriteMode.from_value(self.mode)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillInputSchema":
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in allowed})


@dataclass(frozen=True)
class AgentBlockInstruction:
    block_id: int
    block_type: str
    policy: str
    rewrite_intensity: str
    rewrite_depth: str
    chapter_title: str
    chapter_type: str
    chapter_rewrite_priority: str
    chapter_rewrite_quota: dict[str, object]
    chapter_rewrite_intensity: str
    paragraph_sentence_roles: list[str]
    opening_rewrite_allowed: bool
    opening_reorder_allowed: bool
    topic_sentence_text: str
    high_sensitivity_prose: bool
    revision_pattern: list[str]
    required_actions: list[str]
    required_sentence_actions: list[str]
    required_cluster_actions: list[str]
    required_discourse_actions: list[str]
    optional_actions: list[str]
    forbidden_actions: list[str]
    clarity_constraints: list[str]
    preserve_items: list[str]
    forbidden_patterns: list[str]
    agent_notes: list[str]
    pass_conditions: dict[str, object]
    preview: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentInstructionBundle:
    blocks: list[AgentBlockInstruction]
    text: str
    global_rules: list[str]
    minimum_rewrite_coverage: float
    retry_policy: dict[str, object]
    document_scale: str = "short"
    body_blocks_total: int = 0
    body_sentences_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocks": [block.to_dict() for block in self.blocks],
            "text": self.text,
            "global_rules": list(self.global_rules),
            "minimum_rewrite_coverage": self.minimum_rewrite_coverage,
            "retry_policy": dict(self.retry_policy),
            "document_scale": self.document_scale,
            "body_blocks_total": self.body_blocks_total,
            "body_sentences_total": self.body_sentences_total,
        }

    def __contains__(self, item: object) -> bool:
        return isinstance(item, str) and item in self.text

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class SkillExecutionPlan:
    document_risk: str
    block_policies: list[dict[str, Any]]
    rewrite_intensity_by_block: dict[int, str]
    required_actions_by_block: dict[int, list[str]]
    forbidden_actions_by_block: dict[int, list[str]]
    protected_terms: list[str]
    protected_patterns: list[str]
    format_protected_patterns: list[str]
    write_gate_preconditions: list[str]
    agent_instructions: list[dict[str, Any]]
    human_agent_instructions: str
    revision_doctrine: dict[str, Any]
    minimum_rewrite_coverage: float
    minor_risk_rewrite_coverage: float
    mode: str
    target_style: str
    rewrite_scope: str
    preservation_level: str
    language: str
    preset: str
    document_scale: str = "short"
    body_blocks_total: int = 0
    body_sentences_total: int = 0
    body_rewrite_quota: dict[str, object] | None = None
    chapter_policy_summary: list[dict[str, object]] | None = None

    @property
    def agent_instruction(self) -> str:
        return self.human_agent_instructions

    @property
    def core_protected_terms(self) -> list[str]:
        return self.protected_terms

    @property
    def core_protected_patterns(self) -> list[str]:
        return self.protected_patterns

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["agent_instruction"] = self.human_agent_instructions
        return payload


@dataclass(frozen=True)
class SkillOutputSchema:
    status: str
    rewritten_file_path: str | None
    report_file_path: str | None
    input_normalization: dict[str, Any]
    rewrite_coverage: float
    discourse_change_score: int
    cluster_rewrite_score: int
    blocks_changed: list[int]
    blocks_skipped: list[int]
    blocks_rejected: list[int]
    warnings: list[str]
    failed_obligations: dict[int, list[str]]
    write_gate_decision: str
    write_allowed: bool
    decision: str
    reason_codes: list[str]
    rewrite_report_path: str | None = None
    body_rewrite_coverage: float = 0.0
    body_changed_blocks: int = 0
    body_blocks_total: int = 0
    body_changed_sentences: int = 0
    body_sentences_total: int = 0
    body_discourse_change_score: int = 0
    body_cluster_rewrite_score: int = 0
    document_scale: str = "short"
    rewrite_quota_met: bool = False
    human_like_variation: bool = False
    non_uniform_rewrite_distribution: bool = False
    sentence_cluster_changes_present: bool = False
    narrative_flow_changed: bool = False
    revision_pattern_distribution: dict[str, int] = field(default_factory=dict)
    chapter_rewrite_metrics: list[dict[str, Any]] = field(default_factory=list)
    chapter_policy_consistency_check: bool = False
    chapter_rewrite_quota_check: bool = False
    chapter_rewrite_quota_reason_codes: list[str] = field(default_factory=list)
    paragraph_topic_sentence_preserved: bool = False
    paragraph_opening_style_valid: bool = False
    paragraph_skeleton_consistent: bool = False
    no_dangling_opening_sentence: bool = False
    topic_sentence_not_demoted_to_mid_paragraph: bool = False
    paragraph_skeleton_review: dict[str, Any] = field(default_factory=dict)
    local_transition_natural: bool = False
    local_discourse_not_flat: bool = False
    sentence_uniformity_reduced: bool = False
    revision_realism_present: bool = False
    stylistic_uniformity_controlled: bool = False
    support_sentence_texture_varied: bool = False
    paragraph_voice_variation_present: bool = False
    academic_cliche_density_controlled: bool = False
    local_revision_realism: dict[str, Any] = field(default_factory=dict)
    sentence_completeness_preserved: bool = False
    paragraph_readability_preserved: bool = False
    no_dangling_support_sentences: bool = False
    no_fragment_like_conclusion_sentences: bool = False
    sentence_readability: dict[str, Any] = field(default_factory=dict)
    semantic_role_integrity_preserved: bool = False
    enumeration_integrity_preserved: bool = False
    scaffolding_phrase_density_controlled: bool = False
    over_abstracted_subject_risk_controlled: bool = False
    semantic_role_integrity: dict[str, Any] = field(default_factory=dict)
    assertion_strength_preserved: bool = False
    appendix_like_support_controlled: bool = False
    authorial_stance_present: bool = False
    authorial_intent: dict[str, Any] = field(default_factory=dict)
    evidence_fidelity_preserved: bool = False
    unsupported_expansion_controlled: bool = False
    thesis_tone_restrained: bool = False
    metaphor_or_storytelling_controlled: bool = False
    authorial_claim_risk_controlled: bool = False
    evidence_fidelity: dict[str, Any] = field(default_factory=dict)
    bureaucratic_opening_controlled: bool = False
    explicit_subject_chain_controlled: bool = False
    overstructured_syntax_controlled: bool = False
    main_clause_position_reasonable: bool = False
    slogan_like_goal_phrase_controlled: bool = False
    academic_sentence_naturalization: dict[str, Any] = field(default_factory=dict)
    target_style_alignment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["rewrite_report_path"] is None:
            payload["rewrite_report_path"] = self.report_file_path
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillOutputSchema":
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        data = {key: value for key, value in payload.items() if key in allowed}
        if "rewrite_report_path" not in data:
            data["rewrite_report_path"] = data.get("report_file_path")
        return cls(**data)


@dataclass(frozen=True)
class ExecutionValidationResult:
    ok: bool
    failed_block_ids: list[int]
    failed_obligations: dict[int, list[str]]
    warnings: list[str]
    prefix_only_block_ids: list[int]
    surface_level_block_ids: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_execution_plan(
    guidance: GuidanceReport,
    schema: SkillInputSchema,
) -> SkillExecutionPlan:
    preset = get_skill_preset(schema.preset)
    instruction_bundle = generate_agent_instructions(guidance, schema)
    block_policies: list[dict[str, Any]] = []
    rewrite_intensity_by_block: dict[int, str] = {}
    required_actions_by_block: dict[int, list[str]] = {}
    forbidden_actions_by_block: dict[int, list[str]] = {}

    for block in guidance.block_policies:
        required_actions = _required_actions_for_agent(block.required_structural_actions, block.required_discourse_actions)
        block_policies.append(
            {
                "block_id": block.block_id,
                "block_type": block.block_type,
                "edit_policy": block.edit_policy,
                "risk_level": block.risk_level,
                "rewrite_depth": block.rewrite_depth,
                "rewrite_intensity": block.rewrite_intensity,
                "chapter_title": block.chapter_title,
                "chapter_type": block.chapter_type,
                "chapter_rewrite_priority": block.chapter_rewrite_priority,
                "chapter_rewrite_quota": dict(block.chapter_rewrite_quota),
                "chapter_rewrite_intensity": block.chapter_rewrite_intensity,
                "paragraph_sentence_roles": list(block.paragraph_sentence_roles),
                "opening_rewrite_allowed": block.opening_rewrite_allowed,
                "opening_reorder_allowed": block.opening_reorder_allowed,
                "topic_sentence_text": block.topic_sentence_text,
                "high_sensitivity_prose": block.high_sensitivity_prose,
                "revision_pattern": list(block.revision_pattern),
                "required_actions": required_actions,
                "required_sentence_actions": _required_sentence_actions_for_block(block),
                "required_cluster_actions": _required_cluster_actions_for_block(block),
                "required_discourse_actions": list(block.required_discourse_actions),
                "clarity_constraints": _clarity_constraints_for_block(block),
                "preserve_items": list(block.protected_items),
                "forbidden_patterns": [
                    *block.forbidden_actions,
                    *ACADEMIC_NATURAL_STUDENTLIKE.forbidden_markers,
                ],
                "optional_actions": list(block.optional_actions),
                "forbidden_actions": list(block.forbidden_actions),
                "minimum_sentence_level_changes": block.required_minimum_sentence_level_changes,
                "minimum_cluster_changes": block.required_minimum_cluster_changes,
                "preview": block.preview,
                "notes": list(block.notes),
            }
        )
        rewrite_intensity_by_block[block.block_id] = block.rewrite_intensity
        required_actions_by_block[block.block_id] = required_actions
        forbidden_actions_by_block[block.block_id] = list(block.forbidden_actions)

    return SkillExecutionPlan(
        document_risk=guidance.document_risk,
        block_policies=block_policies,
        rewrite_intensity_by_block=rewrite_intensity_by_block,
        required_actions_by_block=required_actions_by_block,
        forbidden_actions_by_block=forbidden_actions_by_block,
        protected_terms=list(guidance.core_protected_terms),
        protected_patterns=list(guidance.core_protected_patterns),
        format_protected_patterns=list(guidance.format_protected_patterns),
        write_gate_preconditions=list(guidance.write_gate_preconditions),
        agent_instructions=[block.to_dict() for block in instruction_bundle.blocks],
        human_agent_instructions=instruction_bundle.text,
        revision_doctrine=doctrine_for_style(schema.target_style),
        minimum_rewrite_coverage=preset.min_rewrite_coverage,
        minor_risk_rewrite_coverage=DEFAULT_CONFIG.rewrite_coverage_minor_threshold,
        mode=schema.resolved_mode().value,
        target_style=schema.target_style,
        rewrite_scope=schema.rewrite_scope,
        preservation_level=schema.preservation_level,
        language=schema.language,
        preset=schema.preset,
        document_scale=guidance.document_scale,
        body_blocks_total=guidance.body_blocks_total,
        body_sentences_total=guidance.body_sentences_total,
        body_rewrite_quota={
            "body_rewrite_coverage_min": preset.min_rewrite_coverage,
            "uses_body_only_denominator": True,
            "excludes": [
                "headings",
                "captions",
                "image_markdown",
                "formulas",
                "references",
                "paths",
                "technical_dense_blocks",
            ],
        },
        chapter_policy_summary=list(guidance.chapter_policy_summary) if hasattr(guidance, "chapter_policy_summary") else None,
    )


def generate_agent_instructions(
    guidance: GuidanceReport,
    schema: SkillInputSchema | None = None,
) -> AgentInstructionBundle:
    schema = schema or SkillInputSchema(source_path=str(guidance.source_path or ""), preset="academic_natural")
    preset = get_skill_preset(schema.preset)
    style_profile = get_natural_revision_profile(schema.target_style)
    global_rules = [
        "Do not perform prefix-only or word-only rewrites on rewritable body blocks.",
        "The goal is not light polishing; rebuild ordinary body prose as a human academic editor would.",
        "Rewrite sentence groups and the route of explanation, not only individual words or sentence openers.",
        "Apply the block revision_pattern values, but keep their distribution uneven across the document.",
        "Use sentence clusters of two to five sentences for merge, split, reorder, cause-effect rewrite, or conclusion absorption.",
        "Preserve paragraph skeletons: valid topic sentences may be lightly rewritten, but they stay at the paragraph opening.",
        "Improve local human revision realism: soften overexplicit transitions, reduce sentence uniformity, and rebuild paragraph-internal hierarchy without moving the topic sentence.",
        "Preserve sentence completeness and paragraph readability; never trade a higher realism score for fragment-like support or conclusion sentences.",
        "Never start a rewritten paragraph with a dangling connector such as 并进一步, 在这种情况下, 相应地, 围绕这一点, 其中, 此外, or 由此.",
        "Apply chapter-aware policy: background/significance/review/analysis/conclusion/future-work sections need stronger rewrite; methods, formulas, metrics, experiment setup, deployment, and technical-dense sections stay controlled.",
        "Do not produce uniform sentence-by-sentence edits; keep some precise sentences close and rewrite others more deeply.",
        "For long and very_long documents, do not use conservative fallback; keep escalating body rewrite intensity until quota is met.",
        f"Use the {style_profile.name} doctrine: {style_profile.description}",
        "Reduce 的 / 了 / 在……中 overuse only when the sentence remains precise and fluent.",
        "Weaken mechanical connectors by deletion, absorption, or sentence fusion, not by swapping in another template opener.",
        "Break slogan-like parallelism and repeated sentence skeletons without changing the claim.",
        "Every rewritable block must execute its required discourse or cluster actions unless the block is protected.",
        "Preserve headings, formulas, terms, citations, numbers, paths, captions, placeholders, and Markdown structure.",
        "If a rewrite risks protected content, keep the original block and report the reason.",
        "Do not use colloquial/web-style markers: " + ", ".join(style_profile.forbidden_markers),
    ]
    if style_profile.name == "zh_academic_l2_mild":
        global_rules.extend(
            [
                "For zh_academic_l2_mild, make ordinary Chinese prose mildly explanatory and slightly less compressed, as if written carefully by a less fluent Chinese academic writer.",
                "Allow mild 的 / 了 / 来 / 进行 / 能够 / 通过……来 texture, but never create broken grammar, web-style phrasing, or unsupported facts.",
                "Do not apply L2 texture to protected, formula-dense, metric, path, checkpoint, or technical-definition blocks.",
            ]
        )
    block_instructions: list[AgentBlockInstruction] = []
    text_lines = [
        "AIRC execution protocol",
        f"- preset: {preset.name}",
        f"- mode: {schema.resolved_mode().value}",
        f"- target_style: {schema.target_style}",
        f"- language: {schema.language}",
        f"- rewrite_scope: {schema.rewrite_scope}",
        f"- preservation_level: {schema.preservation_level}",
        f"- document_scale: {guidance.document_scale}",
        f"- body_blocks_total: {guidance.body_blocks_total}",
        f"- body_sentences_total: {guidance.body_sentences_total}",
        f"- minimum_rewrite_coverage: {preset.min_rewrite_coverage:.2f}",
        f"- minor_risk_rewrite_coverage: {DEFAULT_CONFIG.rewrite_coverage_minor_threshold:.2f}",
        "",
        "Mandatory rules:",
    ]
    text_lines.extend(f"- {rule}" for rule in global_rules)
    text_lines.append("")
    text_lines.append("Block execution plan:")

    for block in guidance.block_policies:
        required = _required_actions_for_agent(block.required_structural_actions, block.required_discourse_actions)
        pass_conditions = {
            "sentence_level_changes_at_least": block.required_minimum_sentence_level_changes,
            "cluster_changes_at_least": block.required_minimum_cluster_changes,
            "required_actions": required,
            "revision_pattern": list(block.revision_pattern),
            "chapter_type": block.chapter_type,
            "chapter_rewrite_priority": block.chapter_rewrite_priority,
            "chapter_rewrite_quota": dict(block.chapter_rewrite_quota),
            "chapter_rewrite_intensity": block.chapter_rewrite_intensity,
            "paragraph_sentence_roles": list(block.paragraph_sentence_roles),
            "opening_rewrite_allowed": block.opening_rewrite_allowed,
            "opening_reorder_allowed": block.opening_reorder_allowed,
            "topic_sentence_text": block.topic_sentence_text,
            "high_sensitivity_prose": block.high_sensitivity_prose,
            "must_keep_topic_sentence_at_opening": not block.opening_reorder_allowed,
            "must_preserve_sentence_completeness": True,
            "must_preserve_paragraph_readability": block.high_sensitivity_prose,
            "must_avoid_uniform_rewrite": block.edit_policy in {"light_edit", "rewritable"},
            "must_preserve_protected_items": True,
            "style_profile": style_profile.name,
            "must_not_colloquialize": True,
            "natural_revision_checklist": style_profile.checklist,
        }
        if block.edit_policy in {"do_not_touch", "high_risk"}:
            action_line = "preserve verbatim; do not rewrite narration"
        elif block.rewrite_depth == "developmental_rewrite":
            action_line = "perform developmental rewrite with sentence-cluster restructuring"
        else:
            action_line = "perform light sentence-level editing only"

        instruction = AgentBlockInstruction(
            block_id=block.block_id,
            block_type=block.block_type,
            policy=block.edit_policy,
            rewrite_intensity=block.rewrite_intensity,
            rewrite_depth=block.rewrite_depth,
            chapter_title=block.chapter_title,
            chapter_type=block.chapter_type,
            chapter_rewrite_priority=block.chapter_rewrite_priority,
            chapter_rewrite_quota=dict(block.chapter_rewrite_quota),
            chapter_rewrite_intensity=block.chapter_rewrite_intensity,
            paragraph_sentence_roles=list(block.paragraph_sentence_roles),
            opening_rewrite_allowed=block.opening_rewrite_allowed,
            opening_reorder_allowed=block.opening_reorder_allowed,
            topic_sentence_text=block.topic_sentence_text,
            high_sensitivity_prose=block.high_sensitivity_prose,
            revision_pattern=list(block.revision_pattern),
            required_actions=required,
            required_sentence_actions=_required_sentence_actions_for_block(block),
            required_cluster_actions=_required_cluster_actions_for_block(block),
            required_discourse_actions=list(block.required_discourse_actions),
            optional_actions=list(block.optional_actions),
            forbidden_actions=list(block.forbidden_actions),
            clarity_constraints=_clarity_constraints_for_block(block),
            preserve_items=list(block.protected_items),
            forbidden_patterns=[
                *block.forbidden_actions,
                *style_profile.forbidden_markers,
            ],
            agent_notes=list(block.notes),
            pass_conditions=pass_conditions,
            preview=block.preview,
        )
        block_instructions.append(instruction)
        text_lines.extend(
            [
                f"Block {block.block_id}:",
                f"- type: {block.edit_policy}",
                f"- block_kind: {block.block_type}",
                f"- chapter: {block.chapter_title} ({block.chapter_type}, {block.chapter_rewrite_priority})",
                f"- chapter_quota: {block.chapter_rewrite_quota}",
                f"- paragraph_sentence_roles: {', '.join(block.paragraph_sentence_roles) if block.paragraph_sentence_roles else 'none'}",
                f"- high_sensitivity_prose: {block.high_sensitivity_prose}",
                (
                    "- opening_policy: "
                    f"rewrite_allowed={block.opening_rewrite_allowed}, "
                    f"reorder_allowed={block.opening_reorder_allowed}"
                ),
                f"- intensity: {block.rewrite_intensity}",
                f"- revision_pattern: {', '.join(block.revision_pattern) if block.revision_pattern else 'none'}",
                f"- action_policy: {action_line}",
                f"- required: {', '.join(required) if required else 'none'}",
                (
                    "- required_sentence_actions: "
                    f"{', '.join(_required_sentence_actions_for_block(block)) or 'none'}"
                ),
                (
                    "- required_cluster_actions: "
                    f"{', '.join(_required_cluster_actions_for_block(block)) or 'none'}"
                ),
                (
                    "- clarity_constraints: "
                    f"{'; '.join(_clarity_constraints_for_block(block)) or 'none'}"
                ),
                f"- optional: {', '.join(block.optional_actions) if block.optional_actions else 'none'}",
                f"- forbid: {', '.join(block.forbidden_actions) if block.forbidden_actions else 'none'}",
                (
                    "- pass_conditions: "
                    f"sentence>={block.required_minimum_sentence_level_changes}, "
                    f"cluster>={block.required_minimum_cluster_changes}"
                ),
                f"- notes: {'; '.join(block.notes) if block.notes else 'none'}",
                f"- natural_profile: {style_profile.name}",
            ]
        )
        if not block.opening_reorder_allowed:
            text_lines.append("- opening_guard: keep the topic sentence at the paragraph start; rewrite later support/evidence sentences instead.")

    retry_policy = {
        "max_retry_passes": schema.max_retry_passes,
        "escalate_when_coverage_below": preset.min_rewrite_coverage,
        "escalate_when_discourse_below": DEFAULT_CONFIG.developmental_min_discourse_score,
        "retry_intensity": "high",
        "force_actions": ["sentence_cluster_rewrite", "narrative_flow_rebuilder"],
        "force_revision_patterns": ["merge", "reorder", "reframe", "partial_keep", "rewrite_all"],
        "document_scale": guidance.document_scale,
        "body_only_quota": True,
        "conservative_fallback_allowed": guidance.document_scale not in {"long", "very_long"},
    }
    text_lines.extend(
        [
            "",
            "Final self-check before review:",
            "- body_rewrite_coverage meets the scale-aware threshold for ordinary prose blocks only.",
            "- body_changed_blocks is large enough for the document scale.",
            "- at least one cluster-level change appears in developmental rewrite blocks.",
            "- revision patterns are visibly non-uniform across editable body blocks.",
            "- human-like variation is present: some sentences are partly kept, some clusters are merged/split/reordered, and some blocks are rewritten more deeply.",
            "- paragraph openings remain standalone topic sentences where the source paragraph already had one.",
            "- local transitions feel natural and supporting sentences are not all equally complete or flat.",
            "- every rewritten sentence can stand as a complete academic sentence; support and conclusion sentences are not fragments.",
            "- no paragraph begins with a dangling connector such as 并进一步, 围绕这一点, 在这种情况下, 其中, 此外, or 由此.",
            "- no rewritable block was skipped without a protection reason.",
            "- all protected core and format items remain byte-stable in meaning and Markdown structure.",
        ]
    )
    if guidance.document_scale in {"long", "very_long"}:
        text_lines.extend(
            [
                "- long/very_long rule: rewrite broadly across the body; a small set of polished paragraphs must fail self-check.",
                "- long/very_long rule: developmental rewrite must appear in part of the body, not only light sentence cleanup.",
                "- long/very_long rule: cluster-level rewrite and non-uniform human-like variation must both be present.",
            ]
        )
    return AgentInstructionBundle(
        blocks=block_instructions,
        text="\n".join(text_lines),
        global_rules=global_rules,
        minimum_rewrite_coverage=preset.min_rewrite_coverage,
        retry_policy=retry_policy,
        document_scale=guidance.document_scale,
        body_blocks_total=guidance.body_blocks_total,
        body_sentences_total=guidance.body_sentences_total,
    )


def build_output_schema(
    rewrite_report: RewriteExecutionReport,
    review: ReviewReport,
    write_gate: WriteGateDecision,
    rewritten_file_path: str | Path | None,
    rewrite_report_path: str | Path | None = None,
    input_normalization: dict[str, Any] | None = None,
    execution_validation: ExecutionValidationResult | None = None,
) -> SkillOutputSchema:
    changed = list(rewrite_report.changed_block_ids)
    attempted = [candidate.block_id for candidate in rewrite_report.block_candidates]
    skipped = [block_id for block_id in attempted if block_id not in changed]
    failed_obligations = (
        execution_validation.failed_obligations
        if execution_validation is not None
        else _failed_obligations_from_candidates(rewrite_report)
    )
    blocks_rejected = sorted(set([*review.failed_block_ids, *failed_obligations.keys()]))
    report_path = str(rewrite_report_path) if rewrite_report_path else None
    return SkillOutputSchema(
        status="success" if write_gate.write_allowed else "failed",
        rewritten_file_path=str(rewritten_file_path) if rewritten_file_path else None,
        report_file_path=report_path,
        rewrite_report_path=report_path,
        input_normalization=input_normalization or {},
        rewrite_coverage=review.rewrite_coverage,
        discourse_change_score=review.discourse_change_score,
        cluster_rewrite_score=review.cluster_rewrite_score,
        blocks_changed=changed,
        blocks_skipped=skipped,
        blocks_rejected=blocks_rejected,
        warnings=[*review.problems, *review.warnings, *write_gate.warnings],
        failed_obligations=failed_obligations,
        write_gate_decision=write_gate.decision,
        write_allowed=write_gate.write_allowed,
        decision=write_gate.decision,
        reason_codes=list(write_gate.reason_codes),
        body_rewrite_coverage=review.body_rewrite_coverage,
        body_changed_blocks=review.body_changed_blocks,
        body_blocks_total=review.body_blocks_total,
        body_changed_sentences=review.body_changed_sentences,
        body_sentences_total=review.body_sentences_total,
        body_discourse_change_score=review.body_discourse_change_score,
        body_cluster_rewrite_score=review.body_cluster_rewrite_score,
        document_scale=review.document_scale,
        rewrite_quota_met=review.rewrite_quota_met,
        human_like_variation=review.human_like_variation,
        non_uniform_rewrite_distribution=review.non_uniform_rewrite_distribution,
        sentence_cluster_changes_present=review.sentence_cluster_changes_present,
        narrative_flow_changed=review.narrative_flow_changed,
        revision_pattern_distribution=dict(review.revision_pattern_distribution),
        chapter_rewrite_metrics=list(review.chapter_rewrite_metrics),
        chapter_policy_consistency_check=review.chapter_policy_consistency_check,
        chapter_rewrite_quota_check=review.chapter_rewrite_quota_check,
        chapter_rewrite_quota_reason_codes=list(review.chapter_rewrite_quota_reason_codes),
        paragraph_topic_sentence_preserved=review.paragraph_topic_sentence_preserved,
        paragraph_opening_style_valid=review.paragraph_opening_style_valid,
        paragraph_skeleton_consistent=review.paragraph_skeleton_consistent,
        no_dangling_opening_sentence=review.no_dangling_opening_sentence,
        topic_sentence_not_demoted_to_mid_paragraph=review.topic_sentence_not_demoted_to_mid_paragraph,
        paragraph_skeleton_review=dict(review.paragraph_skeleton_review),
        local_transition_natural=review.local_transition_natural,
        local_discourse_not_flat=review.local_discourse_not_flat,
        sentence_uniformity_reduced=review.sentence_uniformity_reduced,
        revision_realism_present=review.revision_realism_present,
        stylistic_uniformity_controlled=review.stylistic_uniformity_controlled,
        support_sentence_texture_varied=review.support_sentence_texture_varied,
        paragraph_voice_variation_present=review.paragraph_voice_variation_present,
        academic_cliche_density_controlled=review.academic_cliche_density_controlled,
        local_revision_realism=dict(review.local_revision_realism),
        sentence_completeness_preserved=review.sentence_completeness_preserved,
        paragraph_readability_preserved=review.paragraph_readability_preserved,
        no_dangling_support_sentences=review.no_dangling_support_sentences,
        no_fragment_like_conclusion_sentences=review.no_fragment_like_conclusion_sentences,
        sentence_readability=dict(review.sentence_readability),
        semantic_role_integrity_preserved=review.semantic_role_integrity_preserved,
        enumeration_integrity_preserved=review.enumeration_integrity_preserved,
        scaffolding_phrase_density_controlled=review.scaffolding_phrase_density_controlled,
        over_abstracted_subject_risk_controlled=review.over_abstracted_subject_risk_controlled,
        semantic_role_integrity=dict(review.semantic_role_integrity),
        assertion_strength_preserved=review.assertion_strength_preserved,
        appendix_like_support_controlled=review.appendix_like_support_controlled,
        authorial_stance_present=review.authorial_stance_present,
        authorial_intent=dict(review.authorial_intent),
        evidence_fidelity_preserved=review.evidence_fidelity_preserved,
        unsupported_expansion_controlled=review.unsupported_expansion_controlled,
        thesis_tone_restrained=review.thesis_tone_restrained,
        metaphor_or_storytelling_controlled=review.metaphor_or_storytelling_controlled,
        authorial_claim_risk_controlled=review.authorial_claim_risk_controlled,
        evidence_fidelity=dict(review.evidence_fidelity),
        bureaucratic_opening_controlled=review.bureaucratic_opening_controlled,
        explicit_subject_chain_controlled=review.explicit_subject_chain_controlled,
        overstructured_syntax_controlled=review.overstructured_syntax_controlled,
        main_clause_position_reasonable=review.main_clause_position_reasonable,
        slogan_like_goal_phrase_controlled=review.slogan_like_goal_phrase_controlled,
        academic_sentence_naturalization=dict(review.academic_sentence_naturalization),
        target_style_alignment=dict(review.target_style_alignment),
    )


def validate_execution_against_plan(
    plan: SkillExecutionPlan,
    rewrite_report: RewriteExecutionReport,
    review: ReviewReport | None = None,
) -> ExecutionValidationResult:
    failed_obligations: dict[int, list[str]] = {}
    warnings: list[str] = []
    prefix_only_block_ids: list[int] = []
    surface_level_block_ids: list[int] = []
    candidates_by_id = {candidate.block_id: candidate for candidate in rewrite_report.block_candidates}

    for block in plan.block_policies:
        block_id = int(block["block_id"])
        policy = str(block["edit_policy"])
        if policy not in {"light_edit", "rewritable"}:
            continue
        candidate = candidates_by_id.get(block_id)
        if candidate is None:
            failed_obligations.setdefault(block_id, []).append("editable_block_not_attempted")
            continue
        missing: list[str] = []
        if not candidate.effective_change:
            missing.append("no_effective_change")
        if candidate.sentence_level_changes < int(block["minimum_sentence_level_changes"]):
            missing.append(f"sentence_level_change>={block['minimum_sentence_level_changes']}")
        if policy == "rewritable" and candidate.cluster_changes < int(block["minimum_cluster_changes"]):
            missing.append(f"cluster_change>={block['minimum_cluster_changes']}")
        if policy == "rewritable" and candidate.sentence_level_changes <= 0 and candidate.cluster_changes <= 0:
            missing.append("surface_level_rewrite")
            surface_level_block_ids.append(block_id)
        if (
            policy == "rewritable"
            and candidate.effective_change
            and candidate.cluster_changes <= 0
            and not candidate.discourse_actions_used
        ):
            missing.extend(block["required_actions"])
        if not candidate.protected_items_respected:
            missing.append("protected_items_not_respected")
        if missing:
            failed_obligations[block_id] = _deduplicate(missing)

    if review is not None:
        if review.prefix_only_rewrite:
            warnings.append("Document-level prefix-only rewrite risk was detected.")
            prefix_only_block_ids.extend(sorted(candidates_by_id))
        if review.rewrite_coverage < plan.minor_risk_rewrite_coverage:
            warnings.append("Rewrite coverage stayed below the minimum public write threshold.")
        if review.discourse_change_score < DEFAULT_CONFIG.light_edit_min_discourse_score:
            warnings.append("Discourse change score is too low for public execution.")
        if not review.rewrite_quota_met:
            warnings.extend(review.rewrite_quota_reason_codes)
        if (
            review.write_gate_ready
            and review.core_content_integrity
            and review.format_integrity
            and review.rewrite_quota_met
            and review.chapter_rewrite_quota_check
            and review.chapter_policy_consistency_check
            and review.paragraph_topic_sentence_preserved
            and review.paragraph_opening_style_valid
            and review.paragraph_skeleton_consistent
            and review.sentence_completeness_preserved
            and review.paragraph_readability_preserved
            and review.stylistic_uniformity_controlled
            and review.support_sentence_texture_varied
            and review.paragraph_voice_variation_present
            and review.academic_cliche_density_controlled
              and review.semantic_role_integrity_preserved
              and review.enumeration_integrity_preserved
              and review.scaffolding_phrase_density_controlled
              and review.over_abstracted_subject_risk_controlled
              and review.assertion_strength_preserved
              and review.appendix_like_support_controlled
              and review.authorial_stance_present
              and review.evidence_fidelity_preserved
              and review.unsupported_expansion_controlled
              and review.thesis_tone_restrained
              and review.metaphor_or_storytelling_controlled
              and review.authorial_claim_risk_controlled
              and review.bureaucratic_opening_controlled
              and review.explicit_subject_chain_controlled
              and review.overstructured_syntax_controlled
              and review.main_clause_position_reasonable
              and review.slogan_like_goal_phrase_controlled
          ):
              failed_obligations = {}

    failed_block_ids = sorted(failed_obligations)
    return ExecutionValidationResult(
        ok=not failed_block_ids,
        failed_block_ids=failed_block_ids,
        failed_obligations=failed_obligations,
        warnings=_deduplicate(warnings),
        prefix_only_block_ids=sorted(set(prefix_only_block_ids)),
        surface_level_block_ids=sorted(set(surface_level_block_ids)),
    )


def protocol_payload(
    schema: SkillInputSchema,
    plan: SkillExecutionPlan,
    output: SkillOutputSchema | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_schema": schema.to_dict(),
        "execution_plan": plan.to_dict(),
        "available_presets": {
            name: {**asdict(preset), "mode": preset.mode.value} for name, preset in SKILL_PRESETS.items()
        },
    }
    if output is not None:
        payload["output_schema"] = output.to_dict()
    return payload


def _required_actions_for_agent(
    structural_actions: list[str],
    discourse_actions: list[str],
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for action in [*structural_actions, *discourse_actions]:
        if action and action not in seen:
            seen.add(action)
            ordered.append(action)
    return ordered


def _required_sentence_actions_for_block(block: Any) -> list[str]:
    actions: list[str] = []
    if getattr(block, "required_minimum_sentence_level_changes", 0) > 0:
        actions.append("rebuild_sentence_rhythm")
    for action in getattr(block, "required_discourse_actions", []):
        if action in {
            "reduce_function_word_overuse",
            "weaken_template_connectors",
            "rewrite_dense_nominal_phrases",
            "break_parallelism",
            "compress_subject_chain",
        }:
            actions.append(action)
    if getattr(block, "edit_policy", "") == "light_edit" and not actions:
        actions.append("sentence_level_recast")
    return _deduplicate(actions)


def _required_cluster_actions_for_block(block: Any) -> list[str]:
    actions: list[str] = []
    if getattr(block, "required_minimum_cluster_changes", 0) > 0:
        actions.append("sentence_cluster_rewrite")
    for action in getattr(block, "required_discourse_actions", []):
        if action in {"sentence_cluster_rewrite", "conclusion_absorb", "transition_absorption", "proposition_reorder"}:
            actions.append(action)
    return _deduplicate(actions)


def _clarity_constraints_for_block(block: Any) -> list[str]:
    constraints = [
        "preserve_terms_citations_numbers_paths_and_markdown",
        "do_not_add_facts_or_examples",
        "do_not_colloquialize",
    ]
    if "preserve_explicit_subject_if_clarity_needed" in getattr(block, "recommended_actions", []):
        constraints.append("preserve_explicit_subject_if_clarity_needed")
    if "keep_original_if_technical_density_is_high" in getattr(block, "recommended_actions", []):
        constraints.append("keep_original_if_technical_density_is_high")
    if not getattr(block, "opening_reorder_allowed", True):
        constraints.append("preserve_topic_sentence_at_paragraph_opening")
        constraints.append("do_not_start_with_dangling_transition")
    constraints.append("preserve_sentence_completeness")
    constraints.append("do_not_leave_fragment_like_support_or_conclusion_sentences")
    if getattr(block, "high_sensitivity_prose", False):
        constraints.append("high_sensitivity_prose_readability_first")
    if getattr(block, "edit_policy", "") in {"do_not_touch", "high_risk"}:
        constraints.append("do_not_apply_developmental_rewrite")
    return _deduplicate(constraints)


def _failed_obligations_from_candidates(rewrite_report: RewriteExecutionReport) -> dict[int, list[str]]:
    failures: dict[int, list[str]] = {}
    for candidate in rewrite_report.block_candidates:
        if candidate.missing_required_actions:
            failures[candidate.block_id] = list(candidate.missing_required_actions)
    return failures


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
