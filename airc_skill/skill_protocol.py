from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, SKILL_PRESETS, RewriteMode, get_skill_preset
from .models import GuidanceReport, RewriteExecutionReport, ReviewReport, WriteGateDecision


@dataclass(frozen=True)
class SkillInputSchema:
    text_path: str
    mode: str = "balanced"
    target_style: str = "academic_natural"
    rewrite_scope: str = "full"
    preservation_level: str = "strict"
    preset: str = "academic_natural"

    @classmethod
    def from_path(
        cls,
        text_path: str | Path,
        preset: str = "academic_natural",
        mode: str | None = None,
        target_style: str | None = None,
        rewrite_scope: str | None = None,
        preservation_level: str | None = None,
    ) -> "SkillInputSchema":
        preset_config = get_skill_preset(preset)
        return cls(
            text_path=str(text_path),
            mode=mode or preset_config.mode.value,
            target_style=target_style or preset_config.target_style,
            rewrite_scope=rewrite_scope or preset_config.rewrite_scope,
            preservation_level=preservation_level or preset_config.preservation_level,
            preset=preset_config.name,
        )

    def resolved_mode(self) -> RewriteMode:
        if self.mode == "custom":
            return get_skill_preset(self.preset).mode
        return RewriteMode.from_value(self.mode)


@dataclass(frozen=True)
class SkillExecutionPlan:
    block_policies: list[dict[str, Any]]
    rewrite_intensity_by_block: dict[int, str]
    required_actions_by_block: dict[int, list[str]]
    forbidden_actions_by_block: dict[int, list[str]]
    minimum_rewrite_coverage: float
    minor_risk_rewrite_coverage: float
    mode: str
    target_style: str
    rewrite_scope: str
    preservation_level: str
    preset: str
    agent_instruction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillOutputSchema:
    rewritten_file_path: str | None
    rewrite_report_path: str | None
    input_normalization: dict[str, Any]
    rewrite_coverage: float
    discourse_change_score: int
    cluster_rewrite_score: int
    blocks_changed: list[int]
    blocks_skipped: list[int]
    warnings: list[str]
    write_allowed: bool
    decision: str
    reason_codes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_execution_plan(
    guidance: GuidanceReport,
    schema: SkillInputSchema,
) -> SkillExecutionPlan:
    preset = get_skill_preset(schema.preset)
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
                "required_actions": required_actions,
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
        block_policies=block_policies,
        rewrite_intensity_by_block=rewrite_intensity_by_block,
        required_actions_by_block=required_actions_by_block,
        forbidden_actions_by_block=forbidden_actions_by_block,
        minimum_rewrite_coverage=preset.min_rewrite_coverage,
        minor_risk_rewrite_coverage=DEFAULT_CONFIG.rewrite_coverage_minor_threshold,
        mode=schema.resolved_mode().value,
        target_style=schema.target_style,
        rewrite_scope=schema.rewrite_scope,
        preservation_level=schema.preservation_level,
        preset=schema.preset,
        agent_instruction=generate_agent_instructions(guidance, schema),
    )


def generate_agent_instructions(
    guidance: GuidanceReport,
    schema: SkillInputSchema | None = None,
) -> str:
    schema = schema or SkillInputSchema(text_path=str(guidance.source_path or ""), preset="academic_natural")
    preset = get_skill_preset(schema.preset)
    lines = [
        "AIRC execution protocol",
        f"- preset: {preset.name}",
        f"- mode: {schema.resolved_mode().value}",
        f"- target_style: {schema.target_style}",
        f"- preservation_level: {schema.preservation_level}",
        f"- minimum_rewrite_coverage: {preset.min_rewrite_coverage:.2f}",
        f"- minor_risk_rewrite_coverage: {DEFAULT_CONFIG.rewrite_coverage_minor_threshold:.2f}",
        "- mandatory rule: do not perform prefix-only or word-only rewrites on rewritable body blocks.",
        "- mandatory rule: every rewritable block must execute its required discourse or cluster actions unless the block is protected.",
        "- mandatory rule: preserve headings, formulas, terms, citations, numbers, paths, captions, placeholders, and Markdown structure.",
        "",
        "Block execution plan:",
    ]

    for block in guidance.block_policies:
        required = _required_actions_for_agent(block.required_structural_actions, block.required_discourse_actions)
        if block.edit_policy in {"do_not_touch", "high_risk"}:
            action_line = "preserve verbatim; do not rewrite narration"
        elif block.rewrite_depth == "developmental_rewrite":
            action_line = "perform developmental rewrite with sentence-cluster restructuring"
        else:
            action_line = "perform light sentence-level editing only"
        lines.extend(
            [
                f"Block {block.block_id}:",
                f"- type: {block.edit_policy}",
                f"- block_kind: {block.block_type}",
                f"- intensity: {block.rewrite_intensity}",
                f"- action_policy: {action_line}",
                f"- required: {', '.join(required) if required else 'none'}",
                f"- forbid: {', '.join(block.forbidden_actions) if block.forbidden_actions else 'none'}",
                (
                    "- minimum_changes: "
                    f"sentence>={block.required_minimum_sentence_level_changes}, "
                    f"cluster>={block.required_minimum_cluster_changes}"
                ),
                f"- notes: {'; '.join(block.notes) if block.notes else 'none'}",
            ]
        )

    lines.extend(
        [
            "",
            "Final self-check before review:",
            "- rewrite coverage meets the preset threshold for ordinary body blocks.",
            "- at least one cluster-level change appears in developmental rewrite blocks.",
            "- no rewritable block was skipped without a protection reason.",
            "- all protected core and format items remain byte-stable in meaning and Markdown structure.",
        ]
    )
    return "\n".join(lines)


def build_output_schema(
    rewrite_report: RewriteExecutionReport,
    review: ReviewReport,
    write_gate: WriteGateDecision,
    rewritten_file_path: str | Path | None,
    rewrite_report_path: str | Path | None = None,
    input_normalization: dict[str, Any] | None = None,
) -> SkillOutputSchema:
    changed = list(rewrite_report.changed_block_ids)
    attempted = [candidate.block_id for candidate in rewrite_report.block_candidates]
    skipped = [block_id for block_id in attempted if block_id not in changed]
    return SkillOutputSchema(
        rewritten_file_path=str(rewritten_file_path) if rewritten_file_path else None,
        rewrite_report_path=str(rewrite_report_path) if rewrite_report_path else None,
        input_normalization=input_normalization or {},
        rewrite_coverage=review.rewrite_coverage,
        discourse_change_score=review.discourse_change_score,
        cluster_rewrite_score=review.cluster_rewrite_score,
        blocks_changed=changed,
        blocks_skipped=skipped,
        warnings=[*review.problems, *review.warnings, *write_gate.warnings],
        write_allowed=write_gate.write_allowed,
        decision=write_gate.decision,
        reason_codes=list(write_gate.reason_codes),
    )


def protocol_payload(
    schema: SkillInputSchema,
    plan: SkillExecutionPlan,
    output: SkillOutputSchema | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_schema": asdict(schema),
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
