from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from airc_skill.cli import main
from airc_skill.config import RewriteMode
from airc_skill.guidance import guide_document_text
from airc_skill.pipeline import agent_rewrite_from_guidance, decide_write_gate, rewrite_file, run_file
from airc_skill.reviewer import review_rewrite
from airc_skill.rewriter import RewriteStats, Rewriter
from airc_skill.skill_protocol import SkillInputSchema, build_execution_plan, generate_agent_instructions


def _protocol_sample() -> str:
    return (
        "近年来，数字平台正在持续进入课堂评价场景。与此同时，教师需要根据系统反馈调整任务安排，"
        "因此有必要重新评估人工校对流程。\n\n"
        "本研究的主题为课堂反馈机制的优化路径。本研究不仅包含多轮写作任务设计，还完成了反馈链条的流程梳理。"
        "因此，本研究进一步讨论系统输出与人工评价之间的协同方式。\n\n"
        "在实际教学场景中，反馈周期、评价标准和任务复杂度都会影响教师的判断压力。"
        "与此同时，平台能力的提升也改变了原有的工作分配方式，因此需要形成更稳定的人工复核机制。"
    )


def _empty_stats(text: str, mode: RewriteMode) -> RewriteStats:
    sentences = [part for part in text.replace("。", "。\n").splitlines() if part.strip()]
    return RewriteStats(
        mode=mode,
        changed=False,
        applied_rules=[],
        sentence_count_before=len(sentences),
        sentence_count_after=len(sentences),
        sentence_level_change=False,
        changed_characters=0,
        original_sentences=sentences,
        rewritten_sentences=sentences,
        paragraph_char_count=len(text),
        sentence_labels=[],
        subject_heads=[],
        detected_patterns=[],
        structural_actions=[],
        structural_action_count=0,
        high_value_structural_actions=[],
        discourse_actions_used=[],
        sentence_level_changes=0,
        cluster_changes=0,
        discourse_change_score=0,
        rewrite_coverage=0.0,
        prefix_only_rewrite=False,
        repeated_subject_risk=False,
    )


class RetryProbeRewriter:
    def __init__(self) -> None:
        self.inner = Rewriter()

    def reset_document_state(self) -> None:
        self.inner.reset_document_state()

    def rewrite(
        self,
        text: str,
        mode: RewriteMode,
        pass_index: int = 1,
        rewrite_depth: str | None = None,
        rewrite_intensity: str | None = None,
    ):
        if pass_index == 1:
            return text, _empty_stats(text, mode)
        return self.inner.rewrite(
            text,
            mode=mode,
            pass_index=pass_index,
            rewrite_depth=rewrite_depth,
            rewrite_intensity=rewrite_intensity,
        )


def test_agent_instruction_generation() -> None:
    guidance = guide_document_text(_protocol_sample(), metadata={"suffix": ".txt"})
    schema = SkillInputSchema.from_path("paper.txt", preset="academic_natural")

    instructions = generate_agent_instructions(guidance, schema)

    assert "AIRC execution protocol" in instructions
    assert "minimum_rewrite_coverage" in instructions
    assert "Block" in instructions
    assert "sentence_cluster_rewrite" in instructions
    assert "forbid" in instructions


def test_protocol_output_schema(tmp_path: Path) -> None:
    source = tmp_path / "paper.txt"
    source.write_text(_protocol_sample(), encoding="utf-8")

    result = run_file(source, preset="academic_natural", dry_run=True)

    assert result.execution_plan.minimum_rewrite_coverage >= 0.6
    assert result.output_schema.rewrite_coverage >= 0.0
    assert result.output_schema.discourse_change_score == result.rewrite_result.review.discourse_change_score
    assert result.output_schema.cluster_rewrite_score == result.rewrite_result.review.cluster_rewrite_score
    assert result.output_schema.rewrite_report_path == str(result.report_path)


def test_retry_mechanism_improves_coverage(tmp_path: Path) -> None:
    source = tmp_path / "retry.txt"
    source.write_text(_protocol_sample(), encoding="utf-8")

    result = rewrite_file(
        source,
        mode=RewriteMode.BALANCED,
        dry_run=True,
        rewriter=RetryProbeRewriter(),
        debug_rewrite=True,
    )

    assert any("retry-guidance-escalated" in line for line in result.debug_log)
    assert result.review.rewrite_coverage >= 0.4
    assert result.review.discourse_change_score >= 1


def test_public_interface_cli(tmp_path: Path) -> None:
    source = tmp_path / "paper.txt"
    output = tmp_path / "paper.airc.txt"
    report = tmp_path / "paper.airc.report.json"
    source.write_text(_protocol_sample(), encoding="utf-8")

    exit_code = main(
        [
            "run",
            str(source),
            "--preset",
            "academic_natural",
            "--output",
            str(output),
            "--report",
            str(report),
        ]
    )

    assert exit_code == 0
    assert output.exists()
    assert report.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["input_schema"]["preset"] == "academic_natural"
    assert payload["output_schema"]["rewrite_coverage"] >= 0.4
    assert payload["execution_plan"]["agent_instruction"]


def test_external_agent_can_follow_protocol() -> None:
    original = _protocol_sample()
    guidance = guide_document_text(original, metadata={"suffix": ".txt"})
    schema = SkillInputSchema.from_path("paper.txt", preset="academic_natural")
    plan = build_execution_plan(guidance, schema)

    rewrite_report = agent_rewrite_from_guidance(
        original,
        guidance=guidance,
        mode=RewriteMode.BALANCED,
        suffix=".txt",
        convenience_mode=False,
    )
    review = review_rewrite(
        original,
        rewrite_report.rewritten_text,
        guidance=guidance,
        mode=RewriteMode.BALANCED,
        rewrite_stats=rewrite_report.rewrite_stats,
        block_candidates=rewrite_report.block_candidates,
        suffix=".txt",
    )
    gate = decide_write_gate(review, replace(rewrite_report, reviewed=True), {})

    assert plan.rewrite_intensity_by_block
    assert "sentence-cluster" in plan.agent_instruction
    assert rewrite_report.changed_block_ids
    assert review.decision in {"pass", "pass_with_minor_risk"}
    assert gate.write_allowed is True
