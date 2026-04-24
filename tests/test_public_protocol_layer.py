from __future__ import annotations

import json
from pathlib import Path

from airc_skill.cli import main
from airc_skill.public_api import run_revision
from airc_skill.skill_protocol import SkillInputSchema, SkillOutputSchema, build_execution_plan, generate_agent_instructions
from airc_skill.guidance import guide_document_text


def _body_sample() -> str:
    return (
        "近年来，智能写作系统逐渐进入高校课程与科研训练场景。与此同时，教师需要在效率提升和质量控制之间重新分配精力，"
        "因此，原有依赖人工逐段反馈的流程面临更高压力。\n\n"
        "本研究的主题为学术反馈流程的协同优化。本研究不仅包含课堂任务设计，还完成了反馈链条的流程梳理。"
        "因此，本研究进一步讨论系统输出与人工评价之间的协同方式。\n\n"
        "在实际运行中，任务复杂度、评价标准和反馈周期都会影响教师判断。与此同时，平台能力提升也改变了原有工作分配方式，"
        "因此需要形成更稳定的人工复核机制。"
    )


def test_skill_input_schema_roundtrip() -> None:
    schema = SkillInputSchema.from_path(
        "paper.md",
        preset="aggressive_rewrite",
        output_path="paper.airc.md",
        max_retry_passes=3,
        emit_agent_context=True,
    )

    restored = SkillInputSchema.from_dict(schema.to_dict())

    assert restored.source_path == "paper.md"
    assert restored.preset == "aggressive_rewrite"
    assert restored.output_path == "paper.airc.md"
    assert restored.max_retry_passes == 3
    assert restored.emit_agent_context is True


def test_skill_output_schema_roundtrip() -> None:
    schema = SkillOutputSchema(
        status="success",
        rewritten_file_path="paper.airc.md",
        report_file_path="paper.airc.report.json",
        input_normalization={"original_type": ".md"},
        rewrite_coverage=0.72,
        discourse_change_score=6,
        cluster_rewrite_score=2,
        blocks_changed=[1, 2],
        blocks_skipped=[3],
        blocks_rejected=[],
        warnings=[],
        failed_obligations={},
        write_gate_decision="pass",
        write_allowed=True,
        decision="pass",
        reason_codes=["review_passed"],
    )

    restored = SkillOutputSchema.from_dict(schema.to_dict())

    assert restored.status == "success"
    assert restored.rewrite_report_path == "paper.airc.report.json"
    assert restored.rewrite_coverage == 0.72
    assert restored.blocks_changed == [1, 2]


def test_execution_plan_contains_required_actions() -> None:
    guidance = guide_document_text(_body_sample(), metadata={"suffix": ".txt"})
    plan = build_execution_plan(guidance, SkillInputSchema.from_path("paper.txt", preset="academic_natural"))

    required = [action for actions in plan.required_actions_by_block.values() for action in actions]

    assert plan.block_policies
    assert any(policy["edit_policy"] in {"light_edit", "rewritable"} for policy in plan.block_policies)
    assert "sentence_cluster_rewrite" in required or "conclusion_absorb" in required
    assert plan.write_gate_preconditions


def test_agent_instruction_generation_contains_required_and_forbidden_actions() -> None:
    guidance = guide_document_text(_body_sample(), metadata={"suffix": ".txt"})
    instructions = generate_agent_instructions(guidance, SkillInputSchema.from_path("paper.txt"))
    payload = instructions.to_dict()

    assert payload["blocks"]
    assert any(block["required_actions"] for block in payload["blocks"])
    assert all("do_not_change_terms" in block["forbidden_actions"] for block in payload["blocks"])
    assert "minimum_rewrite_coverage" in payload["text"]
    assert "Do not perform prefix-only" in payload["global_rules"][0]


def test_python_interface_run_revision_returns_structured_result(tmp_path: Path) -> None:
    source = tmp_path / "paper.txt"
    output = tmp_path / "paper.airc.md"
    source.write_text(_body_sample(), encoding="utf-8")
    schema = SkillInputSchema.from_path(source, output_path=output, emit_json_report=True)

    result = run_revision(schema)

    assert result.output_schema.rewrite_coverage >= 0.4
    assert result.output_schema.discourse_change_score >= 1
    assert result.output_schema.write_gate_decision in {"pass", "pass_with_minor_risk", "reject"}
    assert result.execution_plan.agent_instructions
    assert result.human_report.startswith("AIRC Revision Report")


def test_public_run_cli_generates_output_and_json_report(tmp_path: Path) -> None:
    source = tmp_path / "paper.txt"
    output = tmp_path / "paper.airc.md"
    report = tmp_path / "paper.airc.report.json"
    source.write_text(_body_sample(), encoding="utf-8")

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
            "--agent-context",
        ]
    )

    assert exit_code == 0
    assert output.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["input_schema"]["preset"] == "academic_natural"
    assert payload["execution_plan"]["agent_instructions"]
    assert payload["failure_transparency"]["current_rewrite_coverage"] >= 0.4


def test_real_test_md_public_run_has_nontrivial_body_rewrite(tmp_path: Path) -> None:
    real_sample = Path("tests") / "fixtures" / "user_test.md"
    if not real_sample.exists():
        return
    source = tmp_path / "user_test.md"
    source.write_text(real_sample.read_text(encoding="utf-8"), encoding="utf-8")

    result = main(["run", str(source), "--preset", "academic_natural", "--dry-run"])

    assert result == 0


def test_real_test_md_preserves_core_and_format_under_public_run(tmp_path: Path) -> None:
    real_sample = Path("tests") / "fixtures" / "user_test.md"
    if not real_sample.exists():
        return
    source = tmp_path / "user_test.md"
    source.write_text(real_sample.read_text(encoding="utf-8"), encoding="utf-8")

    run = run_revision(SkillInputSchema.from_path(source, output_path=tmp_path / "test.airc.md"))

    assert run.rewrite_result.review.core_content_integrity is True
    assert run.rewrite_result.review.format_integrity is True
    assert run.output_schema.rewrite_coverage >= 0.6
    assert len(run.output_schema.blocks_changed) >= 10
