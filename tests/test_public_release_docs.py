from __future__ import annotations

import json
from pathlib import Path

from airc_skill import ACADEMIC_NATURAL_REVISION_DOCTRINE, doctrine_for_agent_context
import airc_skill
from airc_skill.cli import main
from airc_skill.guidance import guide_document_text
from airc_skill.pipeline import run_file
from airc_skill.skill_protocol import SkillInputSchema, build_execution_plan


ROOT = Path(__file__).resolve().parents[1]


def _narrative_sample() -> str:
    return (
        "近年来，智能写作系统逐渐进入高校课程与科研训练场景。与此同时，教师需要在效率提升和质量控制之间重新分配精力，"
        "因此，原有依赖人工逐段反馈的流程面临更高压力。\n\n"
        "随着课程规模扩大，传统反馈流程在时效性、覆盖范围和评价一致性方面都暴露出不足。与此同时，学生对反馈的针对性和连续性提出了更高要求，"
        "因此需要重新组织人机协同的反馈机制。"
    )


def test_public_readme_examples_run(tmp_path: Path) -> None:
    source = tmp_path / "sample.md"
    source.write_text((ROOT / "examples" / "sample.md").read_text(encoding="utf-8"), encoding="utf-8")

    assert main(["guide", str(source), "--as-agent-context"]) == 0
    assert main(["run", str(source), "--preset", "academic_natural", "--dry-run"]) == 0

    run = run_file(source, output_path=tmp_path / "sample.airc.md")
    if run.output_written:
        assert main(["review", str(source), str(run.output_path)]) == 0


def test_skill_doc_matches_current_workflow() -> None:
    skill_text = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert "What This Skill Does" in skill_text
    assert "Output Contract" in skill_text
    assert "required_sentence_actions" in skill_text
    assert "required_cluster_actions" in skill_text
    assert "do_not_touch" in skill_text
    assert "rewritable" in skill_text
    assert "write gate" in skill_text.lower()


def test_natural_revision_doctrine_exposed_to_agent_instructions() -> None:
    guidance = guide_document_text(_narrative_sample(), metadata={"suffix": ".txt"})
    plan = build_execution_plan(guidance, SkillInputSchema.from_path("paper.txt"))
    doctrine = doctrine_for_agent_context()

    assert ACADEMIC_NATURAL_REVISION_DOCTRINE.public_name == "AIRC"
    assert plan.revision_doctrine["public_name"] == doctrine["public_name"]
    assert "academic natural revision" in plan.revision_doctrine["positioning"]
    assert "academic_natural_studentlike" in plan.human_agent_instructions
    assert "reduce_function_word_overuse" in plan.human_agent_instructions
    assert any(block["required_cluster_actions"] for block in plan.agent_instructions)


def test_rewritable_body_blocks_receive_developmental_rewrite() -> None:
    guidance = guide_document_text(_narrative_sample(), metadata={"suffix": ".txt"})
    plan = build_execution_plan(guidance, SkillInputSchema.from_path("paper.txt", preset="academic_natural"))
    rewritable = [block for block in plan.block_policies if block["edit_policy"] == "rewritable"]

    assert rewritable
    assert all(block["rewrite_depth"] == "developmental_rewrite" for block in rewritable)
    assert all(block["required_sentence_actions"] for block in rewritable)
    assert all(block["required_cluster_actions"] for block in rewritable)
    assert all("preserve_terms_citations_numbers_paths_and_markdown" in block["clarity_constraints"] for block in rewritable)
    assert all("do_not_colloquialize" in block["clarity_constraints"] for block in rewritable)


def test_public_usage_doc_exists() -> None:
    public_usage = ROOT / "PUBLIC_USAGE.md"
    public_usage_zh = ROOT / "PUBLIC_USAGE.zh-CN.md"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert public_usage.exists()
    assert public_usage_zh.exists()
    assert "./README.zh-CN.md" in readme
    assert "./PUBLIC_USAGE.zh-CN.md" in readme
    text = public_usage.read_text(encoding="utf-8")
    zh_text = public_usage_zh.read_text(encoding="utf-8")
    assert "airc run paper.md --preset academic_natural" in text
    assert "Agent Usage" in text
    assert "Output Files" in text
    assert "airc run paper.md --target-style-file target-style.md" in text
    assert "airc run paper.md --preset academic_natural" in zh_text
    assert "Agent 使用" in zh_text
    assert "输出文件说明" in zh_text


def test_public_readme_has_cli_agent_python_usage() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "airc guide paper.md --as-agent-context" in readme
    assert "airc run paper.md --preset academic_natural" in readme
    assert "airc run paper.md --preset zh_academic_l2_mild" in readme
    assert "airc run paper.md --target-style-file target-style.md" in readme
    assert "Agent / Codex" in readme
    assert "from airc_skill import SkillInputSchema" in readme
    assert "target_style_file=\"target-style.md\"" in readme


def test_target_style_cli_example_smoke(tmp_path: Path) -> None:
    source = tmp_path / "sample.md"
    target = tmp_path / "target-style.md"
    source.write_text((ROOT / "examples" / "sample.md").read_text(encoding="utf-8"), encoding="utf-8")
    target.write_text((ROOT / "examples" / "_user_excerpt.airc.md").read_text(encoding="utf-8"), encoding="utf-8")

    assert main(["run", str(source), "--target-style-file", str(target), "--dry-run"]) == 0


def test_public_release_docs_consistent_with_skill_protocol() -> None:
    docs = "\n".join(
        [
            (ROOT / "README.md").read_text(encoding="utf-8"),
            (ROOT / "SKILL.md").read_text(encoding="utf-8"),
            (ROOT / "PUBLIC_USAGE.md").read_text(encoding="utf-8"),
        ]
    )

    for token in ("AIRC", "airc", "airc-academic-revision", "academic_natural", "zh_academic_l2_mild"):
        assert token in docs
    assert "--target-style-file" in docs
    assert "Target-style alignment" in docs or "target-style alignment" in docs
    assert "does not copy" in docs or "Do not copy" in docs


def test_import_and_basic_run_smoke(tmp_path: Path) -> None:
    source = tmp_path / "sample.md"
    source.write_text((ROOT / "examples" / "sample.md").read_text(encoding="utf-8"), encoding="utf-8")

    assert hasattr(airc_skill, "run_revision")
    result = run_file(source, dry_run=True)
    assert result.output_schema.status
    assert result.rewrite_result.review.core_content_integrity is True


def test_release_notes_or_public_usage_contains_boundaries() -> None:
    public_docs = "\n".join(
        [
            (ROOT / "README.md").read_text(encoding="utf-8"),
            (ROOT / "README.zh-CN.md").read_text(encoding="utf-8"),
            (ROOT / "SKILL.md").read_text(encoding="utf-8"),
            (ROOT / "PUBLIC_USAGE.md").read_text(encoding="utf-8"),
            (ROOT / "PUBLIC_USAGE.zh-CN.md").read_text(encoding="utf-8"),
            (ROOT / "references" / "agent-guided-workflow.md").read_text(encoding="utf-8"),
        ]
    )

    assert "Artificial Intelligence Rewrite Content" in public_docs
    assert "What AIRC Preserves" in public_docs
    assert "其他人如何使用 AIRC" in public_docs
    assert "developmental rewrite" in public_docs
    for forbidden in ("降低 AI 风格", "降低 AIGC", "规避检测", "骗过系统", "绕过探针"):
        assert forbidden not in public_docs


def test_real_test_md_public_run_produces_natural_revision_checklist(tmp_path: Path) -> None:
    real_sample = ROOT / "tests" / "fixtures" / "user_test.md"
    if not real_sample.exists():
        return
    source = tmp_path / "user_test.md"
    report_path = tmp_path / "test.airc.report.json"
    source.write_text(real_sample.read_text(encoding="utf-8"), encoding="utf-8")

    result = run_file(source, output_path=tmp_path / "test.airc.md", report_path=report_path)
    checklist = result.rewrite_result.review.natural_revision_checklist

    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "natural_revision_checklist" in payload["review"]
    assert "meta_discourse_compressed" in checklist
    assert "narrative_flow_rebuilt" in checklist
    assert "developmental_rewrite_applied_to_body_blocks" in checklist


def test_real_test_md_preserves_core_and_format_under_public_workflow(tmp_path: Path) -> None:
    real_sample = ROOT / "tests" / "fixtures" / "user_test.md"
    if not real_sample.exists():
        return
    source = tmp_path / "user_test.md"
    source.write_text(real_sample.read_text(encoding="utf-8"), encoding="utf-8")

    result = run_file(source, output_path=tmp_path / "test.airc.md")
    review = result.rewrite_result.review

    assert review.core_content_integrity is True
    assert review.format_integrity is True
    assert review.title_integrity is True
    assert review.citation_integrity is True
    assert review.numeric_integrity is True
    assert result.output_schema.rewrite_coverage >= 0.6
