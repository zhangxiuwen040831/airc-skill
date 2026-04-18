from dataclasses import replace
from pathlib import Path

from airc_skill.config import RewriteMode
from airc_skill.guidance import guide_document_text
from airc_skill.models import RewriteExecutionReport
from airc_skill.pipeline import agent_rewrite_from_guidance, decide_write_gate, guide_file, review_file, write_file
from airc_skill.reviewer import review_rewrite
from airc_skill.rewriter import Rewriter


def _sample_document() -> str:
    return """# 方法概述

## Design and Implementation of an AI-Generated Image Detection Tool

## Abstract

This study designs a practical AIGC detection workflow. Stable Diffusion is evaluated on the NTIRE dataset.

本研究的主题为课堂反馈机制的优化路径。本研究不仅包含多轮写作任务设计，还完成了反馈链条的流程梳理。因此，本研究进一步讨论系统输出与人工评价之间的协同方式。

本研究最终采用 checkpoints/best.pth 作为本地部署模型，对应 V10 Phase2 epoch_008，默认运行于 base_only 模式。

如图5-1所示：
![img](file:///C:/demo/system.png)
图5-1 系统分层架构图

近年来，数字平台正在持续进入课堂评价场景。与此同时，教师还需要根据系统反馈调整任务安排，因此有必要重新评估人工校对流程。
"""


def test_guide_returns_block_policies() -> None:
    guidance = guide_document_text(_sample_document(), metadata={"suffix": ".md"})

    assert guidance.block_policies
    assert guidance.rewrite_actions_by_block
    assert guidance.write_gate_preconditions
    assert guidance.document_risk in {"mixed", "high_risk", "light_edit", "rewritable", "do_not_touch"}


def test_agent_notes_present_for_risky_blocks() -> None:
    guidance = guide_document_text(_sample_document(), metadata={"suffix": ".md"})

    assert guidance.agent_notes
    assert any("preserve it verbatim" in note or "high risk" in note for note in guidance.agent_notes)


def test_review_detects_core_content_violation() -> None:
    original = "Stable Diffusion 在 NTIRE 数据集上完成评估。[1]"
    revised = "生成模型在 NTIRE 数据集上完成评估。[1]"
    guidance = guide_document_text(original, metadata={"suffix": ".txt"})

    review = review_rewrite(original, revised, guidance=guidance, mode=RewriteMode.BALANCED, suffix=".txt")

    assert review.decision == "reject"
    assert review.terminology_integrity is False


def test_review_detects_format_integrity_violation() -> None:
    original = "Stable Diffusion is evaluated on the NTIRE dataset."
    revised = "Stable Diffusion is evaluated on the NTIRE dataset.."
    guidance = guide_document_text(original, metadata={"suffix": ".txt"})

    review = review_rewrite(original, revised, guidance=guidance, mode=RewriteMode.BALANCED, suffix=".txt")

    assert review.decision == "reject"
    assert review.english_spacing_integrity is False


def test_review_detects_template_risk() -> None:
    original = "近年来，数字平台正在持续进入课堂评价场景，教师还需要根据系统反馈调整任务安排，因此有必要重新评估人工校对流程。"
    revised = "本研究说明这一问题的重要性。本研究进一步说明这一问题的重要性。因此，本研究继续说明这一问题的重要性。"
    guidance = guide_document_text(original, metadata={"suffix": ".txt"})

    review = review_rewrite(original, revised, guidance=guidance, mode=RewriteMode.BALANCED, suffix=".txt")

    assert review.template_risk is True
    assert review.decision == "reject"


def test_review_detects_repeated_subject_risk() -> None:
    original = "本研究围绕课堂反馈机制展开，讨论其优化路径。"
    revised = "本研究围绕课堂反馈机制展开。本研究进一步说明系统流程。因此，本研究继续强调人工复核。"
    guidance = guide_document_text(original, metadata={"suffix": ".txt"})

    review = review_rewrite(original, revised, guidance=guidance, mode=RewriteMode.BALANCED, suffix=".txt")

    assert review.repeated_subject_risk is True


def test_write_gate_blocks_unreviewed_rewrite() -> None:
    original = "近年来，平台能力不断提升，教师也需要重新评估人工校对流程。"
    candidate = "近年来，平台能力不断提升，教师也需要重新评估人工校对流程，这使得人工校对流程的重构更为必要。"
    guidance = guide_document_text(original, metadata={"suffix": ".txt"})
    review = review_rewrite(original, candidate, guidance=guidance, mode=RewriteMode.BALANCED, suffix=".txt")
    rewrite_report = RewriteExecutionReport(
        rewritten_text=candidate,
        block_candidates=[],
        rewrite_stats=[],
        mode_requested=RewriteMode.BALANCED,
        mode_used=RewriteMode.BALANCED,
        effective_change=True,
        changed_block_ids=[],
        candidate_count=1,
        selected_candidate_reason="Synthetic candidate for write-gate test.",
        convenience_mode=False,
        reviewed=False,
    )

    gate = decide_write_gate(review, rewrite_report, {})

    assert gate.write_allowed is False
    assert gate.decision == "reject"


def test_write_gate_allows_reviewed_pass_candidate() -> None:
    original = "近年来，数字平台正在持续进入课堂评价场景。与此同时，教师还需要根据系统反馈调整任务安排，因此有必要重新评估人工校对流程。"
    rewritten, stats = Rewriter().rewrite(original, mode=RewriteMode.BALANCED)
    guidance = guide_document_text(original, metadata={"suffix": ".txt"})
    review = review_rewrite(
        original,
        rewritten,
        guidance=guidance,
        mode=RewriteMode.BALANCED,
        rewrite_stats=[stats],
        suffix=".txt",
    )
    rewrite_report = RewriteExecutionReport(
        rewritten_text=rewritten,
        block_candidates=[],
        rewrite_stats=[stats],
        mode_requested=RewriteMode.BALANCED,
        mode_used=RewriteMode.BALANCED,
        effective_change=rewritten != original,
        changed_block_ids=[1] if rewritten != original else [],
        candidate_count=1,
        selected_candidate_reason="Single reviewed candidate for write-gate test.",
        convenience_mode=False,
        reviewed=True,
    )

    gate = decide_write_gate(review, rewrite_report, {})

    assert gate.write_allowed is True
    assert gate.decision == "pass"


def test_agent_first_workflow_guide_review_write(tmp_path: Path) -> None:
    source = tmp_path / "paper.md"
    source.write_text(_sample_document(), encoding="utf-8")

    guidance = guide_file(source)
    rewrite_report = agent_rewrite_from_guidance(
        source.read_text(encoding="utf-8"),
        guidance=guidance,
        mode=RewriteMode.BALANCED,
        suffix=".md",
        convenience_mode=False,
    )
    candidate = tmp_path / "candidate.md"
    candidate.write_text(rewrite_report.rewritten_text, encoding="utf-8")

    reviewed = review_file(source, candidate, mode=RewriteMode.BALANCED)
    written = write_file(source, candidate, mode=RewriteMode.BALANCED)

    assert guidance.block_policies
    assert reviewed.review.decision in {"pass", "pass_with_minor_risk", "reject"}
    assert written.output_written == written.write_gate.write_allowed


def test_rewrite_is_not_required_for_guidance() -> None:
    guidance = guide_document_text("## 标题\n\nStable Diffusion 在 NTIRE 数据集上完成评估。[1]", metadata={"suffix": ".md"})

    assert guidance.block_policies
    assert guidance.rewrite_candidate_blocks is not None


def test_do_not_touch_blocks_survive_agent_workflow() -> None:
    original = _sample_document()
    guidance = guide_document_text(original, metadata={"suffix": ".md"})
    rewrite_report = agent_rewrite_from_guidance(
        original,
        guidance=guidance,
        mode=RewriteMode.BALANCED,
        suffix=".md",
        convenience_mode=False,
    )

    for block in guidance.do_not_touch_blocks:
        if block.original_text:
            assert block.original_text in rewrite_report.rewritten_text


def test_real_test_md_agent_first_policy(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    original = (root / "test.md").read_text(encoding="utf-8")
    source = tmp_path / "user_test.md"
    source.write_text(original, encoding="utf-8")

    guidance = guide_file(source)

    assert guidance.block_policies
    assert any(block.block_type in {"heading", "english_block", "caption", "placeholder"} for block in guidance.do_not_touch_blocks)
    assert any("checkpoint" in block.preview or ".pth" in block.preview for block in guidance.high_risk_blocks + guidance.rewrite_candidate_blocks)
