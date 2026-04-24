from __future__ import annotations

import pytest

from airc_skill.config import RewriteMode
from airc_skill.guidance import guide_document_text
from airc_skill.models import WriteGateDecision
from airc_skill.pipeline import _score_candidate, decide_write_gate
from airc_skill.reviewer import review_revision, review_rewrite


def _body_paragraph(index: int) -> str:
    return (
        "人工智能写作系统正在改变高校论文训练中的反馈流程。"
        "教师需要在效率提升和质量控制之间重新分配注意力。"
        "学生也需要理解生成内容与人工判断之间的边界。"
        "这种变化要求评价机制保持稳定而透明。"
        f"本段讨论的是正文重写压力在论文场景中的持续累积。"
    )


def _make_long_body(paragraphs: int = 160) -> str:
    return "\n\n".join(_body_paragraph(index) for index in range(paragraphs))


def test_very_long_document_with_few_changed_blocks_is_rejected() -> None:
    original = _make_long_body()
    paragraphs = original.split("\n\n")
    revised_paragraphs = list(paragraphs)
    for index in range(5):
        revised_paragraphs[index] = revised_paragraphs[index].replace(
            "人工智能写作系统正在改变高校论文训练中的反馈流程",
            "高校论文训练中的反馈流程正在受到智能写作系统的持续影响",
        )
    revised = "\n\n".join(revised_paragraphs)
    guidance = guide_document_text(original, metadata={"suffix": ".txt"})

    review = review_rewrite(
        original,
        revised,
        guidance=guidance,
        mode=RewriteMode.BALANCED,
        rewrite_stats=[],
        block_candidates=[],
        suffix=".txt",
    )
    gate = decide_write_gate(
        review,
        rewrite_report=_reviewed_external_report(revised),
        policy={},
    )

    assert review.document_scale == "very_long"
    assert review.body_blocks_total >= 50
    assert review.body_changed_blocks < review.body_blocks_total * 0.5
    assert review.body_rewrite_coverage < 0.60
    assert review.rewrite_quota_met is False
    assert gate.write_allowed is False
    assert "body_rewrite_coverage_below_quota" in gate.reason_codes


def test_image_markdown_lines_do_not_trigger_template_family_repetition() -> None:
    original = "\n".join(
        [
            "![本研究](figures/a.png)",
            "![本研究](figures/b.png)",
            "![本研究](figures/c.png)",
            "![本研究](figures/d.png)",
        ]
    )

    review = review_revision(original, original, RewriteMode.BALANCED, suffix=".md")

    assert review.template_risk is False
    assert review.template_issue != "templated_family_repetition"
    assert review.body_blocks_total == 0


def test_body_coverage_excludes_headings_images_captions_and_blank_lines() -> None:
    original = "\n\n".join(
        [
            "# 研究背景",
            "![流程图](figures/process.png)",
            "图 1 反馈流程示意图",
            "教师反馈流程需要同时处理效率、准确性和解释责任。",
            "学生写作训练也需要保留人工判断对论证质量的约束。",
            "平台能力提升会改变原有课堂任务的分配方式。",
            "因此，论文训练需要重新梳理人机协作的评价边界。",
        ]
    )
    revised = original.replace(
        "教师反馈流程需要同时处理效率、准确性和解释责任。",
        "教师反馈流程需要在效率、准确性与解释责任之间建立更稳定的协调关系。",
    ).replace(
        "平台能力提升会改变原有课堂任务的分配方式。",
        "随着平台能力提升，原有课堂任务的分配方式也会被重新组织。",
    )
    guidance = guide_document_text(original, metadata={"suffix": ".md"})

    review = review_rewrite(
        original,
        revised,
        guidance=guidance,
        mode=RewriteMode.BALANCED,
        rewrite_stats=[],
        block_candidates=[],
        suffix=".md",
    )

    assert guidance.body_blocks_total == 4
    assert review.body_blocks_total == 4
    assert review.body_changed_blocks == 2
    assert review.body_rewrite_coverage == pytest.approx(0.5)


def test_candidate_ranking_prefers_global_body_rewrite_over_local_polish() -> None:
    original = "\n\n".join(
        [
            "教师反馈流程需要处理效率问题。学生写作训练需要保留人工判断。",
            "平台能力提升会改变任务分配。评价边界也需要重新说明。",
            "系统输出可以提供初稿线索。人工复核仍然承担质量责任。",
            "课堂任务强调过程反馈。论文训练也强调论证质量。",
        ]
    )
    local = original.replace(
        "教师反馈流程需要处理效率问题。",
        "教师反馈流程需要更稳妥地处理效率问题。",
        1,
    )
    global_rewrite = "\n\n".join(
        [
            "教师反馈流程需要在效率要求与人工判断之间重新建立协调关系。学生写作训练也因此保留了质量约束。",
            "随着平台能力提升，任务分配被重新组织，评价边界也需要被进一步说明。",
            "系统输出可以提供初稿线索，但质量责任仍然需要由人工复核承担。",
            "课堂任务强调过程反馈，论文训练则继续围绕论证质量展开。",
        ]
    )

    local_review = review_rewrite(original, local, mode=RewriteMode.BALANCED, suffix=".txt")
    global_review = review_rewrite(original, global_rewrite, mode=RewriteMode.BALANCED, suffix=".txt")
    gate = WriteGateDecision(
        write_allowed=True,
        decision="pass",
        reason_codes=["review_passed"],
        warnings=[],
        selected_candidate_reason="test",
    )

    assert global_review.body_rewrite_coverage > local_review.body_rewrite_coverage
    assert _score_candidate(global_review, gate) > _score_candidate(local_review, gate)


def _reviewed_external_report(revised: str):
    from airc_skill.models import RewriteExecutionReport

    return RewriteExecutionReport(
        rewritten_text=revised,
        block_candidates=[],
        rewrite_stats=[],
        mode_requested=RewriteMode.BALANCED,
        mode_used=RewriteMode.BALANCED,
        effective_change=True,
        changed_block_ids=[],
        candidate_count=1,
        selected_candidate_reason="Reviewed external candidate.",
        convenience_mode=False,
        block_failures=[],
        reviewed=True,
    )
