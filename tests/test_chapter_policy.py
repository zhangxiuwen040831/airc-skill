from __future__ import annotations

import json
from pathlib import Path

import pytest

from airc_skill.config import RewriteMode
from airc_skill.guidance import guide_document_text
from airc_skill.pipeline import run_file
from airc_skill.chapter_policy import chapter_quota_by_editable_prose, quota_for_priority
from airc_skill.reviewer import review_rewrite
from airc_skill.rewriter import RewriteStats, split_sentences


ROOT = Path(__file__).resolve().parents[1]
REAL_TEST_MD = ROOT / "test.md"


def _body_blocks(guidance):
    return [block for block in guidance.block_policies if block.should_rewrite]


def _stats(
    *,
    block_id: int,
    original: str,
    revised: str,
    structural_actions: list[str] | None = None,
    discourse_actions: list[str] | None = None,
    cluster_changes: int = 0,
    discourse_score: int = 1,
    rewrite_depth: str = "developmental_rewrite",
    rewrite_intensity: str = "medium",
) -> RewriteStats:
    original_sentences = split_sentences(original) or [original]
    revised_sentences = split_sentences(revised) or [revised]
    sentence_changes = max(
        1 if original.strip() != revised.strip() else 0,
        min(len(original_sentences), len(revised_sentences)),
    )
    return RewriteStats(
        mode=RewriteMode.BALANCED,
        changed=original.strip() != revised.strip(),
        applied_rules=["sentence:developmental-recast"],
        sentence_count_before=len(original_sentences),
        sentence_count_after=len(revised_sentences),
        sentence_level_change=original.strip() != revised.strip(),
        changed_characters=abs(len(original) - len(revised)) + 1,
        original_sentences=original_sentences,
        rewritten_sentences=revised_sentences,
        paragraph_char_count=len(original),
        sentence_labels=["detail"] * len(original_sentences),
        subject_heads=[],
        detected_patterns=[],
        structural_actions=structural_actions or ["clause_reorder"],
        structural_action_count=len(structural_actions or ["clause_reorder"]),
        high_value_structural_actions=[
            action
            for action in (structural_actions or ["clause_reorder"])
            if action in {"sentence_cluster_rewrite", "sentence_cluster_merge", "narrative_path_rewrite"}
        ],
        discourse_actions_used=discourse_actions or ["narrative_path_rewrite"],
        sentence_level_changes=sentence_changes,
        cluster_changes=cluster_changes,
        discourse_change_score=discourse_score,
        rewrite_coverage=min(1.0, sentence_changes / max(1, len(original_sentences))),
        prefix_only_rewrite=False,
        repeated_subject_risk=False,
        selected_variants=[],
        candidate_notes=[],
        paragraph_index=block_id,
        block_id=block_id,
        rewrite_depth=rewrite_depth,
        rewrite_intensity=rewrite_intensity,
        revision_patterns=["reframe"],
        human_noise_marks=[],
    )


def test_background_sections_receive_high_rewrite_priority() -> None:
    text = (
        "## 1.1 研究背景\n\n"
        "近年来，生成式人工智能图像系统快速进入内容生产场景。"
        "真实图像与生成图像之间的边界因此变得更加模糊。"
        "检测任务需要重新解释模型判别依据与应用风险。"
    )

    guidance = guide_document_text(text, metadata={"suffix": ".md"})
    block = _body_blocks(guidance)[0]

    assert block.chapter_type == "background"
    assert block.chapter_rewrite_priority == "high"
    assert block.chapter_rewrite_intensity == "high"
    assert block.chapter_rewrite_quota["coverage_min"] >= 0.60
    assert "sentence_cluster_rewrite" in block.recommended_actions


def test_method_sections_are_more_conservative() -> None:
    text = (
        "## 3.1 问题形式化定义\n\n"
        "给定输入图像 x，模型输出概率 p(y=1|x)，并在阈值 0.5 下完成二分类判定。"
        "该定义需要保持符号、阈值与输出含义一致。"
    )

    guidance = guide_document_text(text, metadata={"suffix": ".md"})
    block = _body_blocks(guidance)[0]

    assert block.chapter_type == "problem_definition"
    assert block.chapter_rewrite_priority == "conservative"
    assert block.chapter_rewrite_intensity == "light"
    assert block.rewrite_depth == "light_edit"
    assert "sentence_cluster_rewrite" not in block.required_discourse_actions


def test_conclusion_sections_receive_high_rewrite_priority() -> None:
    text = (
        "## 6.1 总结\n\n"
        "本文完成了面向 AIGC 图像检测的多分支模型设计。"
        "整体结果表明，语义信息与频域信息的协同能够提高判别稳定性。"
        "该研究也说明了决策路径约束的重要性。"
    )

    guidance = guide_document_text(text, metadata={"suffix": ".md"})
    block = _body_blocks(guidance)[0]

    assert block.chapter_type == "conclusion"
    assert block.chapter_rewrite_priority == "high"
    assert block.chapter_rewrite_intensity == "high"
    assert "narrative_path_rewrite" in block.recommended_actions


def test_chapter_rewrite_quota_is_enforced() -> None:
    original = (
        "## 1.1 研究背景\n\n"
        "生成式图像模型正在改变内容生产方式。检测任务因此需要重新定义风险边界。\n\n"
        "真实图像与生成图像之间的差异不断缩小。传统检测线索在开放场景中容易失效。\n\n"
        "## 3.5 损失函数设计\n\n"
        "损失函数由 BCE 损失和对比损失共同组成。该部分只需要保持训练目标清晰。"
    )
    revised = original.replace("该部分只需要保持训练目标清晰。", "训练目标在该部分保持清晰即可。")
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

    assert review.chapter_rewrite_quota_check is False
    assert any(
        "high_priority_chapter_coverage_below_quota" in reason
        for reason in review.chapter_rewrite_quota_reason_codes
    )


def test_real_test_md_outputs_test_airc_md(tmp_path: Path) -> None:
    if not REAL_TEST_MD.exists():
        pytest.skip("real test.md is not available in this checkout")
    output = tmp_path / "test_airc.md"
    report = tmp_path / "test_airc.report.json"

    result = run_file(REAL_TEST_MD, output_path=output, report_path=report)

    assert result.output_written is True
    assert output.exists()
    assert report.exists()
    assert result.rewrite_result.write_gate.write_allowed is True


def test_real_test_md_reports_chapter_level_metrics(tmp_path: Path) -> None:
    if not REAL_TEST_MD.exists():
        pytest.skip("real test.md is not available in this checkout")
    output = tmp_path / "test_airc.md"
    report = tmp_path / "test_airc.report.json"

    result = run_file(REAL_TEST_MD, output_path=output, report_path=report)
    payload = json.loads(report.read_text(encoding="utf-8"))
    metrics = payload["review"]["chapter_rewrite_metrics"]

    assert result.output_schema.chapter_rewrite_metrics
    assert metrics
    required = {
        "chapter_type",
        "chapter_rewrite_priority",
        "chapter_rewrite_coverage",
        "chapter_changed_blocks",
        "chapter_discourse_change_score",
        "chapter_cluster_rewrite_score",
        "chapter_rewrite_quota_met",
    }
    assert required.issubset(metrics[0])


def test_chapter_policy_prevents_overwriting_technical_dense_sections() -> None:
    original = (
        "## 3.5 损失函数设计\n\n"
        "损失函数由 BCE、对比损失和原型约束共同组成。该部分用于约束分类边界。"
    )
    revised = (
        "## 3.5 损失函数设计\n\n"
        "分类边界由 BCE、对比损失和原型约束共同组织，损失函数在这一过程中承担约束作用。"
        "该部分进一步说明训练目标。"
    )
    guidance = guide_document_text(original, metadata={"suffix": ".md"})
    block = _body_blocks(guidance)[0]
    stats = [
        _stats(
            block_id=block.block_id,
            original=block.original_text,
            revised=revised.split("\n\n", 1)[1],
            structural_actions=["sentence_cluster_rewrite", "sentence_cluster_merge"],
            discourse_actions=["sentence_cluster_rewrite", "narrative_path_rewrite"],
            cluster_changes=1,
            discourse_score=8,
            rewrite_intensity="high",
        )
    ]

    review = review_rewrite(
        original,
        revised,
        guidance=guidance,
        mode=RewriteMode.BALANCED,
        rewrite_stats=stats,
        block_candidates=[],
        suffix=".md",
    )

    assert block.chapter_rewrite_priority == "conservative"
    assert review.chapter_policy_consistency_check is False
    assert any(
        "conservative_chapter_over_rewritten" in reason
        for reason in review.chapter_rewrite_quota_reason_codes
    )


def test_body_only_chapter_quota_for_mixed_medium_priority_chapter() -> None:
    quota = chapter_quota_by_editable_prose(
        quota=quota_for_priority("medium"),
        total_blocks=1,
        total_sentences=1,
        changed_blocks=0,
        changed_sentences=0,
    )

    assert quota.coverage_min == 0.0
    assert quota.changed_block_ratio_min == 0.0
    assert quota.discourse_min == 0
