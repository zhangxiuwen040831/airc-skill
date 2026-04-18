from __future__ import annotations

from difflib import unified_diff
from pathlib import Path

from airc_skill.config import DEFAULT_CONFIG, RewriteMode
from airc_skill.guidance import guide_document_text
from airc_skill.models import RewriteExecutionReport, WriteGateDecision
from airc_skill.pipeline import _score_candidate, agent_rewrite_from_guidance, decide_write_gate, rewrite_file
from airc_skill.reviewer import review_rewrite
from airc_skill.rewriter import RewriteStats, Rewriter


def _load_user_body_excerpt() -> str:
    root = Path(__file__).resolve().parents[1] / "test.md"
    text = root.read_text(encoding="utf-8")
    start = text.index("## 1.1 研究背景")
    end = text.index("## 1.4 创新点和研究内容")
    return text[start:end]


def test_developmental_rewrite_blocks_require_cluster_change() -> None:
    text = (
        "近年来，AIGC图像生成技术持续进入内容平台。"
        "与此同时，相关风险也逐渐从实验环境扩展到真实业务场景。"
        "因此，平台在部署检测工具时，不仅需要关注分类结果，还需要评估误报对治理流程造成的影响。"
    )

    guidance = guide_document_text(text, metadata={"suffix": ".txt"})
    block = guidance.rewritable_blocks[0]

    assert block.rewrite_depth == "developmental_rewrite"
    assert block.required_minimum_sentence_level_changes >= 2
    assert block.required_minimum_cluster_changes >= 1
    assert "sentence_cluster_rewrite" in block.required_discourse_actions


def test_light_edit_blocks_require_sentence_level_change() -> None:
    text = (
        "本研究的主题为课堂反馈机制的优化路径。"
        "本研究不仅包含多轮写作任务设计，还完成了教师复核流程梳理。"
    )

    guidance = guide_document_text(text, metadata={"suffix": ".txt"})
    block = guidance.light_edit_blocks[0]

    assert block.rewrite_depth == "light_edit"
    assert block.required_minimum_sentence_level_changes == 1
    assert block.required_minimum_cluster_changes == 0


def test_rewritable_block_not_passed_by_prefix_only_rewrite() -> None:
    original = (
        "近年来，数字平台正在持续进入课堂评价场景。"
        "与此同时，教师还需要根据系统反馈调整任务安排。"
        "因此，平台需要重新评估人工校对流程。"
    )
    revised = (
        "近年来，数字平台正在持续进入课堂评价场景。"
        "同时，教师还需要根据系统反馈调整任务安排。"
        "基于此，平台需要重新评估人工校对流程。"
    )
    guidance = guide_document_text(original, metadata={"suffix": ".txt"})

    stats = RewriteStats(
        mode=RewriteMode.BALANCED,
        changed=True,
        applied_rules=["opening:transition", "opening:implication"],
        sentence_count_before=3,
        sentence_count_after=3,
        sentence_level_change=False,
        changed_characters=14,
        original_sentences=original.split("。")[:3],
        rewritten_sentences=revised.split("。")[:3],
        paragraph_char_count=88,
        sentence_labels=["background", "support", "conclusion"],
        subject_heads=["数字平台", "教师", "平台"],
        detected_patterns=["background_transition", "risk_conclusion"],
        structural_actions=[],
        structural_action_count=0,
        high_value_structural_actions=[],
        discourse_actions_used=[],
        sentence_level_changes=0,
        cluster_changes=0,
        discourse_change_score=0,
        prefix_only_rewrite=True,
        repeated_subject_risk=False,
        selected_variants=["同时", "基于此"],
        candidate_notes=[],
        paragraph_index=1,
        block_id=1,
        rewrite_depth="developmental_rewrite",
    )

    review = review_rewrite(
        original,
        revised,
        guidance=guidance,
        mode=RewriteMode.BALANCED,
        rewrite_stats=[stats],
        suffix=".txt",
    )

    assert review.decision == "reject"
    assert review.prefix_only_rewrite is True


def test_discourse_change_score_increases_for_body_paragraphs() -> None:
    text = (
        "近年来，AIGC图像生成技术持续进入内容平台。"
        "与此同时，相关风险也逐渐从实验环境扩展到真实业务场景。"
        "因此，平台在部署检测工具时，不仅需要关注分类结果，还需要评估误报对治理流程造成的影响。"
    )

    rewritten, stats = Rewriter().rewrite(
        text,
        mode=RewriteMode.BALANCED,
        rewrite_depth="developmental_rewrite",
    )

    assert rewritten != text
    assert stats.discourse_change_score >= DEFAULT_CONFIG.developmental_min_discourse_score
    assert stats.cluster_changes >= DEFAULT_CONFIG.developmental_min_cluster_changes


def test_rewrite_coverage_not_trivial() -> None:
    text = (
        "近年来，数字平台正在持续进入课堂评价场景。"
        "与此同时，教师还需要根据系统反馈调整任务安排。"
        "因此，平台需要重新评估人工校对流程。"
    )

    _, stats = Rewriter().rewrite(
        text,
        mode=RewriteMode.BALANCED,
        rewrite_depth="developmental_rewrite",
        rewrite_intensity="medium",
    )

    assert stats.rewrite_coverage >= DEFAULT_CONFIG.rewrite_coverage_pass_threshold


def test_sentence_cluster_rewriter_changes_narrative_flow() -> None:
    text = (
        "近年来，数字平台正在持续进入课堂评价场景。"
        "与此同时，教师还需要根据系统反馈调整任务安排。"
        "因此，平台需要重新评估人工校对流程。"
    )

    rewritten, stats = Rewriter().rewrite(
        text,
        mode=RewriteMode.BALANCED,
        rewrite_depth="developmental_rewrite",
    )

    assert rewritten != text
    assert "sentence_cluster_rewrite" in stats.discourse_actions_used
    assert len(stats.rewritten_sentences) <= len(stats.original_sentences)


def test_cluster_rewrite_changes_sentence_order() -> None:
    text = (
        "背景信息显示，平台内容治理压力正在上升。"
        "风险分析表明，虚假图像会影响证据判断。"
        "本文进一步讨论检测流程与人工复核之间的衔接方式。"
    )

    rewritten, stats = Rewriter().rewrite(
        text,
        mode=RewriteMode.BALANCED,
        rewrite_depth="developmental_rewrite",
        rewrite_intensity="high",
    )

    assert "paragraph_reorder" in stats.structural_actions
    assert rewritten.startswith("风险分析表明")


def test_discourse_transform_applied() -> None:
    text = (
        "本研究的主题为课堂反馈机制的优化路径。"
        "本研究不仅包含多轮写作任务设计，还完成了教师复核流程梳理。"
        "因此，本研究进一步讨论系统输出与人工评价之间的协同方式。"
    )

    rewritten, stats = Rewriter().rewrite(
        text,
        mode=RewriteMode.BALANCED,
        rewrite_depth="developmental_rewrite",
        rewrite_intensity="high",
    )

    assert rewritten != text
    assert {"meta_compression", "conclusion_absorb"} & set(stats.discourse_actions_used)


def test_user_test_md_body_rewrite_changes_more_than_trivial_lines(tmp_path: Path) -> None:
    source = tmp_path / "body_excerpt.md"
    source.write_text(_load_user_body_excerpt(), encoding="utf-8")

    original = source.read_text(encoding="utf-8")
    guidance = guide_document_text(original, metadata={"suffix": ".md", "source_path": source})
    rewrite_report = agent_rewrite_from_guidance(
        original,
        guidance=guidance,
        mode=RewriteMode.BALANCED,
        suffix=".md",
        convenience_mode=False,
    )

    diff_lines = [
        line
        for line in unified_diff(original.splitlines(), rewrite_report.rewritten_text.splitlines(), lineterm="")
        if (line.startswith("+") or line.startswith("-")) and not line.startswith("+++") and not line.startswith("---")
    ]

    assert len(rewrite_report.changed_block_ids) >= 4
    assert len(diff_lines) >= 10


def test_user_test_md_not_minimal_change(tmp_path: Path) -> None:
    source = tmp_path / "body_excerpt.md"
    source.write_text(_load_user_body_excerpt(), encoding="utf-8")

    original = source.read_text(encoding="utf-8")
    guidance = guide_document_text(original, metadata={"suffix": ".md", "source_path": source})
    rewrite_report = agent_rewrite_from_guidance(
        original,
        guidance=guidance,
        mode=RewriteMode.BALANCED,
        suffix=".md",
        convenience_mode=False,
    )

    assert len(rewrite_report.changed_block_ids) >= 4
    assert sum(stats.discourse_change_score for stats in rewrite_report.rewrite_stats) >= 20


def test_candidate_ranking_prefers_high_discourse_change() -> None:
    original = (
        "近年来，数字平台正在持续进入课堂评价场景。"
        "与此同时，教师还需要根据系统反馈调整任务安排。"
        "因此，平台需要重新评估人工校对流程。"
    )
    low_revised = "近年来，数字平台正在持续进入课堂评价场景，教师还需要根据系统反馈调整任务安排。因此，平台需要重新评估人工校对流程。"
    high_revised = "教师需要根据系统反馈调整任务安排，这一变化发生在数字平台持续进入课堂评价场景的过程中。平台因此需要重新评估人工校对流程。"
    guidance = guide_document_text(original, metadata={"suffix": ".txt"})

    low_stats = RewriteStats(
        mode=RewriteMode.BALANCED,
        changed=True,
        applied_rules=["structure:merge-short-followup"],
        sentence_count_before=3,
        sentence_count_after=2,
        sentence_level_change=True,
        changed_characters=30,
        original_sentences=[
            "近年来，数字平台正在持续进入课堂评价场景。",
            "与此同时，教师还需要根据系统反馈调整任务安排。",
            "因此，平台需要重新评估人工校对流程。",
        ],
        rewritten_sentences=[
            "近年来，数字平台正在持续进入课堂评价场景，教师还需要根据系统反馈调整任务安排。",
            "因此，平台需要重新评估人工校对流程。",
        ],
        paragraph_char_count=88,
        sentence_labels=["background", "support", "conclusion"],
        subject_heads=["数字平台", "平台"],
        detected_patterns=["background_transition"],
        structural_actions=["sentence_merge"],
        structural_action_count=1,
        high_value_structural_actions=[],
        discourse_actions_used=["sentence_cluster_rewrite"],
        sentence_level_changes=1,
        cluster_changes=1,
        discourse_change_score=4,
        rewrite_coverage=0.34,
        prefix_only_rewrite=False,
        repeated_subject_risk=False,
        selected_variants=["省略过渡"],
        candidate_notes=[],
        paragraph_index=1,
        block_id=1,
        rewrite_depth="developmental_rewrite",
        rewrite_intensity="medium",
    )
    high_stats = RewriteStats(
        mode=RewriteMode.BALANCED,
        changed=True,
        applied_rules=["paragraph:high-intensity-promote-second", "fusion:implication-followup"],
        sentence_count_before=3,
        sentence_count_after=2,
        sentence_level_change=True,
        changed_characters=80,
        original_sentences=low_stats.original_sentences,
        rewritten_sentences=[
            "教师需要根据系统反馈调整任务安排，这一变化发生在数字平台持续进入课堂评价场景的过程中。",
            "平台因此需要重新评估人工校对流程。",
        ],
        paragraph_char_count=88,
        sentence_labels=["background", "support", "conclusion"],
        subject_heads=["教师", "平台"],
        detected_patterns=["background_transition", "risk_conclusion"],
        structural_actions=["paragraph_reorder", "conclusion_absorb"],
        structural_action_count=2,
        high_value_structural_actions=["paragraph_reorder", "conclusion_absorb"],
        discourse_actions_used=["proposition_reorder", "sentence_cluster_rewrite", "conclusion_absorb"],
        sentence_level_changes=3,
        cluster_changes=2,
        discourse_change_score=11,
        rewrite_coverage=1.0,
        prefix_only_rewrite=False,
        repeated_subject_risk=False,
        selected_variants=["proposition_reorder", "省略因果句首"],
        candidate_notes=[],
        paragraph_index=1,
        block_id=1,
        rewrite_depth="developmental_rewrite",
        rewrite_intensity="high",
    )
    low_review = review_rewrite(original, low_revised, guidance=guidance, mode=RewriteMode.BALANCED, rewrite_stats=[low_stats])
    high_review = review_rewrite(original, high_revised, guidance=guidance, mode=RewriteMode.BALANCED, rewrite_stats=[high_stats])
    gate = WriteGateDecision(True, "pass", [], [], "synthetic gate")

    assert _score_candidate(high_review, gate) > _score_candidate(low_review, gate)


def test_pass_with_minor_risk_can_write_when_body_rewrite_is_substantive() -> None:
    original = (
        "本研究的主题为课堂反馈机制的优化路径。"
        "本研究不仅包含多轮写作任务设计，还完成了教师复核流程梳理。"
        "因此，本研究进一步讨论系统输出与人工评价之间的协同方式。"
    )
    revised = (
        "围绕课堂反馈机制的优化路径这一问题，本研究不仅包含多轮写作任务设计，还完成了教师复核流程梳理。"
        "本研究进一步讨论系统输出与人工评价之间的协同方式。"
    )
    guidance = guide_document_text(original, metadata={"suffix": ".txt"})
    stats = RewriteStats(
        mode=RewriteMode.BALANCED,
        changed=True,
        applied_rules=["subject:meta-compression", "fusion:implication-followup"],
        sentence_count_before=3,
        sentence_count_after=2,
        sentence_level_change=True,
        changed_characters=52,
        original_sentences=[
            "本研究的主题为课堂反馈机制的优化路径。",
            "本研究不仅包含多轮写作任务设计，还完成了教师复核流程梳理。",
            "因此，本研究进一步讨论系统输出与人工评价之间的协同方式。",
        ],
        rewritten_sentences=[
            "围绕课堂反馈机制的优化路径这一问题，本研究不仅包含多轮写作任务设计，还完成了教师复核流程梳理。",
            "本研究进一步讨论系统输出与人工评价之间的协同方式。",
        ],
        paragraph_char_count=104,
        sentence_labels=["objective", "support", "conclusion"],
        subject_heads=["本研究", "本研究"],
        detected_patterns=["study_description", "subject_chain"],
        structural_actions=["meta_compression", "subject_chain_compression", "conclusion_absorb"],
        structural_action_count=3,
        high_value_structural_actions=["subject_chain_compression", "conclusion_absorb"],
        discourse_actions_used=["meta_compression", "sentence_cluster_rewrite", "conclusion_absorb"],
        sentence_level_changes=3,
        cluster_changes=2,
        discourse_change_score=9,
        prefix_only_rewrite=False,
        repeated_subject_risk=False,
        selected_variants=["围绕……这一问题", "这也使得"],
        candidate_notes=[],
        paragraph_index=1,
        block_id=1,
        rewrite_depth="developmental_rewrite",
    )

    review = review_rewrite(
        original,
        revised,
        guidance=guidance,
        mode=RewriteMode.BALANCED,
        rewrite_stats=[stats],
        suffix=".txt",
    )
    rewrite_report = RewriteExecutionReport(
        rewritten_text=revised,
        block_candidates=[],
        rewrite_stats=[stats],
        mode_requested=RewriteMode.BALANCED,
        mode_used=RewriteMode.BALANCED,
        effective_change=True,
        changed_block_ids=[1],
        candidate_count=1,
        selected_candidate_reason="Synthetic substantive rewrite with a minor repeated-subject residue.",
        convenience_mode=False,
        block_failures=[],
        reviewed=True,
    )
    write_gate = decide_write_gate(review, rewrite_report, {})

    assert review.decision == "pass_with_minor_risk"
    assert write_gate.write_allowed is True


def test_hard_preserve_layer_still_rejects_core_or_format_breakage() -> None:
    original = "## Design and Implementation of an AI-Generated Image Detection Tool\nStable Diffusion 在 NTIRE 数据集上完成评估。[1]"
    revised = "## DesignandImplementationofanAI-GeneratedImageDetectionTool\n生成模型在 NTIRE 数据集上完成评估。[1]"
    guidance = guide_document_text(original, metadata={"suffix": ".md"})

    review = review_rewrite(original, revised, guidance=guidance, mode=RewriteMode.BALANCED, suffix=".md")

    assert review.decision == "reject"
    assert review.core_content_integrity is False or review.format_integrity is False
