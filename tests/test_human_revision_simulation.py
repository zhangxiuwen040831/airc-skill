from __future__ import annotations

from airc_skill.config import RewriteMode
from airc_skill.guidance import guide_document_text
from airc_skill.models import RewriteExecutionReport
from airc_skill.pipeline import decide_write_gate
from airc_skill.reviewer import review_rewrite
from airc_skill.rewriter import RewriteStats, Rewriter
from airc_skill.skill_protocol import SkillInputSchema, build_execution_plan


ALLOWED_REVISION_PATTERNS = {
    "compress",
    "expand",
    "merge",
    "split",
    "reorder",
    "soften",
    "reframe",
    "partial_keep",
    "rewrite_all",
}


def _body_paragraph() -> str:
    return (
        "人工智能写作系统正在改变高校论文训练中的反馈流程。"
        "教师需要在效率提升和质量控制之间重新分配注意力。"
        "学生也需要理解生成内容与人工判断之间的边界。"
        "因此，评价机制需要保持稳定、透明和可追溯。"
    )


def _uniform_revised_paragraph() -> str:
    return (
        "高校论文训练中的反馈流程正在被智能写作系统重新组织。"
        "教师在效率提升与质量控制之间重新安排注意力。"
        "学生需要继续辨认生成内容和人工判断之间的边界。"
        "评价机制因此需要保持稳定、透明和可追溯。"
    )


def _very_long_body(paragraphs: int = 180) -> str:
    return "\n\n".join(_body_paragraph() for _ in range(paragraphs))


def _uniform_very_long_rewrite(paragraphs: int = 180) -> str:
    return "\n\n".join(_uniform_revised_paragraph() for _ in range(paragraphs))


def _stats_for_policy(block_id: int, original: str, revised: str, *, coverage: float = 1.0) -> RewriteStats:
    original_sentences = [
        "人工智能写作系统正在改变高校论文训练中的反馈流程。",
        "教师需要在效率提升和质量控制之间重新分配注意力。",
        "学生也需要理解生成内容与人工判断之间的边界。",
        "因此，评价机制需要保持稳定、透明和可追溯。",
    ]
    revised_sentences = [
        "高校论文训练中的反馈流程正在被智能写作系统重新组织。",
        "教师在效率提升与质量控制之间重新安排注意力。",
        "学生需要继续辨认生成内容和人工判断之间的边界。",
        "评价机制因此需要保持稳定、透明和可追溯。",
    ]
    changed_units = max(1, round(len(original_sentences) * coverage))
    return RewriteStats(
        mode=RewriteMode.BALANCED,
        changed=True,
        applied_rules=["fusion:context-followup"],
        sentence_count_before=len(original_sentences),
        sentence_count_after=len(revised_sentences),
        sentence_level_change=True,
        changed_characters=48,
        original_sentences=original_sentences,
        rewritten_sentences=revised_sentences,
        paragraph_char_count=len(original),
        sentence_labels=["background", "detail", "support", "conclusion"],
        subject_heads=["反馈流程", "教师", "学生", "评价机制"],
        detected_patterns=[],
        structural_actions=["sentence_cluster_merge", "pair_fusion"],
        structural_action_count=2,
        high_value_structural_actions=["sentence_cluster_merge", "pair_fusion"],
        discourse_actions_used=["sentence_cluster_rewrite", "sentence_cluster_merge", "transition_absorption"],
        sentence_level_changes=changed_units,
        cluster_changes=1,
        discourse_change_score=5,
        rewrite_coverage=coverage,
        prefix_only_rewrite=False,
        repeated_subject_risk=False,
        selected_variants=[],
        candidate_notes=[],
        paragraph_index=block_id,
        block_id=block_id,
        rewrite_depth="developmental_rewrite",
        rewrite_intensity="high",
        revision_patterns=["rewrite_all"],
        human_noise_marks=[],
    )


def test_guidance_assigns_revision_patterns_to_body_blocks() -> None:
    text = "\n\n".join(
        [
            _body_paragraph(),
            (
                "本研究的主题为课堂反馈机制的优化路径。"
                "本研究进一步讨论系统输出与人工评价之间的协同方式。"
                "与此同时，课程规模扩大也提高了反馈流程的组织压力。"
            ),
            (
                "随着课程任务不断增加，传统反馈流程在时效性和覆盖范围方面暴露出不足。"
                "学生对反馈的针对性和连续性提出更高要求。"
                "因此，需要重新组织人机协同的评价机制。"
            ),
        ]
    )

    guidance = guide_document_text(text, metadata={"suffix": ".txt"})
    body_policies = [policy for policy in guidance.block_policies if policy.should_rewrite]
    patterns = [pattern for policy in body_policies for pattern in policy.revision_pattern]
    plan = build_execution_plan(guidance, SkillInputSchema.from_path("paper.txt"))

    assert body_policies
    assert all(1 <= len(policy.revision_pattern) <= 2 for policy in body_policies)
    assert set(patterns).issubset(ALLOWED_REVISION_PATTERNS)
    assert len(set(patterns)) >= 2
    assert all("revision_pattern" in block for block in plan.block_policies)
    assert "revision_pattern" in plan.human_agent_instructions


def test_rewriter_emits_sentence_cluster_and_human_variation_signals() -> None:
    original = (
        "当前课堂场景正在发生变化。"
        "本研究的主题为课堂反馈机制的优化路径。"
        "与此同时，文章还讨论了教师如何根据反馈调整任务设计。"
        "因此，构建稳定的反馈链条十分关键。"
    )

    revised, stats = Rewriter().rewrite(original, mode=RewriteMode.STRONG)
    review = review_rewrite(original, revised, mode=RewriteMode.STRONG, rewrite_stats=[stats])

    assert revised != original
    assert stats.cluster_changes > 0
    assert "sentence_cluster_merge" in stats.discourse_actions_used
    assert set(stats.revision_patterns) & {"merge", "reorder", "reframe"}
    assert stats.human_noise_marks
    assert review.sentence_cluster_changes_present is True
    assert review.narrative_flow_changed is True
    assert review.non_uniform_rewrite_distribution is True
    assert review.human_like_variation is True


def test_reviewer_accepts_non_uniform_revision_distribution_signals() -> None:
    original = "\n\n".join(_body_paragraph() for _ in range(4))
    revised = "\n\n".join(_uniform_revised_paragraph() for _ in range(4))
    stats = [
        _stats_for_policy(1, _body_paragraph(), _uniform_revised_paragraph(), coverage=0.25),
        _stats_for_policy(2, _body_paragraph(), _uniform_revised_paragraph(), coverage=1.0),
        _stats_for_policy(3, _body_paragraph(), _uniform_revised_paragraph(), coverage=0.75),
        _stats_for_policy(4, _body_paragraph(), _uniform_revised_paragraph(), coverage=0.5),
    ]
    stats[0].revision_patterns = ["partial_keep", "soften"]
    stats[1].revision_patterns = ["rewrite_all", "merge"]
    stats[2].revision_patterns = ["reorder", "reframe"]
    stats[3].revision_patterns = ["compress", "split"]
    stats[0].human_noise_marks = ["partial_keep"]
    stats[2].human_noise_marks = ["heavy_light_block_contrast"]

    review = review_rewrite(original, revised, mode=RewriteMode.BALANCED, rewrite_stats=stats)

    assert review.sentence_cluster_changes_present is True
    assert review.non_uniform_rewrite_distribution is True
    assert review.human_like_variation is True
    assert review.revision_pattern_distribution["partial_keep"] == 1
    assert "heavy_light_block_contrast" in review.human_noise_marks


def test_very_long_uniform_rewrite_is_rejected_by_human_gate() -> None:
    original = _very_long_body()
    revised = _uniform_very_long_rewrite()
    guidance = guide_document_text(original, metadata={"suffix": ".txt"})
    stats = [
        _stats_for_policy(policy.block_id, policy.original_text, _uniform_revised_paragraph())
        for policy in guidance.block_policies
        if policy.should_rewrite
    ]
    review = review_rewrite(
        original,
        revised,
        guidance=guidance,
        mode=RewriteMode.BALANCED,
        rewrite_stats=stats,
        suffix=".txt",
    )
    rewrite_report = RewriteExecutionReport(
        rewritten_text=revised,
        block_candidates=[],
        rewrite_stats=stats,
        mode_requested=RewriteMode.BALANCED,
        mode_used=RewriteMode.BALANCED,
        effective_change=True,
        changed_block_ids=[stats_item.block_id for stats_item in stats],
        candidate_count=1,
        selected_candidate_reason="Uniform long-document rewrite candidate.",
        convenience_mode=False,
        block_failures=[],
        reviewed=True,
    )
    gate = decide_write_gate(review, rewrite_report=rewrite_report, policy={})

    assert review.document_scale == "very_long"
    assert review.body_rewrite_coverage >= 0.60
    assert review.body_changed_block_ratio >= 0.50
    assert review.body_discourse_change_score >= review.required_body_discourse_change_score
    assert review.body_cluster_rewrite_score >= review.required_body_cluster_rewrite_score
    assert review.sentence_cluster_changes_present is True
    assert review.non_uniform_rewrite_distribution is False
    assert review.human_like_variation is False
    assert gate.write_allowed is False
    assert "human_like_variation_missing_for_long_document" in gate.reason_codes
    assert "non_uniform_rewrite_distribution_missing_for_long_document" in gate.reason_codes
