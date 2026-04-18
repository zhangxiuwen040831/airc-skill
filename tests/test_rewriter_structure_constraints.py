from airc_skill.config import RewriteMode
from airc_skill.reviewer import review_revision
from airc_skill.rewriter import RewriteStats, Rewriter, split_sentences

_META_SUBJECTS = ("本研究", "本文", "该研究", "该系统")


def _max_meta_subject_streak(text: str) -> int:
    streak = 0
    current = ""
    max_streak = 0
    for sentence in split_sentences(text):
        subject = next((item for item in _META_SUBJECTS if sentence.startswith(item)), "")
        if subject and subject == current:
            streak += 1
        elif subject:
            current = subject
            streak = 1
        else:
            current = ""
            streak = 0
        max_streak = max(max_streak, streak)
    return max_streak


def test_balanced_requires_structural_action() -> None:
    text = (
        "当前课堂评价场景正在持续数字化，平台生成能力也在不断进入教师日常工作流程。"
        "本研究的主题为多轮写作反馈机制的优化路径。"
        "与此同时，文章还讨论了教师如何根据系统输出调整任务安排。"
        "生成工具、营销内容、推荐机制，都会对学生判断产生影响。"
        "因此，构建稳定的人工校正机制十分必要。"
    )

    _, stats = Rewriter().rewrite(text, mode=RewriteMode.BALANCED)

    assert stats.paragraph_char_count >= 120
    assert stats.structural_action_count >= 1


def test_strong_requires_multiple_structural_actions() -> None:
    text = (
        "当前课堂评价场景正在持续数字化，平台生成能力也在不断进入教师日常工作流程。"
        "本研究的主题为多轮写作反馈机制的优化路径。"
        "与此同时，文章还讨论了教师如何根据系统输出调整任务安排。"
        "生成工具、营销内容、推荐机制，都会对学生判断产生影响。"
        "对于现实系统而言，平台风险并不局限于模型误差，更在于反馈链条不稳定。"
        "因此，构建稳定的人工校正机制十分必要。"
    )

    _, stats = Rewriter().rewrite(text, mode=RewriteMode.STRONG)

    assert stats.paragraph_char_count >= 120
    assert stats.structural_action_count >= 2
    assert len(stats.high_value_structural_actions) >= 1


def test_prefix_only_rewrite_is_rejected() -> None:
    original = (
        "当前课堂评价场景正在持续数字化。"
        "与此同时，教师还需要根据系统反馈调整任务安排。"
        "因此，本研究需要重新评估人工校对流程。"
        "本研究的主题为多轮写作反馈机制的优化路径。"
    )
    revised = (
        "当前课堂评价场景正在持续数字化。"
        "同时，教师还需要根据系统反馈调整任务安排。"
        "基于此，本研究需要重新评估人工校对流程。"
        "本研究的主题为多轮写作反馈机制的优化路径。"
    )
    stats = RewriteStats(
        mode=RewriteMode.BALANCED,
        changed=True,
        applied_rules=["opening:transition", "opening:implication"],
        sentence_count_before=4,
        sentence_count_after=4,
        sentence_level_change=False,
        changed_characters=12,
        original_sentences=[
            "当前课堂评价场景正在持续数字化。",
            "与此同时，教师还需要根据系统反馈调整任务安排。",
            "因此，本研究需要重新评估人工校对流程。",
            "本研究的主题为多轮写作反馈机制的优化路径。",
        ],
        rewritten_sentences=[
            "当前课堂评价场景正在持续数字化。",
            "同时，教师还需要根据系统反馈调整任务安排。",
            "基于此，本研究需要重新评估人工校对流程。",
            "本研究的主题为多轮写作反馈机制的优化路径。",
        ],
        paragraph_char_count=150,
        sentence_labels=["background", "support", "conclusion", "objective"],
        subject_heads=["当前课堂", "教师", "本研究", "本研究"],
        detected_patterns=["background_transition", "risk_objective"],
        structural_actions=[],
        structural_action_count=0,
        high_value_structural_actions=[],
        prefix_only_rewrite=True,
        repeated_subject_risk=True,
        selected_variants=["同时", "基于此"],
        candidate_notes=[],
        paragraph_index=1,
    )

    review = review_revision(original, revised, RewriteMode.BALANCED, rewrite_stats=[stats])

    assert review.ok is False
    assert review.decision == "reject"
    assert review.prefix_only_rewrite is True


def test_repeated_benyan_subject_is_reduced() -> None:
    text = (
        "本研究尝试回应课堂反馈链条不稳定这一问题。"
        "本研究不仅包含反馈建模，还完成了人工校正流程设计。"
        "因此，本研究不仅关注模型输出，也强调教师复核机制。"
    )

    rewritten, stats = Rewriter().rewrite(text, mode=RewriteMode.STRONG)

    assert rewritten != text
    assert _max_meta_subject_streak(rewritten) < 3
    assert any(
        action in {"merge_consecutive_subject_sentences", "meta_compression", "followup_absorb", "subject_variation"}
        for action in stats.structural_actions
    )


def test_meta_plus_work_sentence_gets_compressed() -> None:
    text = (
        "本研究的主题为课堂反馈机制的优化路径。"
        "本研究不仅包含任务设计分析，还完成了反馈链条的流程梳理。"
    )

    rewritten, stats = Rewriter().rewrite(text, mode=RewriteMode.BALANCED)

    assert rewritten != text
    assert len(split_sentences(rewritten)) <= 2
    assert any(action in {"meta_compression", "merge_consecutive_subject_sentences"} for action in stats.structural_actions)
    assert "本研究的主题为" not in rewritten


def test_followup_absorb_reduces_subject_repetition() -> None:
    text = (
        "相关流程由数据采集、人工复核和结果反馈共同形成完整闭环。"
        "因此，本研究不仅关注输出质量，同时强调教师参与机制。"
    )

    rewritten, stats = Rewriter().rewrite(text, mode=RewriteMode.BALANCED)

    assert rewritten != text
    assert any(action in {"followup_absorb", "conclusion_absorb"} for action in stats.structural_actions)
    assert "因此，本研究" not in rewritten


def test_no_duplicate_subject_chain_in_balanced() -> None:
    text = (
        "本研究的主题为课堂反馈机制的优化路径。"
        "本研究不仅包含多轮写作任务设计，还完成了教师复核流程梳理。"
        "因此，本研究进一步讨论了系统输出与人工评价之间的协同方式。"
        "本研究同时强调真实课堂场景中的风险控制。"
    )

    rewritten, stats = Rewriter().rewrite(text, mode=RewriteMode.BALANCED)

    assert rewritten != text
    assert stats.repeated_subject_risk is False
    assert _max_meta_subject_streak(rewritten) < 2


def test_conclusion_absorb_or_pair_fusion_present() -> None:
    text = (
        "生成工具、营销内容、推荐机制，都会对学生判断造成压力。"
        "因此，构建稳定的人工校正机制十分必要。"
        "与此同时，教师还需要根据系统反馈调整任务设计。"
    )

    balanced_text, balanced_stats = Rewriter().rewrite(text, mode=RewriteMode.BALANCED)
    strong_text, strong_stats = Rewriter().rewrite(text, mode=RewriteMode.STRONG)

    assert balanced_text != text
    assert strong_text != text
    assert any(
        action in {"conclusion_absorb", "pair_fusion"}
        for action in balanced_stats.structural_actions + strong_stats.structural_actions
    )
