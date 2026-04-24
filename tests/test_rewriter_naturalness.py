from collections import Counter

from airc_skill.config import RewriteMode
from airc_skill.rewriter import Rewriter


def test_rewrite_not_only_prefix_substitution() -> None:
    text = (
        "数字平台正在持续进入课堂评价场景。"
        "与此同时，教师还需要根据系统反馈调整任务安排。"
        "生成工具、营销内容、推荐机制，都会对学生判断产生影响。"
        "对于现实系统而言，平台风险并不局限于模型误差，更在于反馈链条不稳定。"
        "因此，构建稳定的校正机制十分必要。"
    )

    rewritten, stats = Rewriter().rewrite(text, mode=RewriteMode.BALANCED)

    assert rewritten != text
    assert stats.prefix_only_rewrite is False
    assert stats.structural_action_count >= 1
    assert any(
        action in {"pair_fusion", "conclusion_absorb", "enumeration_reframe", "sentence_split", "sentence_merge"}
        for action in stats.structural_actions
    )


def test_rewrite_avoids_repeated_patterns() -> None:
    text = (
        "本研究的主题为课堂反馈机制的优化路径。"
        "与此同时，文章还讨论了教师如何根据反馈调整任务设计。"
        "因此，构建稳定的反馈链条十分关键。"
        "本研究的主题为写作评价流程的优化方式。"
        "与此同时，文章还讨论了平台输出如何影响评分判断。"
        "因此，构建更稳健的人工校对机制十分必要。"
    )

    rewritten, stats = Rewriter().rewrite(text, mode=RewriteMode.STRONG)
    counts = Counter(stats.selected_variants)

    assert rewritten != text
    assert max(counts.values(), default=0) <= 2
    assert rewritten.count("这也意味着") <= 1
    assert rewritten.count("本研究聚焦于") <= 1


def test_strong_has_paragraph_level_reorganization() -> None:
    text = (
        "当前课堂场景正在发生变化。"
        "本研究的主题为课堂反馈机制的优化路径。"
        "与此同时，文章还讨论了教师如何根据反馈调整任务设计。"
        "因此，构建稳定的反馈链条十分关键。"
    )

    balanced_text, balanced_stats = Rewriter().rewrite(text, mode=RewriteMode.BALANCED)
    strong_text, strong_stats = Rewriter().rewrite(text, mode=RewriteMode.STRONG)

    assert balanced_text != strong_text
    assert "paragraph_reorder" not in balanced_stats.structural_actions
    assert "paragraph_reorder" in strong_stats.structural_actions
