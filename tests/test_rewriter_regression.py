from pathlib import Path

from airc_skill.config import RewriteMode
from airc_skill.pipeline import rewrite_file
from airc_skill.rewriter import Rewriter, split_sentences


def _load_user_test_md_excerpt() -> str:
    root = Path(__file__).resolve().parents[1] / "test.md"
    text = root.read_text(encoding="utf-8")
    start = text.index("## 1.1 研究背景")
    end = text.index("## 1.2 研究意义")
    return text[start:end]


def _max_benyan_streak(text: str) -> int:
    streak = 0
    current = 0
    for sentence in split_sentences(text):
        if sentence.startswith("本研究"):
            current += 1
        else:
            current = 0
        streak = max(streak, current)
    return streak


def test_no_bad_semicolon_in_cn_text() -> None:
    source = (
        "生成工具，已经能够在多个环节提供辅助。"
        "营销内容，都会对用户判断产生影响。"
        "因此，本研究需要重新评估人工校对流程。"
    )

    rewritten, _ = Rewriter().rewrite(source, mode=RewriteMode.STRONG)

    assert "生成工具；" not in rewritten
    assert "营销内容；" not in rewritten
    assert "因此；" not in rewritten
    assert "；" not in rewritten


def test_balanced_has_meaningful_change(tmp_path: Path) -> None:
    source = tmp_path / "balanced.txt"
    original = (
        "对于现实系统而言，平台风险并不局限于模型误差，更在于反馈链条不稳定。"
        "本研究的主题为多轮写作优化流程。"
        "与此同时，文章还讨论了教师如何根据系统输出调整评价方式。"
        "因此，构建稳定的校正机制十分必要。"
        "当前讨论已经指出，相关流程在很多方面都具有重要意义，但现有表述仍然比较概括。"
    )
    source.write_text(original, encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.BALANCED)

    assert result.mode_used == RewriteMode.BALANCED
    assert result.review.meaningful_change is True
    assert result.text != original
    assert result.review.structural_action_count >= 1
    assert result.review.prefix_only_rewrite is False


def test_strong_differs_from_balanced(tmp_path: Path) -> None:
    source = tmp_path / "strong.txt"
    original = (
        "总的来说，数字平台正在改变课堂互动方式。"
        "本研究的主题为课堂反馈机制的优化路径。"
        "与此同时，文章还讨论了教师如何根据反馈调整任务设计。"
        "因此，构建稳定的反馈链条十分关键。"
    )
    source.write_text(original, encoding="utf-8")

    balanced = rewrite_file(source, mode=RewriteMode.BALANCED)
    strong = rewrite_file(source, mode=RewriteMode.STRONG)

    assert balanced.mode_used == RewriteMode.BALANCED
    assert strong.mode_used == RewriteMode.STRONG
    assert strong.review.meaningful_change is True
    assert strong.text != balanced.text
    assert strong.review.structural_action_count >= 2


def test_markdown_structure_preserved_after_stronger_rewrite(tmp_path: Path) -> None:
    source = tmp_path / "paper.md"
    original = """# 研究背景

总的来说，数字平台正在改变课堂互动方式。本研究的主题为课堂反馈机制的优化路径。与此同时，文章还讨论了教师如何根据反馈调整任务设计。因此，构建稳定的反馈链条十分关键。

## 设计

- 保留列表项 A
- 保留列表项 B

```python
print("do not rewrite code")
```

参考链接见 [Example](https://example.com)。
"""
    source.write_text(original, encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.STRONG)
    revised = result.text

    assert result.mode_used == RewriteMode.STRONG
    assert original.count("# ") + original.count("## ") == revised.count("# ") + revised.count("## ")
    assert original.count("- ") == revised.count("- ")
    assert original.count("```") == revised.count("```")
    assert original.count("[Example](https://example.com)") == revised.count(
        "[Example](https://example.com)"
    )


def test_scope_risk_phrase_stays_natural() -> None:
    source = (
        "对于现实系统而言，AIGC图像带来的风险并不局限于模型误差，更在于反馈链条不稳定。"
        "因此，本研究需要重新评估人工校对流程。"
    )

    rewritten, _ = Rewriter().rewrite(source, mode=RewriteMode.BALANCED)

    assert "风险的风险" not in rewritten
    assert "风险面临的压力" not in rewritten
    assert "更集中地体现在" not in rewritten


def test_balanced_on_user_test_md_produces_effective_change(tmp_path: Path) -> None:
    source = tmp_path / "user_test.md"
    source.write_text(_load_user_test_md_excerpt(), encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.BALANCED)

    assert result.effective_change is True
    assert result.review.decision in {"pass", "pass_with_minor_risk", "reject"}


def test_user_test_md_subject_chain_is_improved(tmp_path: Path) -> None:
    source = tmp_path / "user_chain.md"
    excerpt = _load_user_test_md_excerpt()
    source.write_text(excerpt, encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.BALANCED)

    assert result.review.repeated_subject_risk is False
    assert _max_benyan_streak(result.text) < 3


def test_user_test_md_not_only_prefix_change(tmp_path: Path) -> None:
    source = tmp_path / "user_prefix.md"
    excerpt = _load_user_test_md_excerpt()
    source.write_text(excerpt, encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.BALANCED)

    assert result.review.structural_action_count >= 1
    assert result.text != excerpt


def test_checkpoint_sentence_keeps_explicit_benyan_subject() -> None:
    text = (
        "本研究最终采用 checkpoints/best.pth 作为本地部署模型，对应 V10 Phase2 epoch_008，默认运行于 base_only 模式。"
        "本研究同时完成了前端交互系统设计。"
    )

    rewritten, _ = Rewriter().rewrite(text, mode=RewriteMode.BALANCED)

    first_sentence = split_sentences(rewritten)[0]
    assert first_sentence.startswith("本研究最终采用 checkpoints/best.pth 作为本地部署模型")
