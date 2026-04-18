from pathlib import Path

from airc_skill.config import RewriteMode
from airc_skill.pipeline import guide_file, rewrite_file
from airc_skill.rewriter import Rewriter, split_sentences


def _build_guidance_sample() -> str:
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


def test_skill_returns_rewrite_guidance_not_only_direct_rewrite(tmp_path: Path) -> None:
    source = tmp_path / "guide.md"
    source.write_text(_build_guidance_sample(), encoding="utf-8")

    guidance = guide_file(source)

    assert guidance.do_not_touch_blocks
    assert guidance.rewrite_candidate_blocks
    assert guidance.block_policies
    assert guidance.rewrite_actions_by_block
    assert guidance.write_gate_preconditions
    assert guidance.write_gate_decision == "guidance_only"


def test_high_risk_blocks_are_marked_do_not_touch_or_high_risk(tmp_path: Path) -> None:
    source = tmp_path / "risk.md"
    source.write_text(_build_guidance_sample(), encoding="utf-8")

    guidance = guide_file(source)

    assert any(block.block_kind == "heading" for block in guidance.do_not_touch_blocks)
    assert any(block.block_kind == "english_block" for block in guidance.do_not_touch_blocks)
    assert guidance.high_risk_blocks
    assert any(block.risk_level == "high_risk" for block in guidance.high_risk_blocks)
    assert any("keep_explicit_subject_if_needed" in block.recommended_actions for block in guidance.high_risk_blocks)


def test_paragraph_rewrite_actions_are_recommended_by_risk_level(tmp_path: Path) -> None:
    source = tmp_path / "actions.md"
    source.write_text(_build_guidance_sample(), encoding="utf-8")

    guidance = guide_file(source)

    light_edit_blocks = [block for block in guidance.rewrite_candidate_blocks if block.risk_level == "light_edit"]
    rewritable_blocks = [block for block in guidance.rewrite_candidate_blocks if block.risk_level == "rewritable"]

    assert light_edit_blocks
    assert rewritable_blocks
    assert any("compress_meta_discourse" in block.recommended_actions for block in light_edit_blocks)
    assert any("merge_or_split_sentence_cluster" in block.recommended_actions for block in rewritable_blocks)


def test_repeated_subject_reduced_without_losing_clarity(tmp_path: Path) -> None:
    source = tmp_path / "clarity.txt"
    original = (
        "本研究最终采用 checkpoints/best.pth 作为本地部署模型，对应 V10 Phase2 epoch_008。\n\n"
        "本研究同时完成了前端交互系统设计。"
        "本研究进一步梳理了阈值切换策略。"
    )
    source.write_text(original, encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.BALANCED)

    assert "本研究最终采用 checkpoints/best.pth 作为本地部署模型" in result.text
    assert _max_benyan_streak(result.text) < 3


def test_meta_compression_improves_flow() -> None:
    text = (
        "本研究的主题为课堂反馈机制的优化路径。"
        "本研究不仅包含任务设计分析，还完成了反馈链条的流程梳理。"
    )

    rewritten, stats = Rewriter().rewrite(text, mode=RewriteMode.BALANCED)

    assert rewritten != text
    assert "meta_compression" in stats.structural_actions
    assert "本研究的主题为" not in rewritten


def test_keep_original_when_rewrite_would_be_stiff() -> None:
    text = "总的来说，这一问题在很多方面都具有重要意义。"

    rewritten, stats = Rewriter().rewrite(text, mode=RewriteMode.CONSERVATIVE)

    assert rewritten == text
    assert not stats.applied_rules


def test_humanized_style_is_academic_not_colloquial(tmp_path: Path) -> None:
    source = tmp_path / "formal.txt"
    source.write_text(
        "近年来，数字平台正在持续进入课堂评价场景。与此同时，教师还需要根据系统反馈调整任务安排，因此有必要重新评估人工校对流程。",
        encoding="utf-8",
    )

    result = rewrite_file(source, mode=RewriteMode.BALANCED)

    for marker in ("这块", "这边", "大家", "我们", "里头", "真的", "超级"):
        assert marker not in result.text


def test_real_test_md_guidance_marks_agent_safe_blocks(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    original = (root / "test.md").read_text(encoding="utf-8")
    source = tmp_path / "user_test.md"
    source.write_text(original, encoding="utf-8")

    guidance = guide_file(source)
    result = rewrite_file(source, mode=RewriteMode.BALANCED)

    assert guidance.do_not_touch_blocks
    assert guidance.rewrite_candidate_blocks
    assert result.guidance.write_gate_decision in {"pass", "pass_with_minor_risk", "reject"}
