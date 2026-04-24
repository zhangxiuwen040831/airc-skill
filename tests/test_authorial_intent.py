from __future__ import annotations

import re
from pathlib import Path

import pytest

from airc_skill.authorial_intent import aggregate_authorial_intent, analyze_authorial_intent
from airc_skill.config import RewriteMode
from airc_skill.pipeline import run_file
from airc_skill.rewriter import Rewriter, RuleBasedRewriteBackend, split_sentences


ROOT = Path(__file__).resolve().parents[1]
REAL_TEST_MD = ROOT / "test.md"
REAL_V7_MD = ROOT / "test_airc_v7.md"


def _count_authorial_residue(text: str) -> int:
    patterns = (
        r"本研究用于",
        r"通过这种方式",
        r"可以看出",
        r"这表明",
        r"从[^。；]{2,20}角度来看",
        r"在该设置下",
    )
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


def test_appendix_like_support_ratio_detects_generated_support_style() -> None:
    original = split_sentences("主融合模块通过门控机制对语义与频域特征进行加权融合，生成基础特征。")
    revised = split_sentences("通过这种方式可以实现语义与频域特征融合。")

    signals = analyze_authorial_intent(original, revised, high_sensitivity=True)

    assert signals.appendix_like_support_ratio > 0.2
    assert signals.assertion_strength_score < 0.6


def test_mechanism_sentence_verbs_are_strengthened_after_rewrite() -> None:
    backend = RuleBasedRewriteBackend()
    repaired = backend._strengthen_mechanism_verb(
        sentence="该机制用于刻画图像残差，并可以实现频域约束。",
        original_sentence="噪声分支用于刻画图像残差，并负责约束频域异常响应。",
    )

    assert repaired.startswith("噪声分支")
    assert "负责刻画" in repaired
    assert "直接实现" in repaired or "负责约束" in repaired


def test_authorial_choice_expression_is_restored_from_source() -> None:
    backend = RuleBasedRewriteBackend()
    repaired = backend._restore_authorial_choice_expression(
        sentence="说明后期优化重点是继续压低误报区域。",
        original_sentence="说明后期优化重点并非继续推高已较早达到高位的 Recall，而是通过结构收缩与课程学习持续压缩真实样本误判区域。",
        role="conclusion",
        high_sensitivity_prose=False,
    )

    assert "并非" in repaired
    assert "而是" in repaired


def test_authorial_intent_metrics_present_in_rewrite_stats() -> None:
    original = (
        "本研究围绕AIGC图像检测任务构建整体方案。"
        "通过这种方式可以实现语义与频域特征融合。"
        "说明后期优化重点是继续压低误报区域。"
    )

    revised, stats = Rewriter().rewrite(
        original,
        mode=RewriteMode.STRONG,
        rewrite_depth="developmental_rewrite",
        rewrite_intensity="high",
        high_sensitivity_prose=True,
    )
    aggregate = aggregate_authorial_intent([stats])

    assert revised != original
    assert aggregate["assertion_strength_preserved"] is True
    assert aggregate["appendix_like_support_controlled"] is True
    assert aggregate["authorial_stance_present"] is True


def test_v8_reduces_appendix_like_support_vs_v7(tmp_path: Path) -> None:
    if not REAL_TEST_MD.exists() or not REAL_V7_MD.exists():
        pytest.skip("real test.md or v7 output is not available in this checkout")

    output = tmp_path / "test_airc_v8.md"
    report = tmp_path / "test_airc_v8.report.json"
    result = run_file(REAL_TEST_MD, output_path=output, report_path=report)

    assert result.output_written is True
    assert _count_authorial_residue(output.read_text(encoding="utf-8")) <= _count_authorial_residue(
        REAL_V7_MD.read_text(encoding="utf-8")
    )
