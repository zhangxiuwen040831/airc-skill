from __future__ import annotations

import re
from pathlib import Path

import pytest

from airc_skill.config import RewriteMode
from airc_skill.pipeline import run_file
from airc_skill.reviewer import review_rewrite
from airc_skill.rewriter import Rewriter, RuleBasedRewriteBackend, split_sentences
from airc_skill.semantic_role_integrity import (
    aggregate_semantic_role_integrity,
    analyze_semantic_role_integrity,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_TEST_MD = ROOT / "test.md"
REAL_V6_MD = ROOT / "test_airc_v6.md"


def _count_patterns(text: str, patterns: tuple[str, ...]) -> int:
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


def test_scaffolding_phrase_density_detects_generated_support_style() -> None:
    original = split_sentences("本研究围绕模型部署稳定性讨论约束设计。该机制限制噪声分支对主判据的干扰。")
    revised = split_sentences("这一工作围绕模型部署稳定性展开。相关内容还包括限制噪声分支对主判据的干扰。")

    signals = analyze_semantic_role_integrity(original, revised, high_sensitivity=True)

    assert signals.scaffolding_phrase_density > 0.2
    assert signals.over_abstracted_subject_risk > 0.2
    assert signals.semantic_role_integrity_score < 0.8


def test_enumeration_integrity_preserved_for_innovation_blocks() -> None:
    original = (
        "（1）base_only 判别机制：最终模型将语义与频域特征作为核心决策路径。"
        "（2）困难真实样本课程学习：本研究通过挖掘接近决策边界的真实样本构建缓冲区。"
    )

    revised, stats = Rewriter().rewrite(
        original,
        mode=RewriteMode.STRONG,
        rewrite_depth="developmental_rewrite",
        rewrite_intensity="high",
        high_sensitivity_prose=True,
    )
    review = review_rewrite(original, revised, mode=RewriteMode.STRONG, rewrite_stats=[stats])

    assert review.enumeration_integrity_preserved is True
    assert review.semantic_role_integrity_preserved is True
    assert "还包括" not in revised


def test_fragmented_enumeration_item_restores_direct_flow() -> None:
    backend = RuleBasedRewriteBackend()
    original_sentences = split_sentences(
        "（3）脆弱 AIGC 样本支持：若仅强调困难真实样本的优化，易导致模型过于保守，导致 AIGC 样本漏检。"
        "因此训练中同时维护脆弱 AIGC 缓冲区，对边界附近的正样本给予支持。"
    )
    fragmented = split_sentences(
        "（3）脆弱 AIGC 样本支持：若仅强调困难真实样本的优化。"
        "训练中同时维护脆弱 AIGC 缓冲区，对边界附近的正样本给予支持。"
        "易导致模型过于保守，导致 AIGC 样本漏检。"
    )

    repaired, rules, notes = backend._repair_enumeration_support_fragments(
        original_sentences=original_sentences,
        sentences=fragmented,
    )

    assert repaired == [backend._ensure_sentence_end(sentence) for sentence in original_sentences]
    assert "local:repair-enumeration-flow" in rules
    assert any("restored the original enumeration item" in note for note in notes)


def test_mechanism_sentence_not_rewritten_as_appendix_like_support() -> None:
    original = (
        "最终模型将语义与频域特征作为核心决策路径。"
        "主融合模块通过门控机制对语义与频域特征进行加权融合，生成基础特征与基础对数几率。"
    )

    revised, stats = Rewriter().rewrite(
        original,
        mode=RewriteMode.STRONG,
        rewrite_depth="developmental_rewrite",
        rewrite_intensity="high",
    )
    aggregate = aggregate_semantic_role_integrity([stats])

    assert aggregate["semantic_role_integrity_preserved"] is True
    assert aggregate["scaffolding_phrase_density_controlled"] is True
    assert "还包括" not in revised
    assert "这一机制" not in revised


def test_v7_reduces_huanbaokuo_style_expansion(tmp_path: Path) -> None:
    if not REAL_TEST_MD.exists() or not REAL_V6_MD.exists():
        pytest.skip("real test.md or v6 output is not available in this checkout")

    output = tmp_path / "test_airc_v7.md"
    report = tmp_path / "test_airc_v7.report.json"
    result = run_file(REAL_TEST_MD, output_path=output, report_path=report)

    v6_text = REAL_V6_MD.read_text(encoding="utf-8")

    assert result.output_written is True
    v7_text = output.read_text(encoding="utf-8")
    assert _count_patterns(v7_text, (r"还包括", r"也包括", r"进一步包括")) <= _count_patterns(
        v6_text,
        (r"还包括", r"也包括", r"进一步包括"),
    )


def test_abstracted_subject_risk_controlled_in_high_sensitivity_prose() -> None:
    original = (
        "本研究的主题为“人工智能生成图像检测工具的设计与实现”。"
        "本研究不仅包含模型训练与算法设计，还完成了后端推理服务与前端交互系统。"
    )

    revised, stats = Rewriter().rewrite(
        original,
        mode=RewriteMode.STRONG,
        rewrite_depth="developmental_rewrite",
        rewrite_intensity="high",
        high_sensitivity_prose=True,
    )
    aggregate = aggregate_semantic_role_integrity([stats])

    assert aggregate["over_abstracted_subject_risk_controlled"] is True
    assert aggregate["scaffolding_phrase_density_controlled"] is True
    assert not re.search(r"^(?:这一工作|这项工作|相关内容)", split_sentences(revised)[-1])
