from __future__ import annotations

import re
from pathlib import Path

import pytest

from airc_skill.config import RewriteMode
from airc_skill.evidence_fidelity import aggregate_evidence_fidelity, analyze_evidence_fidelity
from airc_skill.pipeline import run_file
from airc_skill.rewriter import RuleBasedRewriteBackend, split_sentences


ROOT = Path(__file__).resolve().parents[1]
REAL_TEST_MD = ROOT / "test.md"
REAL_V7_MD = ROOT / "test_airc_v7.md"


def _count_externalized_commentary(text: str) -> int:
    patterns = (
        r"主流观点",
        r"业界通常认为",
        r"普遍认为",
        r"超过八成",
        r"超过半数",
        r"幻想的破灭",
        r"边界修复手术",
        r"判决书",
        r"挣脱出来",
        r"终于摆脱",
        r"^我们",
        r"^我们的",
    )
    return sum(len(re.findall(pattern, text, flags=re.MULTILINE)) for pattern in patterns)


def test_unsupported_expansion_risk_detects_external_background_injection() -> None:
    original = split_sentences("本研究围绕AIGC图像检测任务展开，重点讨论部署稳定性。")
    revised = split_sentences("在2024年AIGC泛滥的内容生态中，超过八成的虚假信息投诉指向图像伪造。")

    signals = analyze_evidence_fidelity(original, revised, high_sensitivity=True)

    assert signals.unsupported_expansion_risk > 0.4
    assert signals.evidence_fidelity_score < 0.6


def test_metaphor_or_storytelling_risk_detects_commentary_style() -> None:
    original = split_sentences("系统通过统一接口对外提供检测能力。")
    revised = split_sentences("整个过程如同一次精密的边界修复手术，最终给出判决书。")

    signals = analyze_evidence_fidelity(original, revised, high_sensitivity=True)

    assert signals.metaphor_or_storytelling_risk > 0.3
    assert signals.thesis_tone_restraint_score < 0.7


def test_thesis_tone_restraint_preserves_academic_register() -> None:
    original = split_sentences("系统通过统一接口对外提供检测能力，并对异常样本执行分层分析。")
    revised = split_sentences("系统通过统一接口对外提供检测能力，并对异常样本执行分层分析。")

    signals = analyze_evidence_fidelity(original, revised, high_sensitivity=True)

    assert signals.thesis_tone_restraint_score >= 0.85
    assert signals.evidence_fidelity_score >= 0.85


def test_we_subject_not_introduced_without_source_support() -> None:
    backend = RuleBasedRewriteBackend()
    repaired = backend._replace_we_with_original_subject_style(
        sentence="我们让模型直接完成特征融合。",
        original_sentence="模型直接完成特征融合。",
        role="mechanism",
    )

    assert repaired.startswith("模型")
    assert "我们" not in repaired


def test_v9_reduces_externalized_commentary_on_real_test_md(tmp_path: Path) -> None:
    if not REAL_TEST_MD.exists() or not REAL_V7_MD.exists():
        pytest.skip("real test.md or v7 output is not available in this checkout")

    output = tmp_path / "test_airc_v9.md"
    report = tmp_path / "test_airc_v9.report.json"
    result = run_file(REAL_TEST_MD, output_path=output, report_path=report)
    v9_text = output.read_text(encoding="utf-8")
    aggregate = aggregate_evidence_fidelity(result.rewrite_result.rewrite_report.rewrite_stats)

    assert result.output_written is True
    assert aggregate["evidence_fidelity_preserved"] is True
    assert aggregate["unsupported_expansion_controlled"] is True
    assert aggregate["thesis_tone_restrained"] is True
    assert aggregate["metaphor_or_storytelling_controlled"] is True
    assert aggregate["authorial_claim_risk_controlled"] is True
    assert _count_externalized_commentary(v9_text) <= _count_externalized_commentary(
        REAL_V7_MD.read_text(encoding="utf-8")
    )
