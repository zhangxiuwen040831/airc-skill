from __future__ import annotations

import re
from pathlib import Path

import pytest

from airc_skill.academic_sentence_naturalization import analyze_academic_sentence_naturalization
from airc_skill.pipeline import run_file
from airc_skill.rewriter import RuleBasedRewriteBackend, split_sentences


ROOT = Path(__file__).resolve().parents[1]
REAL_TEST_MD = ROOT / "test.md"
REAL_V9_MD = ROOT / "test_airc_v9.md"
REAL_V10_MD = ROOT / "test_airc_v10.md"


def _project_style_count(text: str) -> int:
    patterns = (
        r"本研究面向[^。；，,]{2,36}需求",
        r"本研究的主题为",
        r"在方法上",
        r"围绕这一目标",
        r"在解决[^。；]{2,60}目标下",
        r"以实现[^。；]{1,36}落地",
        r"旨在构建",
        r"形成完整闭环",
        r"具有显著[^。；]{0,24}价值",
        r"并不是[^。；]{2,80}而是",
        r"在结构上[^。；]{2,80}在决策层面",
    )
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


def _author_style_risk_count(text: str) -> int:
    patterns = (
        r"在[^，。；]{2,28}(?:中|下|过程中)[，,]",
        r"不仅[^。；]{2,80}(?:还|也|同时)",
        r"旨在",
        r"以实现",
        r"从而实现",
        r"进行(?:评估|分析|验证|控制|处理)",
        r"实现(?:控制|部署|融合|落地)",
        r"被视为",
        r"得到体现",
    )
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


def test_bureaucratic_opening_density_detects_project_style_sentence() -> None:
    sentences = split_sentences("本研究面向平台部署需求，构建AIGC图像检测系统。系统保留主要判别路径。")

    signals = analyze_academic_sentence_naturalization(sentences, paragraph_index=3)

    assert signals.bureaucratic_opening_density > 0
    assert signals.bureaucratic_opening_controlled is False
    assert signals.bureaucratic_paragraph_ids == [3]


def test_repeated_explicit_subject_risk_detects_mechanical_subject_chain() -> None:
    sentences = split_sentences("本研究构建检测模型。本研究引入频域分支。本研究保留语义主路径。")

    signals = analyze_academic_sentence_naturalization(sentences)

    assert signals.repeated_explicit_subject_risk > 0.2
    assert signals.explicit_subject_chain_controlled is False


def test_overstructured_syntax_risk_detects_template_parallelism() -> None:
    sentences = split_sentences("该设计并不是简单增加模块，而是重构判别路径。系统随后输出结果。")

    signals = analyze_academic_sentence_naturalization(sentences)

    assert signals.overstructured_syntax_risk > 0
    assert signals.overstructured_syntax_controlled is False


def test_delayed_main_clause_risk_detects_head_heavy_sentence() -> None:
    sentences = split_sentences("在解决跨域部署中频域伪迹波动会影响系统稳定性这一目标下，本研究提出双分支检测框架。")

    signals = analyze_academic_sentence_naturalization(sentences)

    assert signals.delayed_main_clause_risk > 0
    assert signals.main_clause_position_reasonable is False


def test_v10_reduces_project_style_openings_on_real_test_md(tmp_path: Path) -> None:
    if not REAL_TEST_MD.exists() or not REAL_V9_MD.exists():
        pytest.skip("real test.md or v9 output is not available in this checkout")

    output = tmp_path / "test_airc_v10.md"
    report = tmp_path / "test_airc_v10.report.json"
    result = run_file(REAL_TEST_MD, output_path=output, report_path=report)
    v10_text = output.read_text(encoding="utf-8")
    v9_text = REAL_V9_MD.read_text(encoding="utf-8")

    assert result.output_written is True
    assert result.rewrite_result.review.bureaucratic_opening_controlled is True
    assert result.rewrite_result.review.overstructured_syntax_controlled is True
    assert _project_style_count(v10_text) <= _project_style_count(v9_text)


def test_academic_sentence_naturalizer_removes_common_project_style_shapes() -> None:
    backend = RuleBasedRewriteBackend()
    original = "本研究面向平台部署需求，设计统一检测接口。"
    rewritten_sentences, rules, _notes, _marks = backend._apply_academic_sentence_naturalization(
        original_sentences=[original],
        rewritten_sentences=["本研究面向平台部署需求，设计统一检测接口。"],
        rewrite_depth="developmental_rewrite",
        high_sensitivity_prose=True,
    )
    rewritten = "".join(rewritten_sentences)

    assert "面向平台部署需求" not in rewritten
    assert "local:remove-bureaucratic-opening" in rules


def test_directness_score_penalizes_wrapped_introductory_phrasing() -> None:
    sentences = split_sentences("在实际部署过程中，本研究对模型进行评估，并且从而实现控制。")

    signals = analyze_academic_sentence_naturalization(sentences)

    assert signals.directness_score < 0.75
    assert signals.connector_overuse_risk > 0
    assert signals.nominalization_density > 0
    assert signals.author_style_alignment_controlled is False


def test_passive_voice_and_overlong_sentence_risks_are_reported() -> None:
    sentence = (
        "在复杂部署场景中，本研究提出的检测流程被视为系统稳定运行的重要基础，"
        "并且通过多轮验证实现控制，从而实现对前端、后端、模型推理和解释输出等多个环节的统一管理，"
        "同时继续保留阈值策略、可视化结果和异常样本分析之间的对应关系。"
    )

    signals = analyze_academic_sentence_naturalization([sentence])

    assert signals.passive_voice_ratio > 0
    assert signals.overlong_sentence_risk > 0


def test_v11_author_style_layer_reduces_wrapping_on_real_test_md(tmp_path: Path) -> None:
    if not REAL_TEST_MD.exists() or not REAL_V10_MD.exists():
        pytest.skip("real test.md or v10 output is not available in this checkout")

    output = tmp_path / "test_airc_v11.md"
    report = tmp_path / "test_airc_v11.report.json"
    result = run_file(REAL_TEST_MD, output_path=output, report_path=report)
    v11_text = output.read_text(encoding="utf-8")
    v10_text = REAL_V10_MD.read_text(encoding="utf-8")
    metrics = result.rewrite_result.review.academic_sentence_naturalization

    assert result.output_written is True
    assert result.rewrite_result.review.body_rewrite_coverage >= 0.60
    assert result.rewrite_result.review.evidence_fidelity_preserved is True
    assert metrics["directness_score"] >= 0.88
    assert metrics["author_style_alignment_controlled"] is True
    assert _author_style_risk_count(v11_text) <= _author_style_risk_count(v10_text)
