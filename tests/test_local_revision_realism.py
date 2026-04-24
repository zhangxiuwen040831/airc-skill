from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from airc_skill.config import RewriteMode
from airc_skill.local_revision_realism import (
    aggregate_local_revision_realism,
    analyze_local_revision_sentences,
    analyze_local_revision_text,
)
from airc_skill.paragraph_skeleton import paragraph_skeleton_checks
from airc_skill.pipeline import run_file
from airc_skill.reviewer import review_rewrite
from airc_skill.rewriter import Rewriter, split_sentences


ROOT = Path(__file__).resolve().parents[1]
REAL_TEST_MD = ROOT / "test.md"
REAL_V2_MD = ROOT / "test_airc_v2.md"
REAL_V5_MD = ROOT / "test_airc_v5.md"


def _body_paragraph_realism_score(text: str) -> float:
    scores: list[float] = []
    for paragraph in [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]:
        if paragraph.startswith(("#", "图", "表", "[", "!", "|")):
            continue
        signals = analyze_local_revision_text(paragraph)
        if signals.sentence_count >= 2:
            scores.append(signals.revision_realism_score)
    return sum(scores) / max(1, len(scores))


def _average_body_signal(text: str, attr: str) -> float:
    scores: list[float] = []
    for paragraph in [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]:
        if paragraph.startswith(("#", "图", "表", "[", "!", "|")):
            continue
        signals = analyze_local_revision_text(paragraph)
        if signals.sentence_count >= 2:
            scores.append(float(getattr(signals, attr)))
    return sum(scores) / max(1, len(scores))


def test_local_discourse_not_flat_in_rewritten_body() -> None:
    original = (
        "本研究面向AIGC图像检测任务构建检测框架。"
        "同时，系统能够融合语义信息。"
        "此外，系统能够融合频域信息。"
        "因此，系统能够提升检测稳定性。"
    )

    revised, stats = Rewriter().rewrite(
        original,
        mode=RewriteMode.STRONG,
        rewrite_depth="developmental_rewrite",
        rewrite_intensity="high",
    )
    review = review_rewrite(original, revised, mode=RewriteMode.BALANCED, rewrite_stats=[stats])

    assert review.local_discourse_not_flat is True
    assert stats.local_revision_actions
    assert "reduce_sentence_uniformity" in stats.local_revision_actions or "soften_overexplicit_transition" in stats.local_revision_actions


def test_sentence_uniformity_reduced_after_revision() -> None:
    original = (
        "本研究围绕检测任务构建模型。"
        "该模型能够处理语义线索。"
        "该模型能够处理频域线索。"
        "该模型能够输出检测结果。"
    )

    revised, stats = Rewriter().rewrite(
        original,
        mode=RewriteMode.STRONG,
        rewrite_depth="developmental_rewrite",
        rewrite_intensity="high",
    )
    original_signals = analyze_local_revision_sentences(split_sentences(original))
    revised_signals = analyze_local_revision_sentences(split_sentences(revised))
    aggregate = aggregate_local_revision_realism([stats])

    assert aggregate["sentence_uniformity_reduced"] is True
    assert revised_signals.local_discourse_flatness <= original_signals.local_discourse_flatness
    assert stats.sentence_cadence_irregularity >= 0.0


def test_revision_realism_score_present_for_body_blocks(tmp_path: Path) -> None:
    if not REAL_TEST_MD.exists():
        pytest.skip("real test.md is not available in this checkout")
    output = tmp_path / "test_airc_v3.md"
    report = tmp_path / "test_airc_v3.report.json"

    result = run_file(REAL_TEST_MD, output_path=output, report_path=report)
    payload = json.loads(report.read_text(encoding="utf-8"))
    realism = payload["review"]["local_revision_realism"]

    assert result.output_written is True
    assert realism["paragraphs_checked"] > 0
    assert realism["revision_realism_score"] > 0
    assert payload["review"]["revision_realism_present"] is True


def test_long_document_v3_more_locally_natural_than_v2(tmp_path: Path) -> None:
    if not REAL_TEST_MD.exists() or not REAL_V2_MD.exists():
        pytest.skip("real test.md or v2 output is not available in this checkout")
    output = tmp_path / "test_airc_v3.md"
    report = tmp_path / "test_airc_v3.report.json"

    result = run_file(REAL_TEST_MD, output_path=output, report_path=report)
    v2_score = _body_paragraph_realism_score(REAL_V2_MD.read_text(encoding="utf-8"))
    v3_score = _body_paragraph_realism_score(output.read_text(encoding="utf-8"))

    assert result.output_written is True
    assert v3_score >= v2_score


def test_topic_sentence_preservation_not_broken_by_local_realism_layer() -> None:
    original = (
        "本研究围绕AIGC图像检测任务提出多分支判别框架。"
        "同时，该框架能够整合语义线索。"
        "此外，该框架能够补充频域证据。"
        "因此，该框架能够增强开放场景中的判别稳定性。"
    )

    revised, stats = Rewriter().rewrite(
        original,
        mode=RewriteMode.STRONG,
        rewrite_depth="developmental_rewrite",
        rewrite_intensity="high",
    )
    checks = paragraph_skeleton_checks(original, revised)

    assert stats.local_revision_actions
    assert checks["paragraph_topic_sentence_preserved"] is True
    assert checks["topic_sentence_not_demoted_to_mid_paragraph"] is True
    assert split_sentences(revised)[0].startswith("本研究")


def test_uniformity_penalty_detects_overpolished_paragraph() -> None:
    paragraph = (
        "本研究具有重要意义。"
        "该系统具有重要意义。"
        "该设计具有重要意义。"
        "该策略具有重要意义。"
    )

    signals = analyze_local_revision_text(paragraph)

    assert signals.stylistic_uniformity_score > 0.2
    assert signals.academic_cliche_density > 0.15


def test_texture_variation_present_after_rewrite() -> None:
    original = (
        "本研究围绕检测任务构建整体方案。"
        "该方案能够整合语义线索。"
        "该方案能够整合频域线索。"
        "该方案能够稳定输出判定结果。"
    )

    revised, stats = Rewriter().rewrite(
        original,
        mode=RewriteMode.STRONG,
        rewrite_depth="developmental_rewrite",
        rewrite_intensity="high",
    )
    realism = aggregate_local_revision_realism([stats])

    assert revised != original
    assert realism["support_sentence_texture_variation"] >= 0.30
    assert realism["support_sentence_texture_varied"] is True


def test_cliche_density_reduced_in_high_sensitivity_blocks() -> None:
    original = (
        "总结而言，本研究的核心价值在于通过问题诊断与策略迭代提高系统稳定性。"
        "项目最终实现了更稳定的部署表现，并为后续扩展提供了基础。"
    )

    revised, _ = Rewriter().rewrite(
        original,
        mode=RewriteMode.STRONG,
        rewrite_depth="developmental_rewrite",
        rewrite_intensity="high",
        high_sensitivity_prose=True,
    )
    original_signals = analyze_local_revision_text(original)
    revised_signals = analyze_local_revision_text(revised)

    assert revised_signals.academic_cliche_density <= original_signals.academic_cliche_density
    assert "核心价值在于" not in revised or "价值主要体现在" in revised
    assert "实现了更稳定的部署表现" not in revised


def test_v6_less_uniform_than_v5(tmp_path: Path) -> None:
    if not REAL_TEST_MD.exists() or not REAL_V5_MD.exists():
        pytest.skip("real test.md or v5 output is not available in this checkout")
    output = tmp_path / "test_airc_v6.md"
    report = tmp_path / "test_airc_v6.report.json"

    result = run_file(REAL_TEST_MD, output_path=output, report_path=report)
    v5_text = REAL_V5_MD.read_text(encoding="utf-8")
    v6_text = output.read_text(encoding="utf-8")

    assert result.output_written is True
    assert _average_body_signal(v6_text, "stylistic_uniformity_score") <= _average_body_signal(
        v5_text,
        "stylistic_uniformity_score",
    )
    assert _average_body_signal(v6_text, "support_sentence_texture_variation") >= _average_body_signal(
        v5_text,
        "support_sentence_texture_variation",
    )


def test_partial_retention_preserves_human_style() -> None:
    rewriter = Rewriter()
    backend = rewriter.backend
    original_sentences = [
        "本研究围绕检测任务构建整体方案。",
        "系统同时整合语义线索，以支撑后续判定。",
        "系统通过语义线索支撑后续判定。",
        "系统进一步输出最终判定结果。",
    ]
    polished_sentences = [
        "本研究围绕检测任务构建整体方案。",
        "系统通过吸收语义线索来维持判定稳定性。",
        "系统通过整合语义线索来补充分支判断。",
        "系统进一步输出最终判定结果。",
    ]

    updated, rules, _ = backend._introduce_sentence_asymmetry(
        sentences=polished_sentences,
        original_sentences=original_sentences,
        paragraph_index=2,
        rewrite_intensity="high",
        high_sensitivity_prose=False,
    )

    assert "local:soft-keep-for-human-revision-feel" in rules
    assert updated != polished_sentences
    assert any(
        updated[index] == backend._ensure_sentence_end(backend._soften_overexplicit_transition(original_sentences[index]))
        for index in range(len(updated))
    )
