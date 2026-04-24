from __future__ import annotations

import json
from pathlib import Path

import pytest

from airc_skill.config import RewriteMode
from airc_skill.guidance import guide_document_text
from airc_skill.local_revision_realism import analyze_local_revision_text
from airc_skill.pipeline import run_file
from airc_skill.rewriter import Rewriter
from airc_skill.sentence_readability import (
    analyze_paragraph_readability_text,
    dangling_sentence_risk,
    fragment_like_conclusion_sentence,
    sentence_completeness_score,
    split_sentences_for_readability,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_TEST_MD = ROOT / "test.md"
REAL_V3_MD = ROOT / "test_airc_v3.md"


def _fragment_like_sentence_count(text: str) -> int:
    count = 0
    for sentence in split_sentences_for_readability(text):
        if (
            dangling_sentence_risk(sentence)
            or sentence_completeness_score(sentence, role="support_sentence") < 0.52
            or fragment_like_conclusion_sentence(sentence, is_final=False, role="support_sentence")
        ):
            count += 1
    return count


def test_sentence_completeness_score_flags_fragment_like_sentence() -> None:
    fragment = "是比单纯增加模型复杂度更有效的路径。"
    complete = "本研究表明，特征约束是比单纯增加模型复杂度更有效的路径。"

    assert sentence_completeness_score(fragment, role="conclusion_sentence") < 0.5
    assert sentence_completeness_score(complete, role="conclusion_sentence") > 0.75


def test_dangling_sentence_risk_detects_transition_fragment() -> None:
    assert dangling_sentence_risk("并进一步提出多分支检测策略。") is True
    assert dangling_sentence_risk("在这种情况下，模型训练完成之后。") is True
    assert dangling_sentence_risk("本研究进一步提出多分支检测策略。") is False


def test_formula_and_list_leads_are_not_fragment_like_sentences() -> None:
    formula_lead = "该组合损失函数的核心表达式如下："
    list_lead = "策略具体包含以下三个部分："

    assert dangling_sentence_risk(formula_lead) is False
    assert sentence_completeness_score(formula_lead, role="support_sentence") > 0.8
    assert sentence_completeness_score(list_lead, role="support_sentence") > 0.8
    assert fragment_like_conclusion_sentence(formula_lead, is_final=False, role="support_sentence") is False


def test_readability_repair_rewrites_incomplete_support_sentence() -> None:
    backend = Rewriter().backend
    original = ["本研究提出多分支检测策略。", "该策略能够增强边界样本识别。"]
    rewritten = ["本研究提出多分支检测策略。", "并进一步增强边界样本识别。"]

    repaired, rules, _notes, marks = backend._readability_repair_pass(
        original_sentences=original,
        rewritten_sentences=rewritten,
        rewrite_depth="developmental_rewrite",
        mode=RewriteMode.STRONG,
        high_sensitivity_prose=False,
    )

    assert repaired[1] != rewritten[1]
    assert dangling_sentence_risk(repaired[1]) is False
    assert "readability:sentence-completeness-repair" in rules
    assert "sentence_completeness_repair" in marks


def test_readability_repair_softens_complete_conclusion_transition() -> None:
    backend = Rewriter().backend
    original = ["本研究提出多分支检测策略。", "因此，本研究能够提升部署稳定性。"]
    rewritten = ["本研究提出多分支检测策略。", "因此，本研究能够提升部署稳定性。"]

    repaired, rules, _notes, _marks = backend._readability_repair_pass(
        original_sentences=original,
        rewritten_sentences=rewritten,
        rewrite_depth="developmental_rewrite",
        mode=RewriteMode.STRONG,
        high_sensitivity_prose=True,
    )

    assert repaired[-1].startswith("本研究")
    assert "readability:soften-complete-transition" in rules
    assert fragment_like_conclusion_sentence(repaired[-1], is_final=True, role="conclusion_sentence") is False


def test_readability_repair_splits_collapsed_enumeration_flow() -> None:
    backend = Rewriter().backend
    sentence = "若某一分支的学习速度更快，融合层会偏向该分支，第一，分支贡献易失衡，进而第二，门控机制可能学到错误偏置。"

    repaired, rules, _notes, _marks = backend._readability_repair_pass(
        original_sentences=[sentence],
        rewritten_sentences=[sentence],
        rewrite_depth="developmental_rewrite",
        mode=RewriteMode.STRONG,
        high_sensitivity_prose=False,
    )

    assert len(repaired) == 2
    assert repaired[0].startswith("若某一分支")
    assert repaired[1].startswith("第二")
    assert "readability:repair-enumeration-flow" in rules


def test_readability_repair_removes_generated_discourse_marker() -> None:
    backend = Rewriter().backend
    original = ["实验采用分层评估思路。", "该设置用于验证模型在标准条件下的性能。"]
    rewritten = ["实验采用分层评估思路。", "该段论述用于验证模型在标准条件下的性能。"]

    repaired, rules, _notes, _marks = backend._readability_repair_pass(
        original_sentences=original,
        rewritten_sentences=rewritten,
        rewrite_depth="developmental_rewrite",
        mode=RewriteMode.STRONG,
        high_sensitivity_prose=True,
    )

    assert "该段论述" not in "".join(repaired)
    assert "用于验证模型在标准条件下的性能" in "".join(repaired)
    assert "该机制用于" not in "".join(repaired)
    assert "readability:repair-generated-discourse-marker" in rules


def test_high_sensitivity_prose_blocks_require_readability() -> None:
    text = (
        "## 研究意义\n\n"
        "本研究的意义在于提升AIGC图像检测结果的可解释性。该目标能够支撑后续系统落地。"
    )
    guidance = guide_document_text(text, metadata={"suffix": ".md"})
    sensitive_blocks = [block for block in guidance.block_policies if block.high_sensitivity_prose]
    backend = Rewriter().backend
    original = ["本研究的意义在于提升AIGC图像检测结果的可解释性。", "该目标能够支撑后续系统落地。"]
    rewritten = ["本研究的意义在于提升AIGC图像检测结果的可解释性。", "并进一步支撑后续系统落地。"]

    repaired, rules, _notes, _marks = backend._readability_repair_pass(
        original_sentences=original,
        rewritten_sentences=rewritten,
        rewrite_depth="developmental_rewrite",
        mode=RewriteMode.STRONG,
        high_sensitivity_prose=True,
    )
    signals = analyze_paragraph_readability_text("".join(repaired), high_sensitivity=True)

    assert sensitive_blocks
    assert all(block.high_sensitivity_prose for block in sensitive_blocks)
    assert "readability:high-sensitivity-readability-repair" in rules
    assert signals.dangling_sentence_risk == 0
    assert signals.paragraph_readability_score >= 0.72


def test_v4_reduces_fragment_like_sentences_on_real_test_md(tmp_path: Path) -> None:
    if not REAL_TEST_MD.exists() or not REAL_V3_MD.exists():
        pytest.skip("real test.md or v3 output is not available in this checkout")
    output = tmp_path / "test_airc_v4.md"
    report = tmp_path / "test_airc_v4.report.json"

    result = run_file(REAL_TEST_MD, output_path=output, report_path=report)
    payload = json.loads(report.read_text(encoding="utf-8"))
    v3_count = _fragment_like_sentence_count(REAL_V3_MD.read_text(encoding="utf-8"))
    v4_count = _fragment_like_sentence_count(output.read_text(encoding="utf-8"))

    assert result.output_written is True
    assert payload["review"]["sentence_completeness_preserved"] is True
    assert payload["review"]["paragraph_readability_preserved"] is True
    assert v4_count <= v3_count


def test_realism_score_not_improved_by_incomplete_sentence() -> None:
    complete = analyze_local_revision_text("本研究提出多分支策略。该策略能够增强边界识别。")
    incomplete = analyze_local_revision_text("本研究提出多分支策略。并进一步增强边界识别。")

    assert incomplete.revision_realism_score <= complete.revision_realism_score
