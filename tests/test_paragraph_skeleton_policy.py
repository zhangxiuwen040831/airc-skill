from __future__ import annotations

import json
from pathlib import Path

import pytest

from airc_skill.config import RewriteMode
from airc_skill.guidance import guide_document_text
from airc_skill.paragraph_skeleton import (
    analyze_paragraph_skeleton,
    is_dangling_opening_sentence,
    paragraph_skeleton_checks,
    semantic_topic_anchor_match,
)
from airc_skill.pipeline import run_file
from airc_skill.rewriter import Rewriter, split_sentences


ROOT = Path(__file__).resolve().parents[1]
REAL_TEST_MD = ROOT / "test.md"


def _first_sentence(text: str) -> str:
    return (split_sentences(text.strip()) or [text.strip()])[0]


def _dangling_opening_count(text: str) -> int:
    count = 0
    for paragraph in [part.strip() for part in text.split("\n\n") if part.strip()]:
        first = _first_sentence(paragraph)
        if is_dangling_opening_sentence(first):
            count += 1
    return count


def test_topic_sentence_kept_at_paragraph_start() -> None:
    original = (
        "本研究围绕AIGC图像检测任务提出多分支判别框架。"
        "该框架将语义线索和频域线索结合起来。"
        "同时，系统需要保持可解释性与部署稳定性。"
        "因此，段落应先交代研究对象，再展开方法细节。"
    )

    revised, stats = Rewriter().rewrite(
        original,
        mode=RewriteMode.STRONG,
        rewrite_depth="developmental_rewrite",
        rewrite_intensity="high",
    )
    checks = paragraph_skeleton_checks(original, revised)
    first = _first_sentence(revised)

    assert revised != original
    assert stats.structural_action_count > 0
    assert checks["paragraph_topic_sentence_preserved"] is True
    assert checks["topic_sentence_not_demoted_to_mid_paragraph"] is True
    assert first.startswith("本研究")
    assert not first.startswith(("并进一步", "围绕这一点", "在这种情况下"))


def test_opening_reorder_blocked_for_valid_topic_sentence() -> None:
    original = (
        "在方法上，本研究构建了面向AIGC图像检测的双分支识别框架。"
        "并进一步提出频域约束以补充语义判别。"
        "该设计能够缓解单一路径在复杂场景中的不稳定。"
        "因此，模型结构需要保持清晰的先后论述。"
    )

    revised, stats = Rewriter().rewrite(
        original,
        mode=RewriteMode.STRONG,
        rewrite_depth="developmental_rewrite",
        rewrite_intensity="high",
    )
    first = _first_sentence(revised)
    checks = paragraph_skeleton_checks(original, revised)

    assert "paragraph:high-intensity-promote-second" not in stats.applied_rules
    assert first.startswith("在方法上") or "本研究构建" in first
    assert not first.startswith("并进一步")
    assert checks["paragraph_skeleton_consistent"] is True


def test_no_dangling_transition_style_opening() -> None:
    original = (
        "在方法上，本研究构建了面向AIGC图像检测的双分支识别框架。"
        "并进一步提出频域约束以补充语义判别。"
    )
    bad_revised = (
        "并进一步提出频域约束以补充语义判别。"
        "在方法上，本研究构建了面向AIGC图像检测的双分支识别框架。"
    )

    checks = paragraph_skeleton_checks(original, bad_revised)

    assert checks["paragraph_opening_style_valid"] is False
    assert checks["no_dangling_opening_sentence"] is False
    assert checks["topic_sentence_not_demoted_to_mid_paragraph"] is False


def test_paragraph_skeleton_preserved_after_cluster_rewrite() -> None:
    original = (
        "近年来，生成式图像模型正在改变内容生产方式。"
        "真实图像与生成图像之间的边界因此变得模糊。"
        "检测任务需要同时处理语义差异和频域痕迹。"
        "因此，论文段落应保持先立题、再展开、后收束的骨架。"
    )

    revised, stats = Rewriter().rewrite(
        original,
        mode=RewriteMode.STRONG,
        rewrite_depth="developmental_rewrite",
        rewrite_intensity="high",
    )
    skeleton = analyze_paragraph_skeleton(revised)
    checks = paragraph_skeleton_checks(original, revised)

    assert stats.cluster_changes > 0
    assert skeleton.role_names[0] == "topic_sentence"
    assert checks["paragraph_skeleton_consistent"] is True


def test_paragraph_topic_preservation_allows_surface_rewrite() -> None:
    original = (
        "本研究围绕AIGC图像检测任务构建多分支判别框架。"
        "该框架进一步结合语义线索和频域线索。"
    )
    revised = (
        "AIGC图像检测任务中，多分支判别框架仍然是本研究的主要设计对象。"
        "该框架会结合语义线索和频域线索。"
    )

    checks = paragraph_skeleton_checks(original, revised)

    assert semantic_topic_anchor_match(_first_sentence(original), _first_sentence(revised))
    assert checks["paragraph_topic_sentence_preserved"] is True
    assert checks["paragraph_skeleton_consistent"] is True


def test_real_test_md_openings_are_more_stable_in_v2(tmp_path: Path) -> None:
    if not REAL_TEST_MD.exists():
        pytest.skip("real test.md is not available in this checkout")
    output = tmp_path / "test_airc_v2.md"
    report = tmp_path / "test_airc_v2.report.json"

    result = run_file(REAL_TEST_MD, output_path=output, report_path=report)
    payload = json.loads(report.read_text(encoding="utf-8"))
    revised = output.read_text(encoding="utf-8")

    assert result.output_written is True
    assert result.rewrite_result.write_gate.write_allowed is True
    assert payload["review"]["paragraph_topic_sentence_preserved"] is True
    assert payload["review"]["paragraph_opening_style_valid"] is True
    assert payload["review"]["paragraph_skeleton_consistent"] is True
    assert _dangling_opening_count(revised) == 0


def test_english_blocks_not_included_in_natural_revision_rewrite() -> None:
    text = (
        "# ResearchonAIGCImageDetectionandMultiBranchDecisionConstraints\n\n"
        "AbstractThispaperinvestigatesAIgeneratedimagedetectionwithsemanticandfrequencyfeatures.\n\n"
        "## 摘要\n\n"
        "本研究围绕AIGC图像检测任务提出多分支判别框架。该框架强调语义线索与频域线索的协同。"
    )

    guidance = guide_document_text(text, metadata={"suffix": ".md"})
    english_blocks = [block for block in guidance.block_policies if block.block_type == "english_block"]
    body_blocks = [block for block in guidance.block_policies if block.should_rewrite]

    assert english_blocks
    assert all(block.edit_policy == "do_not_touch" for block in english_blocks)
    assert all(block.should_rewrite is False for block in english_blocks)
    assert all("ResearchonAIGC" not in block.original_text for block in body_blocks)
