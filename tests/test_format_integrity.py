from pathlib import Path

from airc_skill.config import RewriteMode
from airc_skill.core_guard import compare_core_snapshots, extract_caption_spans, snapshot_core_content
from airc_skill.pipeline import rewrite_file
from airc_skill.rewriter import Rewriter


def _build_format_sample() -> str:
    return """# 系统概述

## Design and Implementation of an AI-Generated Image Detection Tool

## Abstract

This study designs a practical AIGC detection workflow. Stable Diffusion is evaluated on the NTIRE dataset.

本研究围绕真实场景中的检测需求展开，不仅完成模型训练流程设计，还补充了前后端联动方案。因此，本研究进一步讨论系统部署阶段的阈值选择。

如图5-1所示：
![img](file:///C:/demo/system.png)
图5-1 系统分层架构图

本研究最终采用 checkpoints/best.pth 作为本地部署模型，对应 V10 Phase2 epoch_008，默认阈值为 0.35[1][2]。
"""


def _rewrite_sample(tmp_path: Path, name: str = "format.md"):
    source = tmp_path / name
    original = _build_format_sample()
    source.write_text(original, encoding="utf-8")
    result = rewrite_file(source, mode=RewriteMode.BALANCED)
    comparison = compare_core_snapshots(
        snapshot_core_content(original, ".md"),
        snapshot_core_content(result.text, ".md"),
    )
    return original, result, comparison


def test_english_heading_spacing_preserved(tmp_path: Path) -> None:
    _, result, comparison = _rewrite_sample(tmp_path, "heading.md")

    assert "## Design and Implementation of an AI-Generated Image Detection Tool" in result.text
    assert "## DesignandImplementationofanAI-GeneratedImageDetectionTool" not in result.text
    assert comparison["heading_format_integrity_check"] is True


def test_english_abstract_no_double_period(tmp_path: Path) -> None:
    original, result, comparison = _rewrite_sample(tmp_path, "abstract.md")

    assert "This study designs a practical AIGC detection workflow." in result.text
    assert ".." not in result.text
    assert snapshot_core_content(original, ".md").english_blocks == snapshot_core_content(result.text, ".md").english_blocks
    assert comparison["english_spacing_integrity_check"] is True


def test_caption_colon_not_changed_to_colon_period(tmp_path: Path) -> None:
    _, result, comparison = _rewrite_sample(tmp_path, "caption.md")

    assert "如图5-1所示：" in result.text
    assert "图5-1 系统分层架构图" in result.text
    assert "：。" not in result.text
    assert comparison["caption_punctuation_integrity_check"] is True


def test_caption_punctuation_only_checks_caption_spans() -> None:
    original = "如图5-1所示：\n\n图5-1 系统分层架构图\n\n正文可以自然改写。"
    revised = "如图5-1所示：\n\n图5-1 系统分层架构图\n\n正文可以自然改写：并且不会影响图注。"

    assert extract_caption_spans(original) == extract_caption_spans(revised)
    assert compare_core_snapshots(
        snapshot_core_content(original, ".md"),
        snapshot_core_content(revised, ".md"),
    )["caption_punctuation_integrity_check"]


def test_image_placeholder_integrity_preserved(tmp_path: Path) -> None:
    original, result, comparison = _rewrite_sample(tmp_path, "placeholder.md")

    placeholder = "![img](file:///C:/demo/system.png)"
    assert result.text.count(placeholder) == 1
    assert snapshot_core_content(original, ".md").placeholders == snapshot_core_content(result.text, ".md").placeholders
    assert comparison["placeholder_integrity_check"] is True


def test_no_extra_trailing_spaces_or_broken_linebreaks(tmp_path: Path) -> None:
    _, result, comparison = _rewrite_sample(tmp_path, "spacing.md")

    assert all(line == line.rstrip(" \t") for line in result.text.splitlines())
    assert comparison["linebreak_whitespace_integrity_check"] is True


def test_markdown_headings_not_modified(tmp_path: Path) -> None:
    original, result, comparison = _rewrite_sample(tmp_path, "headings.md")

    assert snapshot_core_content(original, ".md").headings == snapshot_core_content(result.text, ".md").headings
    assert comparison["title_integrity_check"] is True


def test_numbers_and_paths_are_preserved(tmp_path: Path) -> None:
    original, result, comparison = _rewrite_sample(tmp_path, "paths.md")

    for marker in ("0.35", "epoch_008", "checkpoints/best.pth"):
        assert marker in result.text
    assert snapshot_core_content(original, ".md").paths == snapshot_core_content(result.text, ".md").paths
    assert comparison["numeric_integrity_check"] is True
    assert comparison["path_integrity_check"] is True


def test_checkpoint_reference_is_preserved(tmp_path: Path) -> None:
    _, result, _ = _rewrite_sample(tmp_path, "checkpoint.md")

    assert "本研究最终采用 checkpoints/best.pth 作为本地部署模型" in result.text


def test_benyan_subject_not_replaced_when_reference_would_weaken() -> None:
    text = (
        "本研究最终采用 checkpoints/best.pth 作为本地部署模型，对应 V10 Phase2 epoch_008。"
        "本研究同时完成了前端交互系统设计。"
    )

    rewritten, stats = Rewriter().rewrite(text, mode=RewriteMode.BALANCED)

    assert "本研究最终采用 checkpoints/best.pth 作为本地部署模型" in rewritten
    assert stats.structural_action_count >= 0


def test_real_test_md_format_integrity_is_required_for_write(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parent / "fixtures" / "user_test.md"
    original = fixture.read_text(encoding="utf-8")
    source = tmp_path / "user_test.md"
    source.write_text(original, encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.BALANCED)

    assert "：。" not in result.text
    assert result.review.integrity_checks["placeholder_integrity_check"] is True
    assert result.review.integrity_checks["caption_punctuation_integrity_check"] is True
    assert result.review.integrity_checks["english_spacing_integrity_check"] is True
    assert "本研究最终采用 checkpoints/best.pth" in result.text
    if result.output_written:
        assert result.review.format_integrity_ok is True
        assert result.review.core_content_integrity_ok is True
