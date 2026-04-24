from pathlib import Path

from airc_skill.core_guard import (
    compare_core_snapshots,
    compare_protected_segments,
    normalize_numeric_tokens_for_integrity,
    snapshot_core_content,
)
from airc_skill.pipeline import rewrite_file
from airc_skill.config import RewriteMode


def _build_core_sample() -> str:
    return """# 方法概述

近年来，AIGC 图像检测系统在内容安全场景中受到广泛关注[1][2]。与此同时，相关讨论也从单一识别扩展到流程协同。

本研究的主题为面向内容安全的检测流程组织方式。本研究不仅结合 Stable Diffusion、CLIP ViT 与 NTIRE 数据集进行分析，还保持样本量为 128、默认阈值为 0.35、实验年份为 2023 的实验设定。因此，本研究进一步讨论人工复核与模型判断之间的协同关系。

$$
F_1 = \\frac{2PR}{P+R} \\tag{1}
$$

相关结论与 (Smith, 2023) 保持一致。
"""


def test_titles_are_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "core.md"
    original = _build_core_sample()
    source.write_text(original, encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.BALANCED)

    assert "# 方法概述" in result.text
    assert snapshot_core_content(original, ".md").headings == snapshot_core_content(result.text, ".md").headings


def test_formulas_are_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "formula.md"
    original = _build_core_sample()
    source.write_text(original, encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.BALANCED)

    assert "\\tag{1}" in result.text
    assert snapshot_core_content(original, ".md").formulas == snapshot_core_content(result.text, ".md").formulas


def test_technical_terms_are_preserved(tmp_path: Path) -> None:
    source = tmp_path / "terms.md"
    original = _build_core_sample()
    source.write_text(original, encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.BALANCED)
    revised_snapshot = snapshot_core_content(result.text, ".md")

    for term in ("AIGC", "Stable Diffusion", "CLIP ViT", "NTIRE"):
        assert term in result.text
    assert compare_core_snapshots(snapshot_core_content(original, ".md"), revised_snapshot)["terminology_integrity_check"]


def test_citations_are_preserved(tmp_path: Path) -> None:
    source = tmp_path / "citations.md"
    original = _build_core_sample()
    source.write_text(original, encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.BALANCED)

    assert "[1][2]" in result.text
    assert "(Smith, 2023)" in result.text
    assert compare_core_snapshots(
        snapshot_core_content(original, ".md"),
        snapshot_core_content(result.text, ".md"),
    )["citation_integrity_check"]


def test_numbers_and_years_are_preserved(tmp_path: Path) -> None:
    source = tmp_path / "numbers.md"
    original = _build_core_sample()
    source.write_text(original, encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.BALANCED)

    for item in ("128", "0.35", "2023"):
        assert item in result.text
    assert compare_core_snapshots(
        snapshot_core_content(original, ".md"),
        snapshot_core_content(result.text, ".md"),
    )["numeric_integrity_check"]


def test_numeric_integrity_ignores_safe_format_changes() -> None:
    original = "阈值为 0.350，比例为 90.0%，样本量为 128。"
    revised = "样本量为 128，阈值为 0.35，比例为 90%。"

    assert normalize_numeric_tokens_for_integrity(original) == ["0.35", "90%", "128"]
    assert compare_core_snapshots(
        snapshot_core_content(original, ".md"),
        snapshot_core_content(revised, ".md"),
    )["numeric_integrity_check"]


def test_terminology_integrity_uses_protected_segments() -> None:
    original_terms = ["AIGC", "base_only", "checkpoints/best.pth"]
    revised_terms = ["checkpoints/best.pth", "AIGC", "base_only", "ExtraGenericTerm"]

    assert compare_protected_segments(original_terms, revised_terms)


def test_rewrite_changes_style_but_not_core_content(tmp_path: Path) -> None:
    source = tmp_path / "style.md"
    original = _build_core_sample()
    source.write_text(original, encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.BALANCED)
    comparison = compare_core_snapshots(
        snapshot_core_content(original, ".md"),
        snapshot_core_content(result.text, ".md"),
    )

    assert result.text != original
    assert all(comparison.values())


def test_humanized_style_is_not_colloquial(tmp_path: Path) -> None:
    source = tmp_path / "formal.md"
    original = _build_core_sample()
    source.write_text(original, encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.BALANCED)

    for marker in ("真的", "其实", "然后", "特别", "超级"):
        assert marker not in result.text
