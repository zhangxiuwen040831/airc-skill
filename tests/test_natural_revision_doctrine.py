from __future__ import annotations

from pathlib import Path

from airc_skill.config import RewriteMode
from airc_skill.guidance import guide_document_text
from airc_skill.natural_revision_profile import ACADEMIC_NATURAL_STUDENTLIKE
from airc_skill.pipeline import run_file
from airc_skill.rewriter import Rewriter


COLLOQUIAL_MARKERS = ("这块", "这边", "大家", "我们", "里头", "超", "超级", "真的", "大白话")


def test_reduce_function_word_overuse() -> None:
    text = "本文进行了课堂反馈机制的分析。"

    rewritten, stats = Rewriter().rewrite(text, mode=RewriteMode.BALANCED)

    assert rewritten == "本文分析了课堂反馈机制。"
    assert "reduce_function_word_overuse" in stats.discourse_actions_used
    assert "进行了" not in rewritten


def test_weaken_template_connectors_without_colloquializing() -> None:
    text = "首先，平台能力正在进入课堂评价场景。其次，教师需要重新分配反馈精力。"

    rewritten, stats = Rewriter().rewrite(text, mode=RewriteMode.BALANCED)

    assert "首先" not in rewritten
    assert "其次" not in rewritten
    assert "weaken_template_connectors" in stats.discourse_actions_used
    assert not any(marker in rewritten for marker in COLLOQUIAL_MARKERS)


def test_repeated_study_subject_chain_is_compressed() -> None:
    text = (
        "本研究的主题为课堂反馈机制的优化路径。"
        "本研究不仅包含多轮写作任务设计，还完成了反馈链条的流程梳理。"
        "因此，本研究进一步讨论系统输出与人工评价之间的协同方式。"
    )

    rewritten, stats = Rewriter().rewrite(text, mode=RewriteMode.BALANCED, pass_index=2)

    assert rewritten.count("本研究") < text.count("本研究")
    assert "subject_chain_compression" in stats.discourse_actions_used
    assert stats.repeated_subject_risk is False


def test_parallel_sentences_are_broken_up() -> None:
    text = "不仅要完善反馈流程，还要提升教师判断效率。"

    rewritten, stats = Rewriter().rewrite(text, mode=RewriteMode.BALANCED)

    assert "不仅要" not in rewritten
    assert "还要" not in rewritten
    assert "break_parallelism" in stats.discourse_actions_used


def test_high_density_technical_sentence_prefers_simplify_not_expand() -> None:
    text = "本研究最终采用 Stable Diffusion 与 CIFAKE 数据集，并加载 checkpoints/best.pth 作为部署 checkpoint。"

    guidance = guide_document_text(text, metadata={"suffix": ".txt"})
    policy = guidance.block_policies[0]

    assert policy.edit_policy in {"high_risk", "light_edit"}
    assert "keep_original_if_technical_density_is_high" in policy.recommended_actions


def test_low_density_descriptive_sentence_can_be_developed() -> None:
    text = (
        "数字平台正在进入课堂评价场景。与此同时，教师需要重新分配反馈精力。"
        "因此，原有依赖人工逐段反馈的流程面临更高压力。"
    )

    guidance = guide_document_text(text, metadata={"suffix": ".txt"})
    policy = guidance.block_policies[0]

    assert policy.edit_policy == "rewritable"
    assert policy.rewrite_depth == "developmental_rewrite"
    assert "sentence_cluster_rewrite" in policy.required_discourse_actions
    assert "rebuild_sentence_rhythm" in policy.recommended_actions


def test_academic_natural_studentlike_profile_is_not_colloquial() -> None:
    profile = ACADEMIC_NATURAL_STUDENTLIKE

    assert profile.name == "academic_natural_studentlike"
    assert "reduce_function_word_overuse" in profile.required_actions
    assert "break_parallelism" in profile.required_actions
    assert "这块" in profile.forbidden_markers
    assert "core_consistency" in profile.checklist


def test_real_test_md_body_rewrite_reads_less_template_like() -> None:
    sample = Path("tests") / "fixtures" / "user_test.md"
    if not sample.exists():
        return

    result = run_file(sample, preset="academic_natural", dry_run=True)

    assert result.rewrite_result.review.rewrite_coverage >= 0.6
    assert result.rewrite_result.review.natural_revision_checklist["academic_not_colloquial"] is True
    assert result.rewrite_result.review.natural_revision_checklist["format_and_markdown_preserved"] is True
    original = sample.read_text(encoding="utf-8")
    assert not any(marker in result.rewrite_result.text and marker not in original for marker in COLLOQUIAL_MARKERS)


def test_real_test_md_preserves_core_content_and_markdown() -> None:
    sample = Path("tests") / "fixtures" / "user_test.md"
    if not sample.exists():
        return

    result = run_file(sample, preset="academic_natural", dry_run=True)

    assert result.rewrite_result.review.core_content_integrity is True
    assert result.rewrite_result.review.format_integrity is True
    assert result.rewrite_result.review.markdown_symbol_integrity is True
