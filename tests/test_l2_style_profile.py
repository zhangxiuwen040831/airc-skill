from pathlib import Path

from airc_skill.config import RewriteMode, get_skill_preset
from airc_skill.l2_style_profile import aggregate_l2_style_profile, analyze_l2_style_sentences
from airc_skill.pipeline import run_file
from airc_skill.rewriter import RewriteStats, Rewriter


def _stats(sentences: list[str]) -> RewriteStats:
    return RewriteStats(
        mode=RewriteMode.BALANCED,
        changed=True,
        applied_rules=["local:inject-mild-l2-texture"],
        sentence_count_before=len(sentences),
        sentence_count_after=len(sentences),
        sentence_level_change=True,
        changed_characters=12,
        original_sentences=sentences,
        rewritten_sentences=sentences,
        paragraph_char_count=sum(len(sentence) for sentence in sentences),
        sentence_labels=[],
        subject_heads=[],
        detected_patterns=[],
        structural_actions=["inject_mild_l2_texture"],
        structural_action_count=1,
        high_value_structural_actions=[],
        rewrite_depth="light_edit",
        rewrite_intensity="medium",
        paragraph_index=1,
    )


def test_l2_profile_preserves_facts_and_terms() -> None:
    rewriter = Rewriter(style_profile="zh_academic_l2_mild")
    revised, _ = rewriter.rewrite(
        "AIGC检测模型用于提取频域特征，并保持F_1=0.90这一结果。",
        RewriteMode.BALANCED,
        rewrite_depth="light_edit",
        rewrite_intensity="medium",
    )

    assert "AIGC" in revised
    assert "频域特征" in revised
    assert "F_1=0.90" in revised


def test_l2_profile_adds_mild_function_word_density() -> None:
    signals = analyze_l2_style_sentences(["这个模型是用来进行特征提取的工作，并能够进行结果分析。"])

    assert signals.l2_texture_present
    assert signals.function_word_density > 0


def test_l2_profile_not_colloquial() -> None:
    signals = analyze_l2_style_sentences(["这个系统是用来进行检测工作的。"])

    assert signals.not_colloquial


def test_l2_profile_not_ungrammatical() -> None:
    signals = analyze_l2_style_sentences(["这个策略的目的，是通过课程采样来进行边界样本的训练。"])

    assert signals.not_ungrammatical


def test_l2_profile_flags_bad_l2_collocations() -> None:
    signals = analyze_l2_style_sentences(["多分支模型能够并非天然更优，伪造风险可能被是用来误导决策。"])

    assert not signals.not_ungrammatical


def test_l2_profile_less_native_like_than_academic_natural() -> None:
    default_rewriter = Rewriter(style_profile="academic_natural")
    l2_rewriter = Rewriter(style_profile="zh_academic_l2_mild")
    source = "该模块用于提取特征，并实现语义融合。"

    default_text, _ = default_rewriter.rewrite(
        source,
        RewriteMode.BALANCED,
        rewrite_depth="light_edit",
        rewrite_intensity="medium",
    )
    l2_text, _ = l2_rewriter.rewrite(
        source,
        RewriteMode.BALANCED,
        rewrite_depth="light_edit",
        rewrite_intensity="medium",
    )

    assert l2_text != default_text
    assert "的工作" in l2_text or "是用来" in l2_text


def test_l2_profile_on_real_test_md_outputs_mild_l2_texture(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parent / "fixtures" / "user_test.md"
    output = tmp_path / "test_airc_l2.md"
    report = tmp_path / "test_airc_l2.report.json"

    result = run_file(
        source,
        preset="zh_academic_l2_mild",
        output_path=output,
        report_path=report,
        max_retry_passes=1,
    )

    assert result.output_written
    assert output.exists()
    assert report.exists()
    assert result.rewrite_result.review.l2_style_profile["enabled"] is True
    assert result.rewrite_result.review.l2_style_profile["l2_texture_present"] is True
    assert result.rewrite_result.review.l2_style_profile["not_colloquial"] is True
    assert result.rewrite_result.review.l2_style_profile["not_ungrammatical"] is True
    assert result.rewrite_result.review.body_rewrite_coverage >= 0.60


def test_l2_profile_preset_is_explicit() -> None:
    preset = get_skill_preset("zh_academic_l2_mild")

    assert preset.target_style == "zh_academic_l2_mild"
    assert get_skill_preset("academic_natural").target_style == "academic_natural"


def test_l2_profile_aggregate_requires_texture_when_enabled() -> None:
    profile = aggregate_l2_style_profile([_stats(["模型完成检测。"])], enabled=True)

    assert profile["enabled"] is True
    assert profile["l2_texture_present"] is False
