from __future__ import annotations

from pathlib import Path

from airc_skill.config import RewriteMode
from airc_skill.pipeline import rewrite_file
from airc_skill.reviewer import review_revision
from airc_skill.rewriter import RewriteStats, Rewriter


def _load_user_test_md_excerpt() -> str:
    root = Path(__file__).resolve().parent / "fixtures" / "user_test.md"
    text = root.read_text(encoding="utf-8")
    start = text.index("## 1.1 研究背景")
    end = text.index("## 1.2 研究意义")
    return text[start:end]


def _make_stats(
    *,
    mode: RewriteMode,
    original_sentences: list[str],
    rewritten_sentences: list[str],
    applied_rules: list[str],
    selected_variants: list[str],
    structural_actions: list[str],
    paragraph_char_count: int,
    subject_heads: list[str],
    prefix_only_rewrite: bool = False,
    repeated_subject_risk: bool = False,
) -> RewriteStats:
    return RewriteStats(
        mode=mode,
        changed=True,
        applied_rules=applied_rules,
        sentence_count_before=len(original_sentences),
        sentence_count_after=len(rewritten_sentences),
        sentence_level_change=True,
        changed_characters=36,
        original_sentences=original_sentences,
        rewritten_sentences=rewritten_sentences,
        paragraph_char_count=paragraph_char_count,
        sentence_labels=["objective"] * len(original_sentences),
        subject_heads=subject_heads,
        detected_patterns=["study_description"],
        structural_actions=structural_actions,
        structural_action_count=len(structural_actions),
        high_value_structural_actions=[action for action in structural_actions if action in {"meta_compression", "followup_absorb"}],
        prefix_only_rewrite=prefix_only_rewrite,
        repeated_subject_risk=repeated_subject_risk,
        selected_variants=selected_variants,
        candidate_notes=[],
        paragraph_index=1,
    )


class StubRewriter:
    def __init__(self, rewritten_text: str, stats: RewriteStats) -> None:
        self.rewritten_text = rewritten_text
        self.stats = stats

    def reset_document_state(self) -> None:
        return None

    def rewrite(self, text: str, mode: RewriteMode, pass_index: int = 1) -> tuple[str, RewriteStats]:
        return self.rewritten_text, self.stats


def test_conservative_can_pass_without_strong_template_pattern(tmp_path: Path) -> None:
    source = tmp_path / "user_excerpt.md"
    source.write_text(_load_user_test_md_excerpt(), encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.CONSERVATIVE)

    assert result.review.decision in {"pass", "pass_with_minor_risk", "reject"}


def test_template_family_repetition_is_counted() -> None:
    original = "研究目标涉及系统设计。研究流程覆盖训练阶段。研究任务还关注部署策略。"
    revised = "本研究围绕系统设计展开。本文主要讨论训练阶段。研究重点在于部署策略。"
    stats = _make_stats(
        mode=RewriteMode.BALANCED,
        original_sentences=[
            "研究目标涉及系统设计。",
            "研究流程覆盖训练阶段。",
            "研究任务还关注部署策略。",
        ],
        rewritten_sentences=[
            "本研究围绕系统设计展开。",
            "本文主要讨论训练阶段。",
            "研究重点在于部署策略。",
        ],
        applied_rules=["sentence:study-focus"] * 3,
        selected_variants=["本研究围绕", "本文主要讨论", "研究重点在于"],
        structural_actions=["topic_reframe"],
        paragraph_char_count=96,
        subject_heads=["本研究", "本文", "研究"],
    )

    review = review_revision(original, revised, RewriteMode.BALANCED, rewrite_stats=[stats])

    assert review.decision == "reject"
    assert review.template_issue == "templated_family_repetition"


def test_keep_original_beats_bad_template_rewrite() -> None:
    text = "总的来说，这一问题在很多方面都具有重要意义。"

    rewritten, stats = Rewriter().rewrite(text, mode=RewriteMode.CONSERVATIVE)

    assert rewritten == text
    assert not stats.applied_rules


def test_rejected_candidate_does_not_write(tmp_path: Path) -> None:
    source = tmp_path / "user_full.md"
    root = Path(__file__).resolve().parent / "fixtures" / "user_test.md"
    source.write_text(root.read_text(encoding="utf-8"), encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.BALANCED)

    assert result.review.decision in {"pass", "pass_with_minor_risk", "reject"}
    if result.review.decision == "reject":
        assert result.output_written is False


def test_severe_templated_still_blocks_write(tmp_path: Path) -> None:
    source = tmp_path / "templated.txt"
    original = "系统设计持续推进。相关流程也在不断完善。部署问题仍需统筹考虑。"
    source.write_text(original, encoding="utf-8")

    rewritten = "本研究围绕系统设计推进展开。本文主要讨论相关流程完善。研究重点在于部署问题。"
    stats = _make_stats(
        mode=RewriteMode.BALANCED,
        original_sentences=["系统设计持续推进。", "相关流程也在不断完善。", "部署问题仍需统筹考虑。"],
        rewritten_sentences=["本研究围绕系统设计推进展开。", "本文主要讨论相关流程完善。", "研究重点在于部署问题。"],
        applied_rules=["sentence:study-focus"] * 3,
        selected_variants=["本研究围绕", "本文主要讨论", "研究重点在于"],
        structural_actions=["topic_reframe"],
        paragraph_char_count=92,
        subject_heads=["本研究", "本文", "研究"],
    )

    result = rewrite_file(source, mode=RewriteMode.BALANCED, rewriter=StubRewriter(rewritten, stats))

    assert result.review.decision == "reject"
    assert result.output_written is False
