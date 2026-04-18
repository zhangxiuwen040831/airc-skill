from pathlib import Path

from airc_skill.config import RewriteMode
from airc_skill.pipeline import rewrite_file
from airc_skill.reviewer import review_revision
from airc_skill.rewriter import RewriteStats


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
        high_value_structural_actions=[action for action in structural_actions if action in {"pair_fusion", "conclusion_absorb", "paragraph_reorder", "subject_chain_compression"}],
        prefix_only_rewrite=prefix_only_rewrite,
        repeated_subject_risk=repeated_subject_risk,
        selected_variants=selected_variants,
        candidate_notes=[],
        paragraph_index=1,
        block_id=1,
    )


class StubRewriter:
    def __init__(self, rewritten_text: str, stats: RewriteStats) -> None:
        self.rewritten_text = rewritten_text
        self.stats = stats

    def reset_document_state(self) -> None:
        return None

    def rewrite(self, text: str, mode: RewriteMode, pass_index: int = 1) -> tuple[str, RewriteStats]:
        return self.rewritten_text, self.stats


def test_reject_prefix_only_rewrite() -> None:
    original = "因此，本研究需要重新评估人工校对流程。"
    revised = "基于此，本研究需要重新评估人工校对流程。"
    stats = _make_stats(
        mode=RewriteMode.BALANCED,
        original_sentences=[original],
        rewritten_sentences=[revised],
        applied_rules=["opening:implication"],
        selected_variants=["基于此"],
        structural_actions=[],
        paragraph_char_count=36,
        subject_heads=["本研究"],
        prefix_only_rewrite=True,
    )

    review = review_revision(original, revised, RewriteMode.BALANCED, rewrite_stats=[stats])

    assert review.decision == "reject"
    assert "Only prefix-level rewriting was detected." in review.problems


def test_reject_no_structural_action() -> None:
    original = "本研究聚焦于课堂反馈机制。"
    revised = "本文聚焦于课堂反馈机制。"
    stats = _make_stats(
        mode=RewriteMode.BALANCED,
        original_sentences=[original],
        rewritten_sentences=[revised],
        applied_rules=["sentence:study-focus"],
        selected_variants=["本文"],
        structural_actions=[],
        paragraph_char_count=20,
        subject_heads=["本文"],
    )

    review = review_revision(original, revised, RewriteMode.BALANCED, rewrite_stats=[stats])

    assert review.decision == "reject"
    assert "No structural action was executed." in review.problems


def test_reject_repeated_subject_chain() -> None:
    original = "本研究围绕课堂反馈机制展开。本研究进一步讨论系统流程。因此，本研究继续强调人工复核。"
    revised = original
    stats = _make_stats(
        mode=RewriteMode.BALANCED,
        original_sentences=[
            "本研究围绕课堂反馈机制展开。",
            "本研究进一步讨论系统流程。",
            "因此，本研究继续强调人工复核。",
        ],
        rewritten_sentences=[
            "本研究围绕课堂反馈机制展开。",
            "本研究进一步讨论系统流程。",
            "因此，本研究继续强调人工复核。",
        ],
        applied_rules=["subject:none"],
        selected_variants=["本研究", "本研究", "本研究"],
        structural_actions=["pair_fusion"],
        paragraph_char_count=66,
        subject_heads=["本研究", "本研究", "本研究"],
        repeated_subject_risk=True,
    )

    review = review_revision(original, revised, RewriteMode.BALANCED, rewrite_stats=[stats])

    assert review.decision == "reject"
    assert "Repeated subject chain was not repaired." in review.problems


def test_requires_high_value_action(tmp_path: Path) -> None:
    source = tmp_path / "high_value.txt"
    original = "系统设计持续推进。相关流程也在不断完善。部署问题仍需统筹考虑。"
    source.write_text(original, encoding="utf-8")
    rewritten = "系统设计持续推进，相关流程也在不断完善。部署问题仍需统筹考虑。"
    stats = _make_stats(
        mode=RewriteMode.BALANCED,
        original_sentences=["系统设计持续推进。", "相关流程也在不断完善。", "部署问题仍需统筹考虑。"],
        rewritten_sentences=["系统设计持续推进，相关流程也在不断完善。", "部署问题仍需统筹考虑。"],
        applied_rules=["structure:merge-short-followup"],
        selected_variants=["省略过渡"],
        structural_actions=["sentence_merge"],
        paragraph_char_count=36,
        subject_heads=["系统设计", "部署问题仍需"],
    )

    result = rewrite_file(source, mode=RewriteMode.BALANCED, rewriter=StubRewriter(rewritten, stats))

    assert result.write_gate.decision == "reject"
    assert any(
        "missing_high_value_action" in reason or "required rewrite obligations" in reason
        for reason in result.write_gate.reason_codes
    )
    assert result.output_written is False


def test_template_repetition_rejected() -> None:
    original = "研究目标涉及系统设计。研究流程覆盖训练阶段。研究任务还关注部署策略。"
    revised = "本研究围绕系统设计展开。本文主要讨论训练阶段。研究重点在于部署策略。"
    stats = _make_stats(
        mode=RewriteMode.BALANCED,
        original_sentences=["研究目标涉及系统设计。", "研究流程覆盖训练阶段。", "研究任务还关注部署策略。"],
        rewritten_sentences=["本研究围绕系统设计展开。", "本文主要讨论训练阶段。", "研究重点在于部署策略。"],
        applied_rules=["sentence:study-focus"] * 3,
        selected_variants=["本研究围绕", "本文主要讨论", "研究重点在于"],
        structural_actions=["topic_reframe", "subject_chain_compression"],
        paragraph_char_count=96,
        subject_heads=["本研究", "本文", "研究"],
    )

    review = review_revision(original, revised, RewriteMode.BALANCED, rewrite_stats=[stats])

    assert review.decision == "reject"
    assert review.template_risk is True
