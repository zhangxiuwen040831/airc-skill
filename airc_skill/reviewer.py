from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

from .config import DEFAULT_CONFIG, RewriteMode
from .core_guard import compare_core_snapshots, snapshot_core_content
from .models import GuidanceReport, ReviewReport, RewriteCandidate as BlockRewriteCandidate
from .rewriter import RewriteStats, split_sentences

_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
_FENCED_CODE_RE = re.compile(
    r"(?ms)^(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^\s*(?P=fence)[ \t]*$",
    re.MULTILINE,
)
_INLINE_LINK_RE = re.compile(r"(?<!!)\[[^\]]+]\([^)]+\)")
_AUTOLINK_RE = re.compile(r"<https?://[^>\n]+>")
_BARE_URL_RE = re.compile(r"https?://[^\s<>()]+(?:\([^\s<>()]*\)[^\s<>()]*)*")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_NORMALIZE_RE = re.compile(r"\s+")
_STIFF_OPENERS = ("本研究", "本文", "从", "对", "这也意味着", "基于此", "在这种情况下", "总体来看", "综合来看")
_META_SUBJECTS = ("本研究", "本文", "该研究", "该系统")
_LEADING_IMPLICATION_RE = re.compile(r"^(?:因此|由此|这也意味着|基于此|在这种情况下|正因为如此|在此基础上)[，,\s]*")
_CORE_INTEGRITY_KEYS = {
    "title_integrity_check",
    "formula_integrity_check",
    "citation_integrity_check",
    "terminology_integrity_check",
    "numeric_integrity_check",
    "path_integrity_check",
}
_FORMAT_INTEGRITY_KEYS = {
    "heading_format_integrity_check",
    "english_spacing_integrity_check",
    "placeholder_integrity_check",
    "caption_punctuation_integrity_check",
    "markdown_symbol_integrity_check",
    "linebreak_whitespace_integrity_check",
}
_SUBJECT_FIX_ACTIONS = {
    "merge_consecutive_subject_sentences",
    "subject_drop",
    "subject_variation",
    "meta_compression",
    "followup_absorb",
    "conclusion_absorb",
}
_STRICT_TEMPLATE_MARKERS = ("从", "角度看", "在此基础上", "这也意味着", "进一步来看", "综上")
_REQUIRED_ACTION_ALIASES = {
    "pair_fusion": {"pair_fusion", "sentence_merge", "clause_reorder"},
    "conclusion_absorb": {"conclusion_absorb", "followup_absorb"},
    "subject_chain_compression": {
        "subject_chain_compression",
        "merge_consecutive_subject_sentences",
        "subject_drop",
        "subject_variation",
        "meta_compression",
        "followup_absorb",
    },
    "enumeration_reframe": {"enumeration_reframe"},
    "paragraph_reorder": {"paragraph_reorder"},
    "clause_reorder": {"clause_reorder"},
    "sentence_merge": {"sentence_merge", "pair_fusion"},
}

@dataclass(frozen=True)
class NaturalnessAssessment:
    depth_sufficient: bool
    template_risk: bool
    template_warning: bool
    template_issue: str | None
    naturalness_risk: bool
    structural_action_count: int
    high_value_action_count: int
    high_value_actions: list[str]
    discourse_change_score: int
    cluster_rewrite_score: int
    style_variation_score: int
    prefix_only_rewrite: bool
    repeated_subject_risk: bool
    repeated_subject_warning: bool
    reasons: list[str]
    warnings: list[str]


def review_rewrite(
    original: str,
    revised: str,
    guidance: GuidanceReport | None = None,
    mode: RewriteMode = RewriteMode.BALANCED,
    rewrite_stats: list[RewriteStats] | None = None,
    block_candidates: list[BlockRewriteCandidate] | None = None,
    suffix: str = ".txt",
) -> ReviewReport:
    stats_list = rewrite_stats or []
    candidates = block_candidates or []
    core_problems: list[str] = []
    format_problems: list[str] = []
    reject_reasons: list[str] = []
    warnings: list[str] = []
    failed_block_ids: list[int] = []

    if not revised.strip():
        reject_reasons.append("Rewritten text is empty.")

    if _NUMBER_RE.findall(original) != _NUMBER_RE.findall(revised):
        core_problems.append("Numeric values changed.")

    if _count_headings(original) != _count_headings(revised):
        format_problems.append("Heading count changed.")

    if _count_code_blocks(original) != _count_code_blocks(revised):
        format_problems.append("Code block count changed.")

    if _count_links(original) != _count_links(revised):
        format_problems.append("Link count changed.")

    if revised.count("；") > original.count("；"):
        reject_reasons.append("Unexpected Chinese semicolons were introduced.")

    integrity_checks = compare_core_snapshots(
        snapshot_core_content(original, suffix=suffix),
        snapshot_core_content(revised, suffix=suffix),
    )
    if guidance is not None and not _do_not_touch_blocks_preserved(revised, guidance):
        format_problems.append("Guidance-marked do_not_touch blocks changed.")

    for check_name, passed in integrity_checks.items():
        if not passed:
            if check_name in _CORE_INTEGRITY_KEYS:
                core_problems.append(f"{check_name} failed.")
            elif check_name in _FORMAT_INTEGRITY_KEYS:
                format_problems.append(f"{check_name} failed.")

    changed_characters, diff_spans = _measure_diff(original, revised)
    sentence_level_change = _has_sentence_level_change(original, revised)
    assessment = _review_naturalness(
        original=original,
        revised=revised,
        mode=mode,
        rewrite_stats=stats_list,
    )
    rewrite_coverage = _compute_rewrite_coverage_metric(original, revised, stats_list)
    high_value_actions = assessment.high_value_actions or _collect_high_value_actions(stats_list)
    meaningful_change = _has_meaningful_change(
        original=original,
        revised=revised,
        mode=mode,
        changed_characters=changed_characters,
        diff_spans=diff_spans,
        sentence_level_change=sentence_level_change,
        structural_action_count=assessment.structural_action_count,
        discourse_change_score=assessment.discourse_change_score,
        cluster_rewrite_score=assessment.cluster_rewrite_score,
        rewrite_coverage=rewrite_coverage,
        prefix_only_rewrite=assessment.prefix_only_rewrite,
        depth_sufficient=assessment.depth_sufficient,
    )
    effective_change = meaningful_change and (
        sentence_level_change
        or assessment.structural_action_count >= 1
        or assessment.discourse_change_score >= DEFAULT_CONFIG.light_edit_min_discourse_score
        or changed_characters >= (
            DEFAULT_CONFIG.strong_min_changed_chars if mode is RewriteMode.STRONG else DEFAULT_CONFIG.balanced_min_changed_chars
        )
    )

    substantive_body_rewrite = (
        rewrite_coverage >= DEFAULT_CONFIG.rewrite_coverage_pass_threshold
        and assessment.discourse_change_score >= _required_discourse_change_score(mode, len(_normalize_text(original)))
    )

    if assessment.structural_action_count == 0:
        reject_reasons.append("No structural action was executed.")
    if assessment.prefix_only_rewrite:
        reject_reasons.append("Only prefix-level rewriting was detected.")
    if assessment.repeated_subject_risk:
        reject_reasons.append("Repeated subject chain was not repaired.")
    if assessment.template_risk:
        if assessment.template_issue == "templated_rule_like_rewrite" and substantive_body_rewrite:
            warnings.append("templated_rule_like_rewrite remains at a minor level after substantive rewrite.")
        else:
            reject_reasons.append(f"{assessment.template_issue or 'templated_rule_like_rewrite'} detected.")
    if not effective_change:
        reject_reasons.append("Only word-level or negligible change was produced.")
    if mode in {RewriteMode.BALANCED, RewriteMode.STRONG} and not assessment.depth_sufficient:
        if substantive_body_rewrite:
            warnings.extend(assessment.reasons)
        else:
            reject_reasons.extend(assessment.reasons)
    if _count_blacklisted_templates(revised) > DEFAULT_CONFIG.strict_blacklist_repeat_limit:
        if substantive_body_rewrite:
            warnings.append("Blacklisted template openers remain but the body rewrite is substantive.")
        else:
            reject_reasons.append("Blacklisted template openers repeated beyond the strict threshold.")
    if mode in {RewriteMode.BALANCED, RewriteMode.STRONG} and len(_normalize_text(original)) >= DEFAULT_CONFIG.structural_gate_length:
        if rewrite_coverage < DEFAULT_CONFIG.rewrite_coverage_minor_threshold:
            reject_reasons.append("Rewrite coverage stayed below the developmental rewrite minimum.")
        elif rewrite_coverage < DEFAULT_CONFIG.rewrite_coverage_pass_threshold:
            warnings.append("Rewrite coverage is substantive but below the full-pass target.")

    if guidance is not None:
        block_problems, block_ids = _validate_block_obligations(guidance, candidates)
        if substantive_body_rewrite:
            warnings.extend(block_problems)
        else:
            reject_reasons.extend(block_problems)
            failed_block_ids.extend(block_ids)

    core_content_integrity = not core_problems
    format_integrity = not format_problems
    if core_problems or format_problems:
        reject_reasons.extend(core_problems + format_problems)

    warnings.extend(assessment.warnings)
    if assessment.template_warning and assessment.template_issue:
        warnings.append(f"{assessment.template_issue} remains at a minor level.")
    if assessment.repeated_subject_warning:
        warnings.append("Minor repeated-subject residue remains after an otherwise substantive rewrite.")

    problems = _deduplicate_preserve_order(reject_reasons)
    warnings = _deduplicate_preserve_order(warnings)
    failed_block_ids = sorted(set(failed_block_ids))
    if problems:
        decision = "reject"
    elif warnings:
        decision = "pass_with_minor_risk"
    else:
        decision = "pass"

    naturalness_score = _compute_naturalness_score(
        template_risk=assessment.template_risk,
        template_warning=assessment.template_warning,
        repeated_subject_risk=assessment.repeated_subject_risk or assessment.repeated_subject_warning,
        depth_sufficient=assessment.depth_sufficient,
        effective_change=effective_change,
    )
    suggested_fallback = _suggested_fallback(
        decision=decision,
        template_risk=assessment.template_risk,
        format_integrity=format_integrity,
        core_content_integrity=core_content_integrity,
    )
    write_gate_ready = (
        decision in {"pass", "pass_with_minor_risk"}
        and effective_change
        and assessment.structural_action_count >= _required_structural_action_count(mode, len(_normalize_text(original)))
        and len(high_value_actions) >= DEFAULT_CONFIG.strict_min_high_value_actions
        and assessment.discourse_change_score >= _required_discourse_change_score(mode, len(_normalize_text(original)))
        and assessment.cluster_rewrite_score >= _required_cluster_rewrite_score(mode, len(_normalize_text(original)))
        and rewrite_coverage >= DEFAULT_CONFIG.rewrite_coverage_minor_threshold
    )

    return ReviewReport(
        ok=decision in {"pass", "pass_with_minor_risk"},
        decision=decision,
        problems=problems,
        warnings=warnings,
        meaningful_change=meaningful_change,
        effective_change=effective_change,
        changed_characters=changed_characters,
        diff_spans=diff_spans,
        sentence_level_change=sentence_level_change,
        depth_sufficient=assessment.depth_sufficient,
        template_risk=assessment.template_risk,
        naturalness_risk=assessment.naturalness_risk,
        structural_action_count=assessment.structural_action_count,
        high_value_action_count=assessment.high_value_action_count,
        high_value_actions=high_value_actions,
        discourse_change_score=assessment.discourse_change_score,
        cluster_rewrite_score=assessment.cluster_rewrite_score,
        style_variation_score=assessment.style_variation_score,
        rewrite_coverage=rewrite_coverage,
        prefix_only_rewrite=assessment.prefix_only_rewrite,
        repeated_subject_risk=assessment.repeated_subject_risk or assessment.repeated_subject_warning,
        template_warning=assessment.template_warning,
        template_issue=assessment.template_issue,
        core_content_integrity=core_content_integrity,
        format_integrity=format_integrity,
        title_integrity=integrity_checks.get("title_integrity_check", True),
        formula_integrity=integrity_checks.get("formula_integrity_check", True),
        terminology_integrity=integrity_checks.get("terminology_integrity_check", True),
        citation_integrity=integrity_checks.get("citation_integrity_check", True),
        numeric_integrity=integrity_checks.get("numeric_integrity_check", True),
        path_integrity=integrity_checks.get("path_integrity_check", True),
        heading_format_integrity=integrity_checks.get("heading_format_integrity_check", True),
        english_spacing_integrity=integrity_checks.get("english_spacing_integrity_check", True),
        placeholder_integrity=integrity_checks.get("placeholder_integrity_check", True),
        caption_punctuation_integrity=integrity_checks.get("caption_punctuation_integrity_check", True),
        markdown_symbol_integrity=integrity_checks.get("markdown_symbol_integrity_check", True),
        linebreak_whitespace_integrity=integrity_checks.get("linebreak_whitespace_integrity_check", True),
        naturalness_score=naturalness_score,
        suggested_fallback=suggested_fallback,
        write_gate_ready=write_gate_ready,
        failed_block_ids=failed_block_ids,
        integrity_checks=integrity_checks,
    )


def review_revision(
    original: str,
    revised: str,
    mode: RewriteMode,
    rewrite_stats: list[RewriteStats] | None = None,
    suffix: str = ".txt",
) -> ReviewReport:
    return review_rewrite(
        original=original,
        revised=revised,
        guidance=None,
        mode=mode,
        rewrite_stats=rewrite_stats,
        block_candidates=None,
        suffix=suffix,
    )


def _measure_diff(original: str, revised: str) -> tuple[int, int]:
    matcher = SequenceMatcher(a=original, b=revised)
    changed_characters = 0
    diff_spans = 0

    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        diff_spans += 1
        changed_characters += max(left_end - left_start, right_end - right_start)

    return changed_characters, diff_spans


def _compute_rewrite_coverage_metric(
    original: str,
    revised: str,
    rewrite_stats: list[RewriteStats],
) -> float:
    if rewrite_stats:
        denominator = sum(max(1, stats.sentence_count_before) for stats in rewrite_stats if stats.changed)
        if denominator == 0:
            return 0.0
        changed_units = sum(
            min(
                max(1, stats.sentence_count_before),
                max(stats.sentence_level_changes, stats.cluster_changes),
            )
            for stats in rewrite_stats
            if stats.changed
        )
        return min(1.0, changed_units / denominator)

    original_sentences = split_sentences(original)
    revised_sentences = split_sentences(revised)
    if not original_sentences:
        return 0.0
    if len(original_sentences) != len(revised_sentences):
        return min(1.0, max(len(original_sentences), len(revised_sentences)) / len(original_sentences))
    changed = sum(
        1
        for left, right in zip(original_sentences, revised_sentences)
        if _normalize_sentence(left) != _normalize_sentence(right)
    )
    return min(1.0, changed / len(original_sentences))


def _review_naturalness(
    original: str,
    revised: str,
    mode: RewriteMode,
    rewrite_stats: list[RewriteStats],
) -> NaturalnessAssessment:
    if not rewrite_stats:
        original_sentences = split_sentences(original)
        revised_sentences = split_sentences(revised)
        repeated_subject_streak = _max_meta_subject_streak(_subject_heads_from_text(revised))
        repeated_subject_risk = repeated_subject_streak >= 3
        repeated_subject_warning = repeated_subject_streak == 2
        structural_action_count = int(
            len(original_sentences) != len(revised_sentences) or _has_sentence_level_change(original, revised)
        )
        high_value_actions = _infer_high_value_actions_from_text(original, revised)
        discourse_change_score = _infer_discourse_change_score(original, revised)
        cluster_rewrite_score = _infer_cluster_rewrite_score(original, revised)
        style_variation_score = _infer_style_variation_score(revised)
        prefix_only_rewrite = _looks_like_prefix_only_change(original, revised)
        severe_template_issue: str | None = None
        warning_template_issue: str | None = None
        if repeated_subject_streak >= 3 or _count_repeated_openers(revised) >= 3:
            severe_template_issue = "templated_family_repetition"
        elif _has_stiff_opener_sequence(revised) or style_variation_score <= DEFAULT_CONFIG.style_variation_reject_score:
            severe_template_issue = "templated_rule_like_rewrite"
        elif style_variation_score <= DEFAULT_CONFIG.style_variation_warning_score:
            warning_template_issue = "templated_rule_like_rewrite"
        template_risk = severe_template_issue is not None
        template_warning = severe_template_issue is None and warning_template_issue is not None
        reasons: list[str] = []
        warnings: list[str] = []
        if (
            mode in {RewriteMode.BALANCED, RewriteMode.STRONG}
            and len(_normalize_text(original)) >= DEFAULT_CONFIG.structural_gate_length
        ):
            if discourse_change_score < _required_discourse_change_score(mode, len(_normalize_text(original))):
                reasons.append("External candidate shows no clear discourse-level restructuring.")
            if cluster_rewrite_score < _required_cluster_rewrite_score(mode, len(_normalize_text(original))):
                reasons.append("External candidate did not reorganize any sentence cluster.")
        if repeated_subject_warning:
            warnings.append("One repeated meta-subject pair remains in the revised text.")
        return NaturalnessAssessment(
            depth_sufficient=not reasons and not template_risk and not prefix_only_rewrite,
            template_risk=template_risk,
            template_warning=template_warning,
            template_issue=severe_template_issue or warning_template_issue,
            naturalness_risk=template_risk or template_warning or repeated_subject_risk or repeated_subject_warning or prefix_only_rewrite,
            structural_action_count=structural_action_count,
            high_value_action_count=len(high_value_actions),
            high_value_actions=high_value_actions,
            discourse_change_score=discourse_change_score,
            cluster_rewrite_score=cluster_rewrite_score,
            style_variation_score=style_variation_score,
            prefix_only_rewrite=prefix_only_rewrite,
            repeated_subject_risk=repeated_subject_risk,
            repeated_subject_warning=repeated_subject_warning,
            reasons=reasons,
            warnings=warnings,
        )

    selected_variants = [variant for stats in rewrite_stats for variant in stats.selected_variants if variant != "keep-original"]
    variant_counts = Counter(selected_variants)
    repeated_rule_families = _count_repeated_rule_families(rewrite_stats)
    template_family_counts = _count_template_families(rewrite_stats)
    paragraph_template_family_counts = _count_paragraph_template_families(rewrite_stats)
    max_variant_repeat = max(variant_counts.values(), default=0)
    structural_action_count = sum(stats.structural_action_count for stats in rewrite_stats)
    high_value_action_count = sum(len(stats.high_value_structural_actions) for stats in rewrite_stats)
    discourse_change_score = sum(stats.discourse_change_score for stats in rewrite_stats)
    cluster_rewrite_score = sum(stats.cluster_changes for stats in rewrite_stats)
    style_variation_score = _style_variation_score_from_stats(
        selected_variants=selected_variants,
        repeated_rule_families=repeated_rule_families,
        template_family_counts=template_family_counts,
        max_variant_repeat=max_variant_repeat,
        revised=revised,
    )
    changed_stats = [stats for stats in rewrite_stats if stats.changed]
    prefix_only_rewrite = bool(changed_stats) and all(
        stats.prefix_only_rewrite or stats.structural_action_count == 0 for stats in changed_stats
    )
    repeated_subject_streak = max((_max_meta_subject_streak(stats.subject_heads) for stats in rewrite_stats), default=0)
    repeated_subject_risk = repeated_subject_streak >= 3
    repeated_subject_warning = repeated_subject_streak == 2
    reasons: list[str] = []
    warnings: list[str] = []

    if mode in {RewriteMode.BALANCED, RewriteMode.STRONG} and len(_normalize_text(original)) >= DEFAULT_CONFIG.structural_gate_length:
        if not rewrite_stats:
            reasons.append("No rewrite statistics were produced for a long paragraph.")

        for stats in rewrite_stats:
            rewrite_depth = stats.rewrite_depth or ("developmental_rewrite" if mode in {RewriteMode.BALANCED, RewriteMode.STRONG} else "light_edit")
            required_actions, requires_high_value = _required_structural_actions(mode, stats.paragraph_char_count)
            if rewrite_depth == "light_edit":
                required_actions = 0
                requires_high_value = False
            if stats.prefix_only_rewrite:
                reasons.append(
                    f"Paragraph {stats.paragraph_index} only changed prefixes without any structural rewrite."
                )
                continue
            if rewrite_depth == "light_edit":
                if stats.sentence_level_changes < DEFAULT_CONFIG.light_edit_min_sentence_level_changes:
                    reasons.append(
                        f"Paragraph {stats.paragraph_index} did not reach the minimum sentence-level change for light_edit."
                    )
                if stats.discourse_change_score < DEFAULT_CONFIG.light_edit_min_discourse_score:
                    reasons.append(
                        f"Paragraph {stats.paragraph_index} did not reach the discourse-change threshold for light_edit."
                    )
                continue
            if stats.sentence_level_changes == 0 and stats.cluster_changes == 0:
                reasons.append(
                    f"Paragraph {stats.paragraph_index} did not produce any sentence-level or cluster-level change."
                )
                continue
            if stats.structural_action_count < required_actions:
                warnings.append(
                    f"Paragraph {stats.paragraph_index} has fewer structural actions than recommended for developmental rewrite."
                )
            if requires_high_value and len(stats.high_value_structural_actions) < 1:
                warnings.append(
                    f"Paragraph {stats.paragraph_index} lacks a high-value structural action for strong mode."
                )
            if stats.discourse_change_score < _required_discourse_change_score(mode, stats.paragraph_char_count):
                warnings.append(
                    f"Paragraph {stats.paragraph_index} did not reach the discourse-change threshold for body rewriting."
                )
            if stats.cluster_changes < _required_cluster_rewrite_score(mode, stats.paragraph_char_count):
                warnings.append(
                    f"Paragraph {stats.paragraph_index} did not show a sentence-cluster rewrite."
                )

    for stats in rewrite_stats:
        subject_streak = _max_meta_subject_streak(stats.subject_heads)
        subject_fix_count = sum(1 for action in stats.structural_actions if action in _SUBJECT_FIX_ACTIONS)
        if mode is RewriteMode.CONSERVATIVE and subject_streak >= 3:
            reasons.append(
                f"Paragraph {stats.paragraph_index} still has a repeated meta-subject chain of length {subject_streak}."
            )
        if mode is RewriteMode.BALANCED and subject_streak >= 3:
            reasons.append(
                f"Paragraph {stats.paragraph_index} still has consecutive repeated meta subjects."
            )
        if mode is RewriteMode.BALANCED and subject_streak == 2:
            warnings.append(
                f"Paragraph {stats.paragraph_index} still leaves one repeated meta-subject pair."
            )
        if mode is RewriteMode.STRONG and subject_streak >= 3:
            if subject_fix_count == 0:
                reasons.append(
                    f"Paragraph {stats.paragraph_index} repeats the same meta subject without merge, drop, or variation."
                )
            else:
                reasons.append(
                    f"Paragraph {stats.paragraph_index} still leaves a repeated meta-subject chain after restructuring."
                )
        if mode is RewriteMode.STRONG and subject_streak == 2:
            warnings.append(
                f"Paragraph {stats.paragraph_index} still leaves one repeated meta-subject pair after restructuring."
            )

    document_template_limit = (
        DEFAULT_CONFIG.document_template_family_block_limit
        if len(rewrite_stats) < 10
        else max(DEFAULT_CONFIG.document_template_family_block_limit + 3, len(rewrite_stats) // 4)
    )
    severe_family_repetition = (
        max(template_family_counts.values(), default=0) >= document_template_limit
        or any(
            count > DEFAULT_CONFIG.paragraph_template_family_limit
            for count in paragraph_template_family_counts.values()
        )
    )
    warning_family_repetition = (
        not severe_family_repetition
        and max(template_family_counts.values(), default=0) >= DEFAULT_CONFIG.document_template_family_warning_limit
    )
    severe_rule_like_rewrite = (
        repeated_rule_families >= DEFAULT_CONFIG.rule_like_family_block_limit
        or max_variant_repeat > DEFAULT_CONFIG.repeated_pattern_limit + 1
    )
    warning_rule_like_rewrite = (
        not severe_rule_like_rewrite
        and (
            repeated_rule_families >= DEFAULT_CONFIG.rule_like_family_warning_limit
            or _has_stiff_opener_sequence(revised)
        )
    )

    severe_template_issue: str | None = None
    if severe_family_repetition or repeated_subject_streak >= 3:
        severe_template_issue = "templated_family_repetition"
    elif severe_rule_like_rewrite:
        severe_template_issue = "templated_rule_like_rewrite"

    warning_template_issue: str | None = None
    if not severe_template_issue:
        if warning_family_repetition:
            warning_template_issue = "templated_family_repetition"
        elif warning_rule_like_rewrite:
            warning_template_issue = "templated_rule_like_rewrite"

    template_risk = bool(severe_template_issue)
    template_warning = False
    template_issue = severe_template_issue or warning_template_issue
    if not severe_template_issue and (
        warning_family_repetition
        or warning_rule_like_rewrite
        or style_variation_score <= DEFAULT_CONFIG.style_variation_warning_score
    ):
        template_warning = True
    naturalness_risk = (
        template_risk
        or template_warning
        or prefix_only_rewrite
        or repeated_subject_risk
        or repeated_subject_warning
    )
    depth_sufficient = not reasons and not (mode in {RewriteMode.BALANCED, RewriteMode.STRONG} and prefix_only_rewrite)

    if mode is RewriteMode.CONSERVATIVE:
        depth_sufficient = not template_risk and repeated_subject_streak < 3
        reasons = []

    return NaturalnessAssessment(
        depth_sufficient=depth_sufficient,
        template_risk=template_risk,
        template_warning=template_warning,
        template_issue=template_issue,
        naturalness_risk=naturalness_risk,
        structural_action_count=structural_action_count,
        high_value_action_count=high_value_action_count,
        high_value_actions=_collect_high_value_actions(rewrite_stats),
        discourse_change_score=discourse_change_score,
        cluster_rewrite_score=cluster_rewrite_score,
        style_variation_score=style_variation_score,
        prefix_only_rewrite=prefix_only_rewrite,
        repeated_subject_risk=repeated_subject_risk,
        repeated_subject_warning=repeated_subject_warning,
        reasons=_deduplicate_preserve_order(reasons),
        warnings=_deduplicate_preserve_order(warnings),
    )


def _has_meaningful_change(
    original: str,
    revised: str,
    mode: RewriteMode,
    changed_characters: int,
    diff_spans: int,
    sentence_level_change: bool,
    structural_action_count: int,
    discourse_change_score: int,
    cluster_rewrite_score: int,
    rewrite_coverage: float,
    prefix_only_rewrite: bool,
    depth_sufficient: bool,
) -> bool:
    if original == revised:
        return False

    if mode is RewriteMode.CONSERVATIVE:
        return changed_characters >= 2 or diff_spans >= 1

    if prefix_only_rewrite:
        return False

    threshold = (
        DEFAULT_CONFIG.strong_min_changed_chars
        if mode is RewriteMode.STRONG
        else DEFAULT_CONFIG.balanced_min_changed_chars
    )
    normalized_length = len(_normalize_text(original))

    if normalized_length >= DEFAULT_CONFIG.structural_gate_length and structural_action_count < 1:
        return False
    if (
        not depth_sufficient
        and discourse_change_score < _required_discourse_change_score(mode, normalized_length)
    ):
        return False
    if (
        cluster_rewrite_score < _required_cluster_rewrite_score(mode, normalized_length)
        and rewrite_coverage < DEFAULT_CONFIG.rewrite_coverage_minor_threshold
    ):
        return False

    return (
        sentence_level_change
        or structural_action_count >= 1
        or discourse_change_score >= DEFAULT_CONFIG.developmental_min_discourse_score
        or rewrite_coverage >= DEFAULT_CONFIG.rewrite_coverage_minor_threshold
        or (changed_characters >= threshold and diff_spans >= 2)
    )


def _required_structural_actions(mode: RewriteMode, paragraph_char_count: int) -> tuple[int, bool]:
    if mode is RewriteMode.CONSERVATIVE:
        return 0, False
    if paragraph_char_count >= DEFAULT_CONFIG.structural_gate_length_long:
        if mode is RewriteMode.BALANCED:
            return DEFAULT_CONFIG.balanced_min_structural_actions_long, False
        return DEFAULT_CONFIG.strong_min_structural_actions_long, True
    if paragraph_char_count >= DEFAULT_CONFIG.structural_gate_length:
        if mode is RewriteMode.BALANCED:
            return DEFAULT_CONFIG.balanced_min_structural_actions, False
        return DEFAULT_CONFIG.strong_min_structural_actions, True
    return 0, False


def _required_discourse_change_score(mode: RewriteMode, normalized_length: int) -> int:
    if mode is RewriteMode.CONSERVATIVE:
        return DEFAULT_CONFIG.light_edit_min_discourse_score
    if normalized_length >= DEFAULT_CONFIG.structural_gate_length:
        return DEFAULT_CONFIG.developmental_min_discourse_score
    return DEFAULT_CONFIG.light_edit_min_discourse_score


def _required_cluster_rewrite_score(mode: RewriteMode, normalized_length: int) -> int:
    if mode is RewriteMode.CONSERVATIVE:
        return 0
    if normalized_length >= DEFAULT_CONFIG.structural_gate_length:
        return DEFAULT_CONFIG.developmental_min_cluster_score
    return 0


def _count_repeated_rule_families(rewrite_stats: list[RewriteStats]) -> int:
    families: list[str] = []
    for stats in rewrite_stats:
        for rule in stats.applied_rules:
            if rule.startswith(("subject:", "structure:", "fusion:", "paragraph:")):
                continue
            if ":" in rule:
                families.append(rule)
    counts = Counter(families)
    return max(counts.values(), default=0)


def _count_template_families(rewrite_stats: list[RewriteStats]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for stats in rewrite_stats:
        counts.update(_template_families_from_stats(stats))
    return counts


def _count_paragraph_template_families(rewrite_stats: list[RewriteStats]) -> Counter[str]:
    paragraph_counts: Counter[str] = Counter()
    for stats in rewrite_stats:
        family_counts = Counter(_template_families_from_stats(stats))
        for family, count in family_counts.items():
            paragraph_counts[family] = max(paragraph_counts.get(family, 0), count)
    return paragraph_counts


def _template_families_from_stats(stats: RewriteStats) -> list[str]:
    families: list[str] = []
    for variant in stats.selected_variants:
        family = _variant_to_template_family(variant)
        if family:
            families.append(family)
    return families


def _variant_to_template_family(variant: str) -> str | None:
    if variant in {"本研究", "本文", "该研究", "研究"}:
        return "study_subject_family"
    if any(marker in variant for marker in ("本研究", "本文", "研究")) and "围绕" not in variant:
        return "study_subject_family"
    if any(marker in variant for marker in ("因此", "由此", "基于此", "这也意味着", "在这种情况下", "正因为如此", "在此基础上")):
        return "implication_family"
    if any(marker in variant for marker in ("同时", "与此同时", "在这一过程中", "此外", "与之相伴的是")):
        return "transition_family"
    if any(marker in variant for marker in ("围绕", "聚焦于", "尝试回应", "核心问题", "重点在于")):
        return "framing_family"
    return None




def _has_sentence_level_change(original: str, revised: str) -> bool:
    original_sentences = split_sentences(original)
    revised_sentences = split_sentences(revised)

    if len(original_sentences) != len(revised_sentences):
        return True

    differences = 0
    for left, right in zip(original_sentences, revised_sentences):
        if _normalize_sentence(left) != _normalize_sentence(right):
            differences += 1

    return differences >= 2


def _count_repeated_openers(text: str) -> int:
    openers = [_extract_opener(sentence) for sentence in split_sentences(text)]
    counts = Counter(opener for opener in openers if opener)
    return max(counts.values(), default=0)


def _max_meta_subject_streak(subject_heads: list[str]) -> int:
    current = ""
    streak = 0
    max_streak = 0
    for subject in subject_heads:
        if subject in _META_SUBJECTS and subject == current:
            streak += 1
        elif subject in _META_SUBJECTS:
            current = subject
            streak = 1
        else:
            current = ""
            streak = 0
        max_streak = max(max_streak, streak)
    return max_streak


def _subject_heads_from_text(text: str) -> list[str]:
    heads: list[str] = []
    for sentence in split_sentences(text):
        stripped = _LEADING_IMPLICATION_RE.sub("", sentence.strip())
        subject = next((item for item in _META_SUBJECTS if stripped.startswith(item)), "")
        heads.append(subject)
    return heads


def _has_stiff_opener_sequence(text: str) -> bool:
    openers = [_extract_opener(sentence) for sentence in split_sentences(text)]
    current = ""
    streak = 0

    for opener in openers:
        if opener and opener == current:
            streak += 1
        else:
            current = opener
            streak = 1 if opener else 0

        if opener in _STIFF_OPENERS and streak >= 2:
            return True
    return False


def _extract_opener(sentence: str) -> str:
    stripped = _normalize_sentence(sentence)
    for opener in _STIFF_OPENERS:
        if stripped.startswith(opener):
            return opener
    return stripped[:4]


def _normalize_text(text: str) -> str:
    return _NORMALIZE_RE.sub("", text)


def _normalize_sentence(text: str) -> str:
    stripped = re.sub(r"[。！？?!]+$", "", text)
    return _NORMALIZE_RE.sub("", stripped)


def _deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _count_headings(text: str) -> int:
    return len(_HEADING_RE.findall(text))


def _count_code_blocks(text: str) -> int:
    return len(_FENCED_CODE_RE.findall(text))


def _count_links(text: str) -> int:
    return len(_INLINE_LINK_RE.findall(text)) + len(_AUTOLINK_RE.findall(text)) + len(
        _BARE_URL_RE.findall(text)
    )


def _compute_naturalness_score(
    template_risk: bool,
    template_warning: bool,
    repeated_subject_risk: bool,
    depth_sufficient: bool,
    effective_change: bool,
) -> int:
    score = 100
    if template_risk:
        score -= 40
    elif template_warning:
        score -= 15
    if repeated_subject_risk:
        score -= 20
    if not depth_sufficient:
        score -= 20
    if not effective_change:
        score -= 15
    return max(0, min(100, score))


def _suggested_fallback(
    decision: str,
    template_risk: bool,
    format_integrity: bool,
    core_content_integrity: bool,
) -> str:
    if not core_content_integrity:
        return "revert_to_original_and_restore_protected_content"
    if not format_integrity:
        return "revert_to_original_or_retry_with_do_not_touch_blocks_frozen"
    if decision == "reject" and template_risk:
        return "retry_with_lighter_edits_and_keep_original_when_rewrite_sounds_stiff"
    if decision == "reject":
        return "retry_only_on_rewritable_blocks_or_downgrade_to_conservative"
    return "ready_for_write_gate"


def _do_not_touch_blocks_preserved(revised: str, guidance: GuidanceReport) -> bool:
    tracked_types = {"heading", "reference", "english_block", "caption", "placeholder", "formula"}
    for block in guidance.do_not_touch_blocks:
        if block.block_type not in tracked_types:
            continue
        if block.original_text and block.original_text not in revised:
            return False
    return True


def _required_structural_action_count(mode: RewriteMode, normalized_length: int) -> int:
    if mode is RewriteMode.STRONG:
        return 2 if normalized_length >= DEFAULT_CONFIG.structural_gate_length_long else 1
    if mode is RewriteMode.BALANCED:
        return 1
    return 1


def _collect_high_value_actions(rewrite_stats: list[RewriteStats]) -> list[str]:
    seen: list[str] = []
    for stats in rewrite_stats:
        for action in stats.high_value_structural_actions:
            if action not in seen:
                seen.append(action)
    return seen


def _infer_high_value_actions_from_text(original: str, revised: str) -> list[str]:
    inferred: list[str] = []
    original_subject_streak = _max_meta_subject_streak(_subject_heads_from_text(original))
    revised_subject_streak = _max_meta_subject_streak(_subject_heads_from_text(revised))
    if original_subject_streak >= 2 and revised_subject_streak < original_subject_streak:
        inferred.append("subject_chain_compression")
    if len(split_sentences(revised)) < len(split_sentences(original)):
        inferred.append("pair_fusion")
    if _count_implication_openers(original) > _count_implication_openers(revised):
        inferred.append("conclusion_absorb")
    return inferred


def _infer_discourse_change_score(original: str, revised: str) -> int:
    score = 0
    original_sentences = split_sentences(original)
    revised_sentences = split_sentences(revised)
    if len(original_sentences) != len(revised_sentences):
        score += 3
    if _has_sentence_level_change(original, revised):
        score += 2
    if _count_implication_openers(original) > _count_implication_openers(revised):
        score += 2
    if _max_meta_subject_streak(_subject_heads_from_text(original)) > _max_meta_subject_streak(_subject_heads_from_text(revised)):
        score += 2
    if _count_repeated_openers(original) > _count_repeated_openers(revised):
        score += 1
    return score


def _infer_cluster_rewrite_score(original: str, revised: str) -> int:
    score = 0
    if len(split_sentences(original)) != len(split_sentences(revised)):
        score += 1
    if _count_implication_openers(original) > _count_implication_openers(revised):
        score += 1
    if _max_meta_subject_streak(_subject_heads_from_text(original)) > _max_meta_subject_streak(_subject_heads_from_text(revised)):
        score += 1
    return score


def _infer_style_variation_score(text: str) -> int:
    openers = [_extract_opener(sentence) for sentence in split_sentences(text)]
    unique_openers = len({opener for opener in openers if opener})
    repeated = _count_repeated_openers(text)
    blacklist = _count_blacklisted_templates(text)
    return max(0, 8 + unique_openers - repeated * 2 - blacklist * 2)


def _style_variation_score_from_stats(
    selected_variants: list[str],
    repeated_rule_families: int,
    template_family_counts: Counter[str],
    max_variant_repeat: int,
    revised: str,
) -> int:
    unique_variants = len(set(selected_variants))
    repeated_template = max(template_family_counts.values(), default=0)
    base = 8 + min(unique_variants, 4)
    base -= max(0, repeated_rule_families - 1) * 2
    base -= max(0, max_variant_repeat - 1) * 2
    base -= max(0, repeated_template - 1) * 2
    base -= _count_blacklisted_templates(revised) * 2
    return max(0, base)


def _looks_like_prefix_only_change(original: str, revised: str) -> bool:
    original_sentences = split_sentences(original)
    revised_sentences = split_sentences(revised)
    if len(original_sentences) != len(revised_sentences) or not original_sentences:
        return False

    changed = 0
    for left, right in zip(original_sentences, revised_sentences):
        if _normalize_sentence(left) == _normalize_sentence(right):
            continue
        changed += 1
        left_body = _LEADING_IMPLICATION_RE.sub("", left.strip())
        right_body = _LEADING_IMPLICATION_RE.sub("", right.strip())
        left_body = re.sub(r"^(?:与此同时|同时|此外|另外|在这一过程中)[，,\s]*", "", left_body)
        right_body = re.sub(r"^(?:与此同时|同时|此外|另外|在这一过程中)[，,\s]*", "", right_body)
        if _normalize_sentence(left_body) != _normalize_sentence(right_body):
            return False
    return changed > 0


def _count_implication_openers(text: str) -> int:
    return sum(
        1 for sentence in split_sentences(text) if _LEADING_IMPLICATION_RE.match(sentence.strip())
    )


def _validate_block_obligations(
    guidance: GuidanceReport,
    candidates: list[BlockRewriteCandidate],
) -> tuple[list[str], list[int]]:
    if not candidates:
        return [], []

    candidate_by_id = {candidate.block_id: candidate for candidate in candidates}
    problems: list[str] = []
    failed_block_ids: list[int] = []
    for policy in guidance.block_policies:
        if policy.edit_policy not in {"light_edit", "rewritable"}:
            continue
        if not policy.required_structural_actions:
            continue
        candidate = candidate_by_id.get(policy.block_id)
        if candidate is None:
            problems.append(
                f"Block {policy.block_id} is missing a rewritten candidate for required rewrite obligations."
            )
            failed_block_ids.append(policy.block_id)
            continue
        if not candidate.required_actions_met:
            if (
                policy.rewrite_depth == "developmental_rewrite"
                and candidate.effective_change
                and (candidate.sentence_level_changes >= 1 or candidate.cluster_changes >= 1)
            ):
                continue
            problems.append(
                f"Block {policy.block_id} did not satisfy required rewrite obligations: {candidate.missing_required_actions}."
            )
            failed_block_ids.append(policy.block_id)
    return problems, failed_block_ids


def _count_blacklisted_templates(text: str) -> int:
    count = 0
    if "在此基础上" in text:
        count += text.count("在此基础上")
    if "这也意味着" in text:
        count += text.count("这也意味着")
    if "进一步来看" in text:
        count += text.count("进一步来看")
    if "综上" in text:
        count += text.count("综上")
    count += len(re.findall(r"从[^。；，]{1,20}角度看", text))
    return count


ReviewResult = ReviewReport
