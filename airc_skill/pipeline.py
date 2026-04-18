from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from .chunker import chunk_text
from .config import DEFAULT_CONFIG, RewriteMode, fallback_modes, get_skill_preset
from .core_guard import collect_protection_stats, protect_core_content, restore_core_content
from .guidance import guide_document_text
from .input_normalizer import (
    InputNormalizationError,
    InputNormalizationReport,
    cleanup_normalized_file,
    normalize_to_markdown,
)
from .io_utils import build_output_path, generate_diff, read_text_file, write_text_file
from .markdown_guard import protect, restore
from .models import (
    BlockPolicy,
    GuidanceReport,
    RewriteCandidate as BlockRewriteCandidate,
    RewriteExecutionReport,
    ReviewReport,
    WriteGateDecision,
)
from .reviewer import ReviewResult, review_rewrite, review_revision
from .rewriter import RewriteStats, Rewriter
from .skill_protocol import (
    SkillExecutionPlan,
    SkillInputSchema,
    SkillOutputSchema,
    build_execution_plan,
)
from .suggester import Suggestion, generate_suggestions
from .validator import ValidatedFile, validate_input_file

_SUBJECT_CHAIN_ACTIONS = {
    "merge_consecutive_subject_sentences",
    "subject_drop",
    "subject_variation",
    "meta_compression",
    "followup_absorb",
    "conclusion_absorb",
}
_WRITE_PASS_DECISIONS = {"pass", "pass_with_minor_risk"}


@dataclass(frozen=True)
class RewriteResult:
    source_path: Path
    output_path: Path
    text: str
    mode_used: RewriteMode
    review: ReviewReport
    requested_mode: RewriteMode
    dry_run: bool
    effective_change: bool
    output_written: bool
    selected_candidate_reason: str
    skipped_write_reason: str | None
    candidate_count: int
    candidate_scores: list[str]
    diff: str
    debug_log: list[str]
    guidance: GuidanceReport
    write_gate: WriteGateDecision
    write_gate_decision: str
    changed_block_ids: list[int]
    block_failures: list[str]


@dataclass(frozen=True)
class CandidateRecord:
    label: str
    mode: RewriteMode
    pass_index: int
    rewrite_report: RewriteExecutionReport
    review: ReviewReport
    write_gate: WriteGateDecision
    score: tuple[int, ...]
    guidance: GuidanceReport


@dataclass(frozen=True)
class SuggestionResult:
    source_path: Path
    suggestions: list[Suggestion]


@dataclass(frozen=True)
class ReviewFileResult:
    original_path: Path
    candidate_path: Path
    guidance: GuidanceReport
    rewrite_report: RewriteExecutionReport
    review: ReviewReport
    write_gate: WriteGateDecision
    diff: str


@dataclass(frozen=True)
class WriteResult:
    original_path: Path
    candidate_path: Path
    output_path: Path
    guidance: GuidanceReport
    rewrite_report: RewriteExecutionReport
    review: ReviewReport
    write_gate: WriteGateDecision
    output_written: bool
    skipped_write_reason: str | None
    diff: str
    dry_run: bool


@dataclass(frozen=True)
class PublicRunResult:
    source_path: Path
    output_path: Path
    report_path: Path
    input_normalization: InputNormalizationReport
    output_written: bool
    report_written: bool
    rewrite_result: RewriteResult
    execution_plan: SkillExecutionPlan
    output_schema: SkillOutputSchema
    agent_instructions: str


def rewrite_file(
    path: str | Path,
    mode: RewriteMode = RewriteMode.BALANCED,
    dry_run: bool = False,
    output_path: str | Path | None = None,
    rewriter: Rewriter | None = None,
    debug_rewrite: bool = False,
    allow_low_quality_write: bool = False,
    strict_mode: bool = True,
) -> RewriteResult:
    if allow_low_quality_write:
        raise ValueError("--allow-low-quality-write is disabled in the strict architecture.")
    validated = validate_input_file(path, max_size_bytes=DEFAULT_CONFIG.max_file_size_bytes)
    original_text = read_text_file(validated)
    rewriter_instance = rewriter or Rewriter()
    protection_stats = collect_protection_stats(original_text, validated.suffix)
    guidance = guide_document_text(
        original_text,
        metadata={"suffix": validated.suffix, "source_path": validated.path},
        max_chars=DEFAULT_CONFIG.max_chunk_chars,
    )

    debug_log: list[str] = [
        f"frozen_heading_blocks={protection_stats.frozen_heading_blocks}",
        f"frozen_formula_blocks={protection_stats.frozen_formula_blocks}",
        f"frozen_english_blocks={protection_stats.frozen_english_blocks}",
        f"frozen_placeholder_blocks={protection_stats.frozen_placeholder_blocks}",
        f"frozen_caption_blocks={protection_stats.frozen_caption_blocks}",
        f"guidance_risk_level={guidance.document_risk}",
        f"guidance_do_not_touch_count={len(guidance.do_not_touch_blocks)}",
        f"guidance_high_risk_count={len(guidance.high_risk_blocks)}",
        f"guidance_light_edit_count={len(guidance.light_edit_blocks)}",
        f"guidance_rewritable_count={len(guidance.rewritable_blocks)}",
    ]
    debug_log.extend(_format_guidance_debug(guidance))

    baseline_review = review_rewrite(
        original=original_text,
        revised=original_text,
        guidance=guidance,
        mode=RewriteMode.CONSERVATIVE,
        rewrite_stats=[],
        suffix=validated.suffix,
    )
    baseline_rewrite = RewriteExecutionReport(
        rewritten_text=original_text,
        block_candidates=[],
        rewrite_stats=[],
        mode_requested=mode,
        mode_used=mode,
        effective_change=False,
        changed_block_ids=[],
        candidate_count=0,
        selected_candidate_reason="Original text kept as the baseline candidate.",
        convenience_mode=True,
        block_failures=[],
        reviewed=True,
    )
    baseline_gate = decide_write_gate(baseline_review, baseline_rewrite, {"strict_mode": strict_mode})
    candidates: list[CandidateRecord] = [
        CandidateRecord(
            label="original",
            mode=mode,
            pass_index=0,
            rewrite_report=baseline_rewrite,
            review=baseline_review,
            write_gate=baseline_gate,
            score=_score_candidate(baseline_review, baseline_gate),
            guidance=guidance,
        )
    ]

    candidate_modes = [mode] if strict_mode else fallback_modes(mode)
    for candidate_mode in candidate_modes:
        for pass_index in range(1, DEFAULT_CONFIG.max_rewrite_passes + 1):
            if pass_index > 1 and candidate_mode is RewriteMode.CONSERVATIVE:
                break

            attempt_guidance = guidance
            if pass_index > 1:
                attempt_guidance = _escalate_guidance_for_retry(guidance, candidate_mode)
                debug_log.append(
                    "retry-guidance-escalated "
                    f"mode={candidate_mode.value} pass={pass_index} "
                    "rewrite_intensity=high required=sentence_cluster_rewrite,narrative_flow_rebuilder"
                )

            debug_log.append(f"rewrite-attempt mode={candidate_mode.value} pass={pass_index}")
            rewrite_report = agent_rewrite_from_guidance(
                original_text,
                guidance=attempt_guidance,
                mode=candidate_mode,
                suffix=validated.suffix,
                rewriter=rewriter_instance,
                pass_index=pass_index,
                convenience_mode=True,
            )
            debug_log.extend(_format_block_rewrite_debug(rewrite_report, candidate_mode, pass_index))

            review = review_rewrite(
                original=original_text,
                revised=rewrite_report.rewritten_text,
                guidance=attempt_guidance,
                mode=candidate_mode,
                rewrite_stats=rewrite_report.rewrite_stats,
                block_candidates=rewrite_report.block_candidates,
                suffix=validated.suffix,
            )
            reviewed_report = replace(rewrite_report, reviewed=True)
            write_gate = decide_write_gate(review, reviewed_report, {"strict_mode": strict_mode})
            candidates.append(
                CandidateRecord(
                    label=f"{candidate_mode.value}-pass{pass_index}",
                    mode=candidate_mode,
                    pass_index=pass_index,
                    rewrite_report=reviewed_report,
                    review=review,
                    write_gate=write_gate,
                    score=_score_candidate(review, write_gate),
                    guidance=attempt_guidance,
                )
            )
            debug_log.append(
                "review-result "
                f"mode={candidate_mode.value} pass={pass_index} "
                f"decision={review.decision} "
                f"write_gate={write_gate.decision} "
                f"meaningful_change={review.meaningful_change} "
                f"effective_change={review.effective_change} "
                f"sentence_level_change={review.sentence_level_change} "
                f"changed_characters={review.changed_characters} "
                f"diff_spans={review.diff_spans} "
                f"depth_sufficient={review.depth_sufficient} "
                f"structural_action_count={review.structural_action_count} "
                f"high_value_action_count={review.high_value_action_count} "
                f"rewrite_coverage={review.rewrite_coverage:.2f} "
                f"discourse_change_score={review.discourse_change_score} "
                f"cluster_rewrite_score={review.cluster_rewrite_score} "
                f"prefix_only_rewrite={review.prefix_only_rewrite} "
                f"repeated_subject_risk={review.repeated_subject_risk} "
                f"template_risk={review.template_risk} "
                f"template_warning={review.template_warning} "
                f"template_issue={review.template_issue or 'none'} "
                f"naturalness_risk={review.naturalness_risk} "
                f"problems={review.problems or ['none']}"
            )

            if (
                write_gate.write_allowed
                and review.decision in _WRITE_PASS_DECISIONS
                and review.effective_change
                and not _needs_retry_escalation(review, candidate_mode)
            ):
                break

            if pass_index == 1 and candidate_mode in {RewriteMode.BALANCED, RewriteMode.STRONG}:
                debug_log.append(
                    f"second-pass-triggered mode={candidate_mode.value} reason={review.problems or ['review-failed']}"
                )
        latest = candidates[-1]
        if (
            latest.write_gate.write_allowed
            and latest.review.decision in _WRITE_PASS_DECISIONS
            and latest.review.effective_change
            and not _needs_retry_escalation(latest.review, latest.mode)
        ):
            break

    selected_candidate, selected_candidate_reason = _select_candidate(candidates)
    selected_review = selected_candidate.review if selected_candidate else baseline_review
    selected_rewrite = selected_candidate.rewrite_report if selected_candidate else baseline_rewrite
    selected_gate = selected_candidate.write_gate if selected_candidate else baseline_gate
    selected_mode = selected_candidate.mode if selected_candidate else mode
    selected_guidance = selected_candidate.guidance if selected_candidate else guidance
    candidate_scores = [_format_candidate_score(candidate) for candidate in candidates]

    output_written = False
    skipped_write_reason: str | None = None
    selected_text = selected_rewrite.rewritten_text

    if not selected_gate.write_allowed:
        skipped_write_reason = _reason_text_from_gate(selected_gate)
    elif dry_run:
        skipped_write_reason = "Dry-run enabled."
    else:
        output_written = True

    target_path = Path(output_path).expanduser().resolve() if output_path else build_output_path(validated.path)
    diff = generate_diff(
        original_text,
        selected_text,
        source_name=validated.path.name,
        revised_name=target_path.name,
    )

    if output_written and not dry_run:
        write_text_file(target_path, selected_text, encoding=validated.encoding)

    debug_log.append(f"candidate_count={len(candidates)}")
    debug_log.append(f"candidate_scores={candidate_scores}")
    debug_log.append(f"selected_candidate_reason={selected_candidate_reason}")
    debug_log.append(f"effective_change={selected_review.effective_change}")
    debug_log.append(f"output_written={output_written and not dry_run}")
    debug_log.append(f"skipped_write_reason={skipped_write_reason or 'none'}")
    if selected_review.failed_block_ids:
        debug_log.append(f"failed_block_ids={selected_review.failed_block_ids}")

    final_guidance = selected_guidance.with_review(selected_review, output_written and not dry_run)
    return RewriteResult(
        source_path=validated.path,
        output_path=target_path,
        text=selected_text,
        mode_used=selected_mode,
        review=selected_review,
        requested_mode=mode,
        dry_run=dry_run,
        effective_change=selected_review.effective_change,
        output_written=output_written and not dry_run,
        selected_candidate_reason=selected_candidate_reason,
        skipped_write_reason=skipped_write_reason,
        candidate_count=len(candidates),
        candidate_scores=candidate_scores,
        diff=diff,
        debug_log=debug_log if debug_rewrite else [],
        guidance=final_guidance,
        write_gate=selected_gate,
        write_gate_decision=selected_gate.decision,
        changed_block_ids=selected_rewrite.changed_block_ids,
        block_failures=selected_rewrite.block_failures,
    )


def suggest_file(path: str | Path) -> SuggestionResult:
    validated = validate_input_file(path, max_size_bytes=DEFAULT_CONFIG.max_file_size_bytes)
    text = read_text_file(validated)
    suggestions = generate_suggestions(text, suffix=validated.suffix)
    return SuggestionResult(source_path=validated.path, suggestions=suggestions)


def guide_file(path: str | Path) -> GuidanceReport:
    validated = validate_input_file(path, max_size_bytes=DEFAULT_CONFIG.max_file_size_bytes)
    text = read_text_file(validated)
    return guide_document_text(
        text,
        metadata={"suffix": validated.suffix, "source_path": validated.path},
        max_chars=DEFAULT_CONFIG.max_chunk_chars,
    )


def rewrite_block_with_guidance(
    block: str,
    guidance: BlockPolicy,
    mode: RewriteMode,
    context: dict[str, object] | None = None,
) -> BlockRewriteCandidate:
    candidate, _ = _rewrite_block_with_guidance_stats(block, guidance, mode, context or {})
    return candidate


def agent_rewrite_from_guidance(
    text: str,
    guidance: GuidanceReport,
    mode: RewriteMode,
    suffix: str,
    rewriter: Rewriter | None = None,
    pass_index: int = 1,
    convenience_mode: bool = False,
) -> RewriteExecutionReport:
    rewriter_instance = rewriter or Rewriter()
    rewriter_instance.reset_document_state()

    working_text = text
    markdown_placeholders: dict[str, str] = {}
    core_placeholders: dict[str, str] = {}

    if suffix == ".md":
        working_text, markdown_placeholders = protect(text)
    working_text, core_placeholders = protect_core_content(working_text, suffix=suffix)

    chunks = chunk_text(
        working_text,
        suffix=suffix,
        max_chars=DEFAULT_CONFIG.max_chunk_chars,
    )

    rewritten_chunks: list[str] = []
    block_candidates: list[BlockRewriteCandidate] = []
    rewrite_stats: list[RewriteStats] = []
    changed_block_ids: list[int] = []
    block_failures: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        block_policy = guidance.block_policies[index - 1] if index - 1 < len(guidance.block_policies) else None
        if not chunk.rewritable or block_policy is None or not _should_rewrite_policy(block_policy):
            rewritten_chunks.append(chunk.text)
            continue

        candidate, stats = _rewrite_block_with_guidance_stats(
            chunk.text,
            block_policy,
            mode,
            {"pass_index": pass_index, "rewriter": rewriter_instance},
        )
        rewritten_chunks.append(candidate.rewritten_text if candidate.effective_change else chunk.text)
        block_candidates.append(candidate)
        rewrite_stats.append(stats)
        if candidate.effective_change:
            changed_block_ids.append(index)
        if not candidate.required_actions_met:
            if not _can_defer_block_obligation_to_coverage(candidate, block_policy):
                block_failures.append(
                    f"Block {candidate.block_id} missing required rewrite obligations: {candidate.missing_required_actions}"
                )

    rewritten_text = "".join(rewritten_chunks)
    rewritten_text = restore_core_content(rewritten_text, core_placeholders)
    if suffix == ".md":
        rewritten_text = restore(rewritten_text, markdown_placeholders)

    selected_candidate_reason = (
        "Rewrote only blocks that guidance marked as light_edit or rewritable."
        if changed_block_ids
        else "No block exceeded the keep-original threshold under the current guidance."
    )
    return RewriteExecutionReport(
        rewritten_text=rewritten_text,
        block_candidates=block_candidates,
        rewrite_stats=rewrite_stats,
        mode_requested=mode,
        mode_used=mode,
        effective_change=rewritten_text != text,
        changed_block_ids=changed_block_ids,
        candidate_count=len(block_candidates),
        selected_candidate_reason=selected_candidate_reason,
        convenience_mode=convenience_mode,
        block_failures=block_failures,
        reviewed=False,
    )


def review_file(
    original_path: str | Path,
    candidate_path: str | Path,
    mode: RewriteMode = RewriteMode.BALANCED,
) -> ReviewFileResult:
    original_validated = validate_input_file(original_path, max_size_bytes=DEFAULT_CONFIG.max_file_size_bytes)
    candidate_validated = validate_input_file(candidate_path, max_size_bytes=DEFAULT_CONFIG.max_file_size_bytes)
    if original_validated.suffix != candidate_validated.suffix:
        raise ValueError("Original and candidate files must have the same suffix.")

    original_text = read_text_file(original_validated)
    candidate_text = read_text_file(candidate_validated)
    guidance = guide_document_text(
        original_text,
        metadata={"suffix": original_validated.suffix, "source_path": original_validated.path},
        max_chars=DEFAULT_CONFIG.max_chunk_chars,
    )
    rewrite_report = RewriteExecutionReport(
        rewritten_text=candidate_text,
        block_candidates=[],
        rewrite_stats=[],
        mode_requested=mode,
        mode_used=mode,
        effective_change=candidate_text != original_text,
        changed_block_ids=[],
        candidate_count=1 if candidate_text != original_text else 0,
        selected_candidate_reason="Reviewed agent-supplied candidate against the current guidance.",
        convenience_mode=False,
        block_failures=[],
        reviewed=False,
    )
    review = review_rewrite(
        original=original_text,
        revised=candidate_text,
        guidance=guidance,
        mode=mode,
        rewrite_stats=[],
        block_candidates=[],
        suffix=original_validated.suffix,
    )
    reviewed_report = replace(rewrite_report, reviewed=True)
    write_gate = decide_write_gate(review, reviewed_report, {"strict_mode": True})
    diff = generate_diff(
        original_text,
        candidate_text,
        source_name=original_validated.path.name,
        revised_name=candidate_validated.path.name,
    )
    return ReviewFileResult(
        original_path=original_validated.path,
        candidate_path=candidate_validated.path,
        guidance=guidance,
        rewrite_report=reviewed_report,
        review=review,
        write_gate=write_gate,
        diff=diff,
    )


def write_file(
    original_path: str | Path,
    candidate_path: str | Path,
    output_path: str | Path | None = None,
    mode: RewriteMode = RewriteMode.BALANCED,
    dry_run: bool = False,
) -> WriteResult:
    reviewed = review_file(
        original_path=original_path,
        candidate_path=candidate_path,
        mode=mode,
    )
    target_path = Path(output_path).expanduser().resolve() if output_path else build_output_path(reviewed.original_path)
    output_written = False
    skipped_write_reason: str | None = None

    if not reviewed.write_gate.write_allowed:
        skipped_write_reason = _reason_text_from_gate(reviewed.write_gate)
    elif dry_run:
        skipped_write_reason = "Dry-run enabled."
    else:
        write_text_file(target_path, reviewed.rewrite_report.rewritten_text)
        output_written = True

    return WriteResult(
        original_path=reviewed.original_path,
        candidate_path=reviewed.candidate_path,
        output_path=target_path,
        guidance=reviewed.guidance,
        rewrite_report=reviewed.rewrite_report,
        review=reviewed.review,
        write_gate=reviewed.write_gate,
        output_written=output_written,
        skipped_write_reason=skipped_write_reason,
        diff=reviewed.diff,
        dry_run=dry_run,
    )


def run_file(
    path: str | Path,
    preset: str = "academic_natural",
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    dry_run: bool = False,
    debug_rewrite: bool = False,
    keep_intermediate: bool = False,
) -> PublicRunResult:
    original_path = Path(path).expanduser().resolve()
    preset_config = get_skill_preset(preset)
    schema = SkillInputSchema.from_path(original_path, preset=preset_config.name)
    try:
        normalization = normalize_to_markdown(original_path, keep_intermediate=keep_intermediate)
    except InputNormalizationError as exc:
        _write_normalization_failure_report(
            original_path=original_path,
            report_path=report_path,
            schema=schema,
            normalization=exc.report,
            error=str(exc),
            dry_run=dry_run,
        )
        raise

    normalized_path = Path(normalization.normalized_path) if normalization.normalized_path else original_path
    effective_output_path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else _build_public_output_path(original_path)
    )
    try:
        result = rewrite_file(
            path=normalized_path,
            mode=preset_config.mode,
            dry_run=dry_run,
            output_path=effective_output_path,
            debug_rewrite=debug_rewrite,
            strict_mode=True,
        )
    finally:
        if not keep_intermediate:
            cleanup_normalized_file(normalization)

    execution_plan = build_execution_plan(result.guidance, schema)
    target_report_path = (
        Path(report_path).expanduser().resolve()
        if report_path
        else _build_rewrite_report_path(result.output_path)
    )
    attempted_ids = [block.block_id for block in result.guidance.rewrite_candidate_blocks]
    skipped_ids = [block_id for block_id in attempted_ids if block_id not in set(result.changed_block_ids)]
    output_schema = SkillOutputSchema(
        rewritten_file_path=str(result.output_path) if result.output_written else None,
        rewrite_report_path=str(target_report_path),
        rewrite_coverage=result.review.rewrite_coverage,
        discourse_change_score=result.review.discourse_change_score,
        cluster_rewrite_score=result.review.cluster_rewrite_score,
        blocks_changed=list(result.changed_block_ids),
        blocks_skipped=skipped_ids,
        warnings=[*result.review.problems, *result.review.warnings, *result.write_gate.warnings],
        write_allowed=result.write_gate.write_allowed,
        decision=result.write_gate.decision,
        reason_codes=list(result.write_gate.reason_codes),
        input_normalization=normalization.to_dict(),
    )
    report_payload = _build_public_run_report(
        schema=schema,
        execution_plan=execution_plan,
        output_schema=output_schema,
        result=result,
    )
    report_written = False
    if not dry_run:
        write_text_file(
            target_report_path,
            json.dumps(_json_safe(report_payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report_written = True

    return PublicRunResult(
        source_path=original_path,
        output_path=result.output_path,
        report_path=target_report_path,
        input_normalization=normalization,
        output_written=result.output_written,
        report_written=report_written,
        rewrite_result=result,
        execution_plan=execution_plan,
        output_schema=output_schema,
        agent_instructions=execution_plan.agent_instruction,
    )


def decide_write_gate(
    review_report: ReviewReport,
    rewrite_report: RewriteExecutionReport,
    policy: dict[str, object] | None = None,
) -> WriteGateDecision:
    policy = policy or {}
    reason_codes: list[str] = []
    warnings = list(review_report.warnings)

    if not rewrite_report.reviewed:
        reason_codes.append("unreviewed_candidate")
        return WriteGateDecision(
            write_allowed=False,
            decision="reject",
            reason_codes=reason_codes,
            warnings=warnings,
            selected_candidate_reason=rewrite_report.selected_candidate_reason,
        )
    if review_report.decision != "pass":
        if review_report.decision == "pass_with_minor_risk":
            warnings.extend(review_report.problems)
        else:
            reason_codes.extend(review_report.problems or ["review_reject"])
            return WriteGateDecision(
                write_allowed=False,
                decision="reject",
                reason_codes=reason_codes,
                warnings=warnings,
                selected_candidate_reason=rewrite_report.selected_candidate_reason,
            )

    if review_report.structural_action_count < _required_structural_action_threshold(review_report, rewrite_report):
        reason_codes.append("structural_action_count_below_threshold")
    if review_report.high_value_action_count < DEFAULT_CONFIG.strict_min_high_value_actions:
        reason_codes.append("missing_high_value_action")
    if review_report.rewrite_coverage < DEFAULT_CONFIG.rewrite_coverage_minor_threshold:
        reason_codes.append("rewrite_coverage_below_write_threshold")
    if rewrite_report.block_failures:
        if (
            review_report.rewrite_coverage >= DEFAULT_CONFIG.rewrite_coverage_pass_threshold
            and review_report.discourse_change_score >= DEFAULT_CONFIG.developmental_min_discourse_score
        ):
            warnings.extend(rewrite_report.block_failures)
        else:
            reason_codes.extend(rewrite_report.block_failures)

    if reason_codes:
        return WriteGateDecision(
            write_allowed=False,
            decision="reject",
            reason_codes=reason_codes,
            warnings=warnings,
            selected_candidate_reason=rewrite_report.selected_candidate_reason,
        )

    return WriteGateDecision(
        write_allowed=True,
        decision=review_report.decision,
        reason_codes=["review_passed"],
        warnings=warnings,
        selected_candidate_reason=rewrite_report.selected_candidate_reason,
    )


def _rewrite_block_with_guidance_stats(
    block: str,
    guidance: BlockPolicy,
    mode: RewriteMode,
    context: dict[str, object],
) -> tuple[BlockRewriteCandidate, RewriteStats]:
    rewriter = context.get("rewriter")
    if rewriter is None or not hasattr(rewriter, "rewrite"):
        rewriter = Rewriter()
    pass_index = int(context.get("pass_index", 1))

    if guidance.edit_policy in {"do_not_touch", "high_risk"} or not guidance.should_rewrite:
        candidate = BlockRewriteCandidate(
            block_id=guidance.block_id,
            original_text=block,
            rewritten_text=block,
            actions_used=["keep_original"],
            discourse_actions_used=[],
            protected_items_respected=True,
            structural_actions=[],
            high_value_actions=[],
            template_family_usage=_template_family_usage(block),
            subject_chain_actions=[],
            effective_change=False,
            sentence_level_changes=0,
            cluster_changes=0,
            required_actions_met=not guidance.required_structural_actions,
            missing_required_actions=list(guidance.required_structural_actions),
            notes=[*guidance.notes, "Guidance blocked direct rewriting for this block."],
            mode_used=guidance.edit_policy,
        )
        empty_stats = RewriteStats(
            mode=mode,
            changed=False,
            applied_rules=[],
            sentence_count_before=0,
            sentence_count_after=0,
            sentence_level_change=False,
            changed_characters=0,
            original_sentences=[],
            rewritten_sentences=[],
            paragraph_char_count=len(block.strip()),
            sentence_labels=[],
            subject_heads=[],
            detected_patterns=[],
            structural_actions=[],
            structural_action_count=0,
            high_value_structural_actions=[],
            discourse_actions_used=[],
            sentence_level_changes=0,
            cluster_changes=0,
            discourse_change_score=0,
            rewrite_coverage=0.0,
            prefix_only_rewrite=False,
            repeated_subject_risk=False,
            selected_variants=[],
            candidate_notes=[],
            paragraph_index=guidance.block_id,
            rewrite_depth=guidance.rewrite_depth,
            rewrite_intensity=guidance.rewrite_intensity,
        )
        return candidate, empty_stats

    effective_mode = _mode_for_policy(mode, guidance)
    try:
        rewritten_text, stats = rewriter.rewrite(
            block,
            mode=effective_mode,
            pass_index=pass_index,
            rewrite_depth=guidance.rewrite_depth,
            rewrite_intensity=guidance.rewrite_intensity,
        )
    except TypeError:
        rewritten_text, stats = rewriter.rewrite(
            block,
            mode=effective_mode,
            pass_index=pass_index,
    )
    stats.block_id = guidance.block_id
    stats.rewrite_depth = guidance.rewrite_depth
    stats.rewrite_intensity = guidance.rewrite_intensity
    protected_respected = _protected_items_respected(block, rewritten_text, guidance.protected_items)
    subject_chain_actions = [action for action in stats.structural_actions if action in _SUBJECT_CHAIN_ACTIONS]
    actions_used = _deduplicate([*guidance.recommended_actions, *stats.applied_rules])
    high_value_actions = [action for action in stats.high_value_structural_actions]
    discourse_actions_used = list(stats.discourse_actions_used)
    missing_required_actions = _missing_required_actions(
        guidance.required_structural_actions,
        stats.structural_actions,
        guidance.required_discourse_actions,
        discourse_actions_used,
        guidance.required_minimum_sentence_level_changes,
        stats.sentence_level_changes,
        guidance.required_minimum_cluster_changes,
        stats.cluster_changes,
    )
    notes = list(guidance.notes)
    if effective_mode != mode:
        notes.append(f"Mode was lowered from {mode.value} to {effective_mode.value} because of block policy.")
    if not stats.changed and "keep_original_if_rewrite_would_be_stiff" in guidance.recommended_actions:
        notes.append("Original sentence was kept because the available rewrite would be too stiff.")

    candidate = BlockRewriteCandidate(
        block_id=guidance.block_id,
        original_text=block,
        rewritten_text=rewritten_text,
        actions_used=actions_used,
        discourse_actions_used=discourse_actions_used,
        protected_items_respected=protected_respected,
        structural_actions=list(stats.structural_actions),
        high_value_actions=high_value_actions,
        template_family_usage=_template_family_usage(rewritten_text),
        subject_chain_actions=subject_chain_actions,
        effective_change=rewritten_text != block,
        sentence_level_changes=stats.sentence_level_changes,
        cluster_changes=stats.cluster_changes,
        required_actions_met=not missing_required_actions,
        missing_required_actions=missing_required_actions,
        notes=_deduplicate([*notes, *stats.candidate_notes]),
        mode_used=effective_mode.value,
    )
    return candidate, stats


def _should_rewrite_policy(block_policy: BlockPolicy) -> bool:
    if block_policy.edit_policy in {"do_not_touch", "high_risk"}:
        return False
    return block_policy.should_rewrite


def _escalate_guidance_for_retry(guidance: GuidanceReport, mode: RewriteMode) -> GuidanceReport:
    escalated_blocks: list[BlockPolicy] = []
    for block in guidance.block_policies:
        if not block.should_rewrite or block.edit_policy in {"do_not_touch", "high_risk"}:
            escalated_blocks.append(block)
            continue

        discourse_actions = list(block.required_discourse_actions)
        structural_actions = list(block.required_structural_actions)
        recommended_actions = list(block.recommended_actions)
        notes = list(block.notes)
        intensity = block.rewrite_intensity
        min_sentence = block.required_minimum_sentence_level_changes
        min_cluster = block.required_minimum_cluster_changes

        if block.rewrite_depth == "developmental_rewrite":
            intensity = "high"
            discourse_actions = _deduplicate(
                [*discourse_actions, "sentence_cluster_rewrite", "proposition_reorder", "transition_absorption"]
            )
            structural_actions = _deduplicate([*structural_actions, "pair_fusion"])
            recommended_actions = _deduplicate(
                [*recommended_actions, "sentence_cluster_rewrite", "narrative_flow_rebuilder", "conclusion_absorb"]
            )
            min_sentence = max(min_sentence, DEFAULT_CONFIG.developmental_min_sentence_level_changes)
            min_cluster = max(min_cluster, DEFAULT_CONFIG.developmental_min_cluster_changes)
            notes.append("Retry escalation: use high-intensity developmental rewrite and rebuild sentence clusters.")
        elif block.edit_policy == "light_edit" and mode is not RewriteMode.CONSERVATIVE:
            intensity = "medium"
            discourse_actions = _deduplicate([*discourse_actions, "sentence_level_recast"])
            recommended_actions = _deduplicate([*recommended_actions, "light_clause_reorder", "sentence_merge_or_split_light"])
            min_sentence = max(min_sentence, DEFAULT_CONFIG.light_edit_min_sentence_level_changes)
            notes.append("Retry escalation: require at least one sentence-level change while preserving light-edit limits.")

        escalated_blocks.append(
            replace(
                block,
                rewrite_intensity=intensity,
                required_structural_actions=structural_actions,
                required_discourse_actions=discourse_actions,
                required_minimum_sentence_level_changes=min_sentence,
                required_minimum_cluster_changes=min_cluster,
                recommended_actions=recommended_actions,
                optional_actions=[
                    action
                    for action in recommended_actions
                    if action not in structural_actions and action not in discourse_actions
                ],
                notes=_deduplicate(notes),
            )
        )

    return replace(
        guidance,
        block_policies=escalated_blocks,
        do_not_touch_blocks=[block for block in escalated_blocks if block.edit_policy == "do_not_touch"],
        high_risk_blocks=[block for block in escalated_blocks if block.edit_policy == "high_risk"],
        light_edit_blocks=[block for block in escalated_blocks if block.edit_policy == "light_edit"],
        rewritable_blocks=[block for block in escalated_blocks if block.edit_policy == "rewritable"],
        rewrite_actions_by_block={block.block_id: block.recommended_actions for block in escalated_blocks},
        agent_notes=_deduplicate(
            [
                *guidance.agent_notes,
                "Auto retry active: raise rewritable blocks to high intensity and force sentence-cluster rebuilding.",
            ]
        ),
    )


def _can_defer_block_obligation_to_coverage(candidate: BlockRewriteCandidate, block_policy: BlockPolicy) -> bool:
    if block_policy.rewrite_depth != "developmental_rewrite":
        return False
    return candidate.effective_change and (
        candidate.sentence_level_changes >= 1 or candidate.cluster_changes >= 1
    )


def _mode_for_policy(requested_mode: RewriteMode, block_policy: BlockPolicy) -> RewriteMode:
    if block_policy.edit_policy == "light_edit":
        if requested_mode is RewriteMode.CONSERVATIVE:
            return RewriteMode.CONSERVATIVE
        return RewriteMode.BALANCED
    return requested_mode


def _protected_items_respected(original: str, revised: str, protected_items: list[str]) -> bool:
    for item in protected_items:
        if item and item in original and item not in revised:
            return False
    return True


def _template_family_usage(text: str) -> dict[str, int]:
    family_markers = {
        "study_subject_family": ("本研究", "本文", "该研究", "研究"),
        "implication_family": ("因此", "由此", "这也意味着", "在这种情况下", "正因为如此", "在此基础上"),
        "transition_family": ("同时", "与此同时", "在这一过程中", "此外"),
        "framing_family": ("围绕", "聚焦于", "尝试回应", "核心问题", "重点在于"),
    }
    usage: dict[str, int] = {}
    for family, markers in family_markers.items():
        count = sum(text.count(marker) for marker in markers)
        if count:
            usage[family] = count
    return usage


def _format_guidance_debug(guidance: GuidanceReport) -> list[str]:
    lines: list[str] = []
    for block in guidance.block_policies:
        lines.append(
            f"guidance-block={block.block_id} kind={block.block_type} "
            f"risk={block.risk_level} policy={block.edit_policy} rewrite_depth={block.rewrite_depth} "
            f"rewrite_intensity={block.rewrite_intensity} should_rewrite={block.should_rewrite} "
            f"required={block.required_structural_actions or ['none']} "
            f"required_discourse={block.required_discourse_actions or ['none']} "
            f"min_sentence_changes={block.required_minimum_sentence_level_changes} "
            f"min_cluster_changes={block.required_minimum_cluster_changes} "
            f"optional={block.optional_actions or ['none']} "
            f"actions={block.recommended_actions or ['none']} forbidden={block.forbidden_actions or ['none']}"
        )
        if block.notes:
            lines.append(f"guidance-block={block.block_id} notes={block.notes}")
    lines.append(f"agent_notes={guidance.agent_notes or ['none']}")
    lines.append(f"write_gate_preconditions={guidance.write_gate_preconditions or ['none']}")
    return lines


def _format_block_rewrite_debug(
    rewrite_report: RewriteExecutionReport,
    mode: RewriteMode,
    pass_index: int,
) -> list[str]:
    lines: list[str] = []
    for candidate, stats in zip(rewrite_report.block_candidates, rewrite_report.rewrite_stats, strict=False):
        lines.append(
            f"block={candidate.block_id} mode={mode.value} pass={pass_index} "
            f"paragraph_index={stats.paragraph_index} paragraph_chars={stats.paragraph_char_count} "
            f"sentence_labels={stats.sentence_labels} subject_heads={stats.subject_heads}"
        )
        lines.append(
            f"block={candidate.block_id} mode={mode.value} pass={pass_index} "
            f"detected_patterns={stats.detected_patterns or ['none']}"
        )
        lines.append(
            f"block={candidate.block_id} mode={mode.value} pass={pass_index} "
            f"structural_actions={stats.structural_actions or ['none']} "
            f"structural_action_count={stats.structural_action_count} "
            f"discourse_actions_used={stats.discourse_actions_used or ['none']} "
            f"discourse_change_score={stats.discourse_change_score} "
            f"cluster_changes={stats.cluster_changes} "
            f"sentence_level_changes={stats.sentence_level_changes} "
            f"rewrite_coverage={stats.rewrite_coverage:.2f}"
        )
        lines.append(
            f"block={candidate.block_id} mode={mode.value} pass={pass_index} "
            f"subject_chain_actions={candidate.subject_chain_actions or ['none']} "
            f"discourse_actions_used={candidate.discourse_actions_used or ['none']} "
            f"sentence_level_changes={candidate.sentence_level_changes} "
            f"cluster_changes={candidate.cluster_changes} "
            f"template_family_usage={candidate.template_family_usage or {'none': 0}} "
            f"required_actions_met={candidate.required_actions_met} "
            f"missing_required_actions={candidate.missing_required_actions or ['none']} "
            f"effective_change={candidate.effective_change}"
        )
        lines.append(
            f"block={candidate.block_id} mode={mode.value} pass={pass_index} "
            f"selected_variants={stats.selected_variants or ['none']} "
            f"candidate_notes={candidate.notes or ['none']}"
        )

    if not rewrite_report.block_candidates:
        lines.append(f"block=none mode={mode.value} pass={pass_index} rewrite=skipped")

    return lines


def _score_candidate(review: ReviewReport, write_gate: WriteGateDecision) -> tuple[int, ...]:
    decision_rank = {
        "pass": 6,
        "pass_with_minor_risk": 4,
        "reject": -2,
    }.get(write_gate.decision, 0)
    return (
        int(write_gate.write_allowed),
        decision_rank,
        review.discourse_change_score,
        review.cluster_rewrite_score,
        int(review.rewrite_coverage * 100),
        int(review.effective_change),
        int(not review.template_risk),
        int(not review.repeated_subject_risk),
        review.structural_action_count,
        -len(review.problems),
        review.changed_characters,
    )


def _needs_retry_escalation(review: ReviewReport, mode: RewriteMode) -> bool:
    if mode is RewriteMode.CONSERVATIVE:
        return False
    return (
        review.rewrite_coverage < DEFAULT_CONFIG.rewrite_coverage_pass_threshold
        or review.discourse_change_score < DEFAULT_CONFIG.developmental_min_discourse_score
    )


def _select_candidate(
    candidates: list[CandidateRecord],
) -> tuple[CandidateRecord | None, str]:
    gated = [candidate for candidate in candidates if candidate.write_gate.write_allowed]
    if gated:
        selected = max(gated, key=lambda candidate: candidate.score)
        return selected, "Selected the highest-ranked candidate that passed guide -> review -> write gate."

    attempted = [candidate for candidate in candidates if candidate.pass_index > 0]
    if attempted:
        selected = max(attempted, key=lambda candidate: candidate.score)
        return selected, "Best reviewed candidate failed the write gate, so the original text was kept."

    return candidates[0] if candidates else None, "No candidate reached a writable state."


def _format_candidate_score(candidate: CandidateRecord) -> str:
    return (
        f"{candidate.label}:decision={candidate.review.decision},"
        f"gate={candidate.write_gate.decision},"
        f"write_allowed={candidate.write_gate.write_allowed},"
        f"effective_change={candidate.review.effective_change},"
        f"changed_characters={candidate.review.changed_characters},"
        f"structural_actions={candidate.review.structural_action_count},"
        f"discourse_change_score={candidate.review.discourse_change_score},"
        f"cluster_rewrite_score={candidate.review.cluster_rewrite_score},"
        f"rewrite_coverage={candidate.review.rewrite_coverage:.2f},"
        f"template_risk={candidate.review.template_risk},"
        f"repeated_subject_risk={candidate.review.repeated_subject_risk},"
        f"score={candidate.score}"
    )


def _build_public_output_path(original_path: Path) -> Path:
    return original_path.with_name(f"{original_path.stem}.airc.md")


def _write_normalization_failure_report(
    *,
    original_path: Path,
    report_path: str | Path | None,
    schema: SkillInputSchema,
    normalization: InputNormalizationReport,
    error: str,
    dry_run: bool,
) -> None:
    if dry_run or report_path is None:
        return
    target = Path(report_path).expanduser().resolve()
    payload = {
        "input_schema": asdict(schema),
        "input_normalization": normalization.to_dict(),
        "output_schema": {
            "rewritten_file_path": None,
            "rewrite_report_path": str(target),
            "input_normalization": normalization.to_dict(),
            "rewrite_coverage": 0.0,
            "discourse_change_score": 0,
            "cluster_rewrite_score": 0,
            "blocks_changed": [],
            "blocks_skipped": [],
            "warnings": [error],
            "write_allowed": False,
            "decision": "normalization_failed",
            "reason_codes": ["input_normalization_failed"],
        },
        "failure_transparency": {
            "stage": "input_normalization",
            "original_path": str(original_path),
            "current_coverage": 0.0,
            "recommended_next_step": (
                "Install pandoc or convert the input to .md/.docx first."
                if normalization.original_type != ".doc"
                else "doc support is experimental, please convert to docx first."
            ),
            "error": error,
        },
    }
    write_text_file(target, json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _build_rewrite_report_path(output_path: Path) -> Path:
    suffix = output_path.suffix
    if suffix:
        return output_path.with_name(f"{output_path.stem}.report.json")
    return output_path.with_name(f"{output_path.name}.report.json")


def _build_public_run_report(
    schema: SkillInputSchema,
    execution_plan: SkillExecutionPlan,
    output_schema: SkillOutputSchema,
    result: RewriteResult,
) -> dict[str, object]:
    return {
        "input_schema": asdict(schema),
        "input_normalization": output_schema.input_normalization,
        "execution_plan": execution_plan.to_dict(),
        "output_schema": output_schema.to_dict(),
        "review": {
            "decision": result.review.decision,
            "effective_change": result.review.effective_change,
            "rewrite_coverage": result.review.rewrite_coverage,
            "discourse_change_score": result.review.discourse_change_score,
            "cluster_rewrite_score": result.review.cluster_rewrite_score,
            "style_variation_score": result.review.style_variation_score,
            "format_integrity": result.review.format_integrity,
            "core_content_integrity": result.review.core_content_integrity,
            "failed_block_ids": result.review.failed_block_ids,
            "problems": result.review.problems,
            "warnings": result.review.warnings,
        },
        "write_gate": {
            "write_allowed": result.write_gate.write_allowed,
            "decision": result.write_gate.decision,
            "reason_codes": result.write_gate.reason_codes,
            "warnings": result.write_gate.warnings,
        },
        "failure_transparency": {
            "blocks_not_changed": output_schema.blocks_skipped,
            "block_failures": result.block_failures,
            "current_coverage": result.review.rewrite_coverage,
            "recommended_next_intensity": "high"
            if result.review.rewrite_coverage < DEFAULT_CONFIG.rewrite_coverage_pass_threshold
            else "current",
        },
        "candidate_scores": result.candidate_scores,
        "agent_instructions": execution_plan.agent_instruction,
    }


def _json_safe(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(_json_safe(key)): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _reason_text_from_gate(write_gate: WriteGateDecision) -> str:
    if "unreviewed_candidate" in write_gate.reason_codes:
        return "Candidate was not reviewed before the write gate."
    if write_gate.reason_codes:
        return "; ".join(write_gate.reason_codes)
    return "Final candidate did not pass the write threshold."


def _missing_required_actions(
    required_structural_actions: list[str],
    structural_actions: list[str],
    required_discourse_actions: list[str],
    discourse_actions: list[str],
    required_minimum_sentence_level_changes: int,
    sentence_level_changes: int,
    required_minimum_cluster_changes: int,
    cluster_changes: int,
) -> list[str]:
    missing: list[str] = []
    structural_set = set(structural_actions)
    discourse_set = set(discourse_actions)
    for action in required_structural_actions:
        allowed = _required_action_aliases(action)
        if structural_set.isdisjoint(allowed):
            missing.append(action)
    for action in required_discourse_actions:
        allowed = _required_action_aliases(action)
        if discourse_set.isdisjoint(allowed):
            missing.append(action)
    if sentence_level_changes < required_minimum_sentence_level_changes:
        missing.append(f"sentence_level_change>={required_minimum_sentence_level_changes}")
    if cluster_changes < required_minimum_cluster_changes:
        missing.append(f"cluster_change>={required_minimum_cluster_changes}")
    return missing


def _required_action_aliases(action: str) -> set[str]:
    aliases = {
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
        "clause_reorder": {"clause_reorder", "pair_fusion", "sentence_merge"},
        "sentence_cluster_rewrite": {"sentence_cluster_rewrite", "pair_fusion", "conclusion_absorb", "sentence_merge"},
        "meta_compression": {"meta_compression", "subject_chain_compression"},
        "proposition_reorder": {"proposition_reorder", "paragraph_reorder"},
        "transition_absorption": {"transition_absorption", "pair_fusion", "sentence_cluster_rewrite"},
        "rationale_expansion": {"rationale_expansion", "clause_reorder"},
    }
    return aliases.get(action, {action})


def _required_structural_action_threshold(review: ReviewReport, rewrite_report: RewriteExecutionReport) -> int:
    mode_value = rewrite_report.mode_used
    mode = mode_value if isinstance(mode_value, RewriteMode) else RewriteMode.from_value(str(mode_value))
    if mode is RewriteMode.STRONG:
        return 2
    return 1


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
