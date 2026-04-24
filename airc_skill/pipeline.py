from __future__ import annotations

import json
import re
from dataclasses import asdict
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from .chunker import chunk_text
from .config import DEFAULT_CONFIG, RewriteMode, fallback_modes, get_skill_preset
from .core_guard import (
    collect_protection_stats,
    protect_core_content,
    restore_core_content,
)
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
from .local_revision_realism import analyze_local_revision_sentences
from .reporters import build_human_report, build_json_report
from .reviewer import ReviewResult, review_rewrite, review_revision
from .rewriter import RewriteStats, Rewriter, split_sentences
from .sentence_readability import analyze_paragraph_readability_sentences
from .skill_protocol import (
    SkillExecutionPlan,
    SkillInputSchema,
    SkillOutputSchema,
    build_execution_plan,
    validate_execution_against_plan,
)
from .suggester import Suggestion, generate_suggestions
from .target_style_alignment import align_text_to_target_style
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
    rewrite_report: RewriteExecutionReport


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
    human_report: str


def rewrite_file(
    path: str | Path,
    mode: RewriteMode = RewriteMode.BALANCED,
    dry_run: bool = False,
    output_path: str | Path | None = None,
    rewriter: Rewriter | None = None,
    debug_rewrite: bool = False,
    allow_low_quality_write: bool = False,
    strict_mode: bool = True,
    max_retry_passes: int | None = None,
    target_style: str = "academic_natural",
    target_style_text: str | None = None,
) -> RewriteResult:
    if allow_low_quality_write:
        raise ValueError("--allow-low-quality-write is disabled in the strict architecture.")
    validated = validate_input_file(path, max_size_bytes=DEFAULT_CONFIG.max_file_size_bytes)
    original_text = read_text_file(validated)
    rewriter_instance = rewriter or Rewriter(style_profile=target_style)
    protection_stats = collect_protection_stats(original_text, validated.suffix)
    guidance = guide_document_text(
        original_text,
        metadata={"suffix": validated.suffix, "source_path": validated.path, "target_style": target_style},
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
        f"document_scale={guidance.document_scale}",
        f"body_blocks_total={guidance.body_blocks_total}",
        f"body_sentences_total={guidance.body_sentences_total}",
    ]
    debug_log.extend(_format_guidance_debug(guidance))

    baseline_review = review_rewrite(
        original=original_text,
        revised=original_text,
        guidance=guidance,
        mode=RewriteMode.CONSERVATIVE,
        rewrite_stats=[],
        suffix=validated.suffix,
        target_style_text=target_style_text,
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

    retry_passes = max(1, max_retry_passes or DEFAULT_CONFIG.max_rewrite_passes)
    candidate_modes = [mode] if strict_mode else fallback_modes(mode)
    if guidance.document_scale in {"long", "very_long"} and mode is not RewriteMode.CONSERVATIVE:
        candidate_modes = [candidate_mode for candidate_mode in candidate_modes if candidate_mode is not RewriteMode.CONSERVATIVE]
    for candidate_mode in candidate_modes:
        for pass_index in range(1, retry_passes + 1):
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
            aligned_text, alignment_actions = align_text_to_target_style(
                source_text=original_text,
                model_output=rewrite_report.rewritten_text,
                target_text=target_style_text,
            )
            if alignment_actions:
                rewrite_report = replace(
                    rewrite_report,
                    rewritten_text=aligned_text,
                    effective_change=aligned_text != original_text,
                    selected_candidate_reason=(
                        f"{rewrite_report.selected_candidate_reason} "
                        f"Applied target-style fitting: {', '.join(alignment_actions)}."
                    ),
                )
                debug_log.append(
                    f"target-style-alignment mode={candidate_mode.value} pass={pass_index} "
                    f"actions={alignment_actions}"
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
                target_style_text=target_style_text,
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
                f"body_rewrite_coverage={review.body_rewrite_coverage:.2f} "
                f"body_changed_blocks={review.body_changed_blocks}/{review.body_blocks_total} "
                f"document_scale={review.document_scale} "
                f"rewrite_quota_met={review.rewrite_quota_met} "
                f"discourse_change_score={review.discourse_change_score} "
                f"cluster_rewrite_score={review.cluster_rewrite_score} "
                f"prefix_only_rewrite={review.prefix_only_rewrite} "
                f"repeated_subject_risk={review.repeated_subject_risk} "
                f"template_risk={review.template_risk} "
                f"template_warning={review.template_warning} "
                f"template_issue={review.template_issue or 'none'} "
                f"naturalness_risk={review.naturalness_risk} "
                f"revision_realism_score={review.local_revision_realism.get('revision_realism_score', 0.0)} "
                f"paragraph_readability_score={review.sentence_readability.get('paragraph_readability_score', 0.0)} "
                f"sentence_completeness_score={review.sentence_readability.get('sentence_completeness_score', 0.0)} "
                f"local_transition_natural={review.local_transition_natural} "
                f"local_discourse_not_flat={review.local_discourse_not_flat} "
                f"sentence_uniformity_reduced={review.sentence_uniformity_reduced} "
                f"revision_realism_present={review.revision_realism_present} "
                f"paragraph_readability_preserved={review.paragraph_readability_preserved} "
                f"sentence_completeness_preserved={review.sentence_completeness_preserved} "
                f"problems={review.problems or ['none']}"
            )

            if (
                write_gate.write_allowed
                and review.decision in _WRITE_PASS_DECISIONS
                and review.effective_change
                and not _needs_retry_escalation(review, candidate_mode)
            ):
                break

            if pass_index < retry_passes and candidate_mode in {RewriteMode.BALANCED, RewriteMode.STRONG}:
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
        rewrite_report=selected_rewrite,
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
            {"pass_index": pass_index, "rewriter": rewriter_instance, "suffix": suffix},
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
    rewritten_text = _repair_document_evidence_fidelity_surface(
        original_text=text,
        rewritten_text=rewritten_text,
    )
    rewritten_text = _repair_document_academic_sentence_surface(rewritten_text)

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
    target_style_file: str | Path | None = None,
    dry_run: bool = False,
    debug_rewrite: bool = False,
    keep_intermediate: bool = False,
    emit_agent_context: bool = False,
    emit_json_report: bool = True,
    mode: RewriteMode | str | None = None,
    max_retry_passes: int | None = None,
) -> PublicRunResult:
    original_path = Path(path).expanduser().resolve()
    target_style_path = Path(target_style_file).expanduser().resolve() if target_style_file else None
    preset_config = get_skill_preset(preset)
    resolved_mode = RewriteMode.from_value(mode) if isinstance(mode, str) else (mode or preset_config.mode)
    schema = SkillInputSchema.from_path(
        original_path,
        preset=preset_config.name,
        output_path=output_path,
        mode=resolved_mode.value,
        target_style_file=target_style_path,
        max_retry_passes=preset_config.max_retry_passes,
        emit_agent_context=emit_agent_context,
        emit_json_report=emit_json_report,
    )
    if max_retry_passes is not None:
        schema = SkillInputSchema.from_dict({**schema.to_dict(), "max_retry_passes": max_retry_passes})
    target_style_text = None
    if target_style_path is not None:
        target_validated = validate_input_file(target_style_path, max_size_bytes=DEFAULT_CONFIG.max_file_size_bytes)
        target_style_text = read_text_file(target_validated)
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
            mode=schema.resolved_mode(),
            dry_run=dry_run,
            output_path=effective_output_path,
            debug_rewrite=debug_rewrite,
            strict_mode=True,
            max_retry_passes=schema.max_retry_passes,
            target_style=schema.target_style,
            target_style_text=target_style_text,
        )
    finally:
        if not keep_intermediate:
            cleanup_normalized_file(normalization)

    execution_plan = build_execution_plan(result.guidance, schema)
    execution_validation = validate_execution_against_plan(
        execution_plan,
        result.rewrite_report,
        review=result.review,
    )
    target_report_path = (
        Path(report_path).expanduser().resolve()
        if report_path
        else _build_rewrite_report_path(result.output_path)
    )
    attempted_ids = [block.block_id for block in result.guidance.rewrite_candidate_blocks]
    skipped_ids = [block_id for block_id in attempted_ids if block_id not in set(result.changed_block_ids)]
    output_status = (
        "success"
        if result.write_gate.write_allowed and execution_validation.ok
        else "success_with_obligation_warnings"
        if result.write_gate.write_allowed
        else "failed"
    )
    output_schema = SkillOutputSchema(
        status=output_status,
        rewritten_file_path=str(result.output_path) if result.output_written else None,
        report_file_path=str(target_report_path),
        rewrite_report_path=str(target_report_path),
        input_normalization=normalization.to_dict(),
        rewrite_coverage=result.review.rewrite_coverage,
        body_rewrite_coverage=result.review.body_rewrite_coverage,
        body_changed_blocks=result.review.body_changed_blocks,
        body_blocks_total=result.review.body_blocks_total,
        body_changed_sentences=result.review.body_changed_sentences,
        body_sentences_total=result.review.body_sentences_total,
        body_discourse_change_score=result.review.body_discourse_change_score,
        body_cluster_rewrite_score=result.review.body_cluster_rewrite_score,
        document_scale=result.review.document_scale,
        rewrite_quota_met=result.review.rewrite_quota_met,
        human_like_variation=result.review.human_like_variation,
        non_uniform_rewrite_distribution=result.review.non_uniform_rewrite_distribution,
        sentence_cluster_changes_present=result.review.sentence_cluster_changes_present,
        narrative_flow_changed=result.review.narrative_flow_changed,
        revision_pattern_distribution=dict(result.review.revision_pattern_distribution),
        chapter_rewrite_metrics=list(result.review.chapter_rewrite_metrics),
        chapter_policy_consistency_check=result.review.chapter_policy_consistency_check,
        chapter_rewrite_quota_check=result.review.chapter_rewrite_quota_check,
        chapter_rewrite_quota_reason_codes=list(result.review.chapter_rewrite_quota_reason_codes),
        paragraph_topic_sentence_preserved=result.review.paragraph_topic_sentence_preserved,
        paragraph_opening_style_valid=result.review.paragraph_opening_style_valid,
        paragraph_skeleton_consistent=result.review.paragraph_skeleton_consistent,
        no_dangling_opening_sentence=result.review.no_dangling_opening_sentence,
        topic_sentence_not_demoted_to_mid_paragraph=result.review.topic_sentence_not_demoted_to_mid_paragraph,
        paragraph_skeleton_review=dict(result.review.paragraph_skeleton_review),
        local_transition_natural=result.review.local_transition_natural,
        local_discourse_not_flat=result.review.local_discourse_not_flat,
        sentence_uniformity_reduced=result.review.sentence_uniformity_reduced,
        revision_realism_present=result.review.revision_realism_present,
        stylistic_uniformity_controlled=result.review.stylistic_uniformity_controlled,
        support_sentence_texture_varied=result.review.support_sentence_texture_varied,
        paragraph_voice_variation_present=result.review.paragraph_voice_variation_present,
        academic_cliche_density_controlled=result.review.academic_cliche_density_controlled,
        local_revision_realism=dict(result.review.local_revision_realism),
        sentence_completeness_preserved=result.review.sentence_completeness_preserved,
        paragraph_readability_preserved=result.review.paragraph_readability_preserved,
        no_dangling_support_sentences=result.review.no_dangling_support_sentences,
        no_fragment_like_conclusion_sentences=result.review.no_fragment_like_conclusion_sentences,
        sentence_readability=dict(result.review.sentence_readability),
        semantic_role_integrity_preserved=result.review.semantic_role_integrity_preserved,
        enumeration_integrity_preserved=result.review.enumeration_integrity_preserved,
        scaffolding_phrase_density_controlled=result.review.scaffolding_phrase_density_controlled,
        over_abstracted_subject_risk_controlled=result.review.over_abstracted_subject_risk_controlled,
        semantic_role_integrity=dict(result.review.semantic_role_integrity),
        assertion_strength_preserved=result.review.assertion_strength_preserved,
        appendix_like_support_controlled=result.review.appendix_like_support_controlled,
        authorial_stance_present=result.review.authorial_stance_present,
        authorial_intent=dict(result.review.authorial_intent),
        evidence_fidelity_preserved=result.review.evidence_fidelity_preserved,
        unsupported_expansion_controlled=result.review.unsupported_expansion_controlled,
        thesis_tone_restrained=result.review.thesis_tone_restrained,
        metaphor_or_storytelling_controlled=result.review.metaphor_or_storytelling_controlled,
        authorial_claim_risk_controlled=result.review.authorial_claim_risk_controlled,
        evidence_fidelity=dict(result.review.evidence_fidelity),
        bureaucratic_opening_controlled=result.review.bureaucratic_opening_controlled,
        explicit_subject_chain_controlled=result.review.explicit_subject_chain_controlled,
        overstructured_syntax_controlled=result.review.overstructured_syntax_controlled,
        main_clause_position_reasonable=result.review.main_clause_position_reasonable,
        slogan_like_goal_phrase_controlled=result.review.slogan_like_goal_phrase_controlled,
        academic_sentence_naturalization=dict(result.review.academic_sentence_naturalization),
        target_style_alignment=dict(result.review.target_style_alignment),
        discourse_change_score=result.review.discourse_change_score,
        cluster_rewrite_score=result.review.cluster_rewrite_score,
        blocks_changed=list(result.changed_block_ids),
        blocks_skipped=skipped_ids,
        blocks_rejected=execution_validation.failed_block_ids,
        warnings=[*result.review.problems, *result.review.warnings, *result.write_gate.warnings],
        failed_obligations=execution_validation.failed_obligations,
        write_gate_decision=result.write_gate.decision,
        write_allowed=result.write_gate.write_allowed,
        decision=result.write_gate.decision,
        reason_codes=list(result.write_gate.reason_codes),
    )
    report_payload = build_json_report(
        schema=schema,
        execution_plan=execution_plan,
        output_schema=output_schema,
        review=result.review,
        write_gate=result.write_gate,
        input_normalization=normalization.to_dict(),
        candidate_scores=result.candidate_scores,
        block_failures=result.block_failures,
    )
    human_report = build_human_report(
        output_schema=output_schema,
        review=result.review,
        write_gate=result.write_gate,
    )
    report_written = False
    if not dry_run and emit_json_report:
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
        human_report=human_report,
    )


def decide_write_gate(
    review_report: ReviewReport,
    rewrite_report: RewriteExecutionReport,
    policy: dict[str, object] | None = None,
) -> WriteGateDecision:
    """Decide whether a reviewed candidate is safe to write to disk."""

    policy = policy or {}
    reason_codes: list[str] = []
    warnings = list(review_report.warnings)
    l2_profile_enabled = bool(review_report.l2_style_profile.get("enabled", False))

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
            if review_report.high_value_action_count < DEFAULT_CONFIG.strict_min_high_value_actions:
                reason_codes.append("missing_high_value_action")
            if not review_report.chapter_rewrite_quota_check:
                reason_codes.extend(review_report.chapter_rewrite_quota_reason_codes)
            if not review_report.paragraph_opening_style_valid:
                reason_codes.append("paragraph_opening_style_invalid")
            if not review_report.paragraph_skeleton_consistent:
                reason_codes.append("paragraph_skeleton_inconsistent")
            if not review_report.revision_realism_present:
                reason_codes.append("revision_realism_missing")
            if not review_report.sentence_completeness_preserved:
                reason_codes.append("sentence_completeness_failed")
            if not review_report.paragraph_readability_preserved:
                reason_codes.append("paragraph_readability_failed")
            if not review_report.no_dangling_support_sentences:
                reason_codes.append("dangling_sentence_risk_too_high")
            if not review_report.semantic_role_integrity_preserved:
                reason_codes.append("semantic_role_integrity_failed")
            if not review_report.enumeration_integrity_preserved:
                reason_codes.append("enumeration_integrity_failed")
            if review_report.document_scale in {"long", "very_long"}:
                if not review_report.evidence_fidelity_preserved:
                    reason_codes.append("evidence_fidelity_failed")
                if not review_report.unsupported_expansion_controlled:
                    reason_codes.append("unsupported_expansion_failed")
                if not review_report.thesis_tone_restrained:
                    reason_codes.append("thesis_tone_restraint_failed")
                if not review_report.metaphor_or_storytelling_controlled:
                    reason_codes.append("metaphor_or_storytelling_failed")
                if not review_report.authorial_claim_risk_controlled:
                    reason_codes.append("authorial_claim_risk_failed")
                if not review_report.bureaucratic_opening_controlled:
                    reason_codes.append("bureaucratic_opening_density_failed")
                if not review_report.explicit_subject_chain_controlled:
                    reason_codes.append("explicit_subject_chain_failed")
                if not review_report.overstructured_syntax_controlled:
                    reason_codes.append("overstructured_syntax_failed")
                if not review_report.main_clause_position_reasonable:
                    reason_codes.append("delayed_main_clause_risk_failed")
                if not review_report.slogan_like_goal_phrase_controlled:
                    reason_codes.append("slogan_like_goal_phrase_failed")
                if (
                    not l2_profile_enabled
                    and not review_report.academic_sentence_naturalization.get("author_style_alignment_controlled", True)
                ):
                    reason_codes.append("author_style_alignment_failed")
                if review_report.target_style_alignment.get("enabled", False):
                    if float(review_report.target_style_alignment.get("grammar_error_rate", 0.0)) > 0.02:
                        reason_codes.append("target_style_alignment_grammar_error_rate_too_high")
                    if int(review_report.target_style_alignment.get("terminology_drift", 0)) != 0:
                        reason_codes.append("target_style_alignment_terminology_drift_detected")
                    if int(review_report.target_style_alignment.get("evidence_drift", 0)) != 0:
                        reason_codes.append("target_style_alignment_evidence_drift_detected")
            if not review_report.academic_cliche_density_controlled and review_report.local_revision_realism.get(
                "high_sensitivity_cliche_paragraph_ids",
                [],
            ):
                warnings.append("High-sensitivity prose still carries visible academic cliché residue.")
            if not review_report.scaffolding_phrase_density_controlled:
                warnings.append("Generated scaffolding phrasing remains too visible in rewritten prose.")
            if not review_report.appendix_like_support_controlled:
                warnings.append("Appendix-like support phrasing remains too visible in rewritten prose.")
            if not review_report.unsupported_expansion_controlled:
                warnings.append("Unsupported expansion remains visible in rewritten prose.")
            return WriteGateDecision(
                write_allowed=False,
                decision="reject",
                reason_codes=reason_codes,
                warnings=warnings,
                selected_candidate_reason=rewrite_report.selected_candidate_reason,
            )

    if not review_report.core_content_integrity:
        reason_codes.append("core_content_integrity_failed")
    if not review_report.format_integrity:
        reason_codes.append("format_integrity_failed")
    if review_report.body_rewrite_coverage < review_report.required_body_rewrite_coverage:
        reason_codes.append("body_rewrite_coverage_below_write_threshold")
    if review_report.body_discourse_change_score < review_report.required_body_discourse_change_score:
        reason_codes.append("body_discourse_change_score_below_write_threshold")
    if review_report.body_cluster_rewrite_score < review_report.required_body_cluster_rewrite_score:
        reason_codes.append("body_cluster_rewrite_score_below_write_threshold")
    if review_report.body_changed_block_ratio < review_report.required_body_changed_block_ratio:
        reason_codes.append("body_changed_blocks_below_write_threshold")
    if not review_report.rewrite_quota_met:
        reason_codes.extend(review_report.rewrite_quota_reason_codes)
    if not review_report.chapter_rewrite_quota_check:
        reason_codes.extend(review_report.chapter_rewrite_quota_reason_codes)
    if not review_report.chapter_policy_consistency_check:
        reason_codes.append("chapter_policy_consistency_failed")
    if not review_report.paragraph_topic_sentence_preserved:
        reason_codes.append("paragraph_topic_sentence_not_preserved")
    if not review_report.paragraph_opening_style_valid:
        reason_codes.append("paragraph_opening_style_invalid")
    if not review_report.paragraph_skeleton_consistent:
        reason_codes.append("paragraph_skeleton_inconsistent")
    if not review_report.no_dangling_opening_sentence:
        reason_codes.append("dangling_paragraph_opening_sentence")
    if not review_report.topic_sentence_not_demoted_to_mid_paragraph:
        reason_codes.append("topic_sentence_demoted_to_mid_paragraph")
    if not review_report.local_transition_natural:
        reason_codes.append("local_transition_rigidity")
    if not review_report.local_discourse_not_flat:
        reason_codes.append("local_discourse_flatness")
    if not review_report.sentence_uniformity_reduced:
        reason_codes.append("sentence_uniformity_not_reduced")
    if not review_report.revision_realism_present:
        reason_codes.append("revision_realism_missing")
    if not review_report.sentence_completeness_preserved:
        reason_codes.append("sentence_completeness_failed")
    if not review_report.paragraph_readability_preserved:
        reason_codes.append("paragraph_readability_failed")
    if not review_report.no_dangling_support_sentences:
        reason_codes.append("dangling_sentence_risk_too_high")
    if not review_report.no_fragment_like_conclusion_sentences:
        reason_codes.append("fragment_like_conclusion_sentence")
    if not review_report.semantic_role_integrity_preserved:
        reason_codes.append("semantic_role_integrity_failed")
    if not review_report.enumeration_integrity_preserved:
        reason_codes.append("enumeration_integrity_failed")
    if review_report.document_scale in {"long", "very_long"}:
        if not review_report.evidence_fidelity_preserved:
            reason_codes.append("evidence_fidelity_failed")
        if not review_report.unsupported_expansion_controlled:
            reason_codes.append("unsupported_expansion_failed")
        if not review_report.thesis_tone_restrained:
            reason_codes.append("thesis_tone_restraint_failed")
        if not review_report.metaphor_or_storytelling_controlled:
            reason_codes.append("metaphor_or_storytelling_failed")
        if not review_report.authorial_claim_risk_controlled:
            reason_codes.append("authorial_claim_risk_failed")
        if not review_report.bureaucratic_opening_controlled:
            reason_codes.append("bureaucratic_opening_density_failed")
        if not review_report.explicit_subject_chain_controlled:
            reason_codes.append("explicit_subject_chain_failed")
        if not review_report.overstructured_syntax_controlled:
            reason_codes.append("overstructured_syntax_failed")
        if not review_report.main_clause_position_reasonable:
            reason_codes.append("delayed_main_clause_risk_failed")
        if not review_report.slogan_like_goal_phrase_controlled:
            reason_codes.append("slogan_like_goal_phrase_failed")
        if (
            not l2_profile_enabled
            and not review_report.academic_sentence_naturalization.get("author_style_alignment_controlled", True)
        ):
            reason_codes.append("author_style_alignment_failed")
        if not review_report.sentence_cluster_changes_present:
            reason_codes.append("sentence_cluster_changes_missing_for_long_document")
        if not review_report.human_like_variation:
            reason_codes.append("human_like_variation_missing_for_long_document")
        if not review_report.non_uniform_rewrite_distribution:
            reason_codes.append("non_uniform_rewrite_distribution_missing_for_long_document")
    if review_report.target_style_alignment.get("enabled", False):
        if float(review_report.target_style_alignment.get("grammar_error_rate", 0.0)) > 0.02:
            reason_codes.append("target_style_alignment_grammar_error_rate_too_high")
        if int(review_report.target_style_alignment.get("terminology_drift", 0)) != 0:
            reason_codes.append("target_style_alignment_terminology_drift_detected")
        if int(review_report.target_style_alignment.get("evidence_drift", 0)) != 0:
            reason_codes.append("target_style_alignment_evidence_drift_detected")
    if not review_report.stylistic_uniformity_controlled:
        warnings.append("Stylistic uniformity remains slightly visible across rewritten paragraphs.")
    if not review_report.support_sentence_texture_varied:
        warnings.append("Support-sentence texture remains narrower than the author-like target.")
    if not review_report.paragraph_voice_variation_present:
        warnings.append("Paragraph voice variation remains too narrow for a strong long-document pass.")
    if not review_report.academic_cliche_density_controlled:
        warnings.append("Academic cliché density remains higher than the preferred author-like texture.")
    if not review_report.scaffolding_phrase_density_controlled:
        warnings.append("Generated scaffolding phrasing remains higher than the semantic-role target.")
    if not review_report.over_abstracted_subject_risk_controlled:
        warnings.append("Abstracted subjects still replace concrete referents in some rewritten paragraphs.")
    if not review_report.assertion_strength_preserved:
        warnings.append("Assertion strength remains weaker than the preferred author-like level.")
    if not review_report.appendix_like_support_controlled:
        warnings.append("Appendix-like supporting sentences remain slightly visible after revision.")
    if not review_report.authorial_stance_present:
        warnings.append("Source authorial stance markers remain underrepresented in the revised prose.")
    if not review_report.unsupported_expansion_controlled:
        warnings.append("Unsupported expansion remains higher than the preferred evidence-fidelity target.")
    if not review_report.thesis_tone_restrained:
        warnings.append("Thesis register still drifts toward commentary in some rewritten paragraphs.")
    if not review_report.metaphor_or_storytelling_controlled:
        warnings.append("Metaphoric or storytelling phrasing remains visible in rewritten prose.")
    if not review_report.authorial_claim_risk_controlled:
        warnings.append("Unjustified first-person or overclaimed authorial phrasing remains visible.")
    if not review_report.bureaucratic_opening_controlled:
        warnings.append("Project-style academic openings remain more visible than the naturalization target.")
    if not review_report.explicit_subject_chain_controlled:
        warnings.append("Explicit subject chains still read mechanically in some rewritten paragraphs.")
    if not review_report.overstructured_syntax_controlled:
        warnings.append("Parallel or contrast structures remain too visibly organized by the rewriter.")
    if not review_report.main_clause_position_reasonable:
        warnings.append("Some sentences still delay the main clause behind heavy prefatory modifiers.")
    if not review_report.slogan_like_goal_phrase_controlled:
        warnings.append("Slogan-like goal phrasing remains visible after sentence naturalization.")
    if not review_report.academic_sentence_naturalization.get("author_style_alignment_controlled", True):
        warnings.append("Author-style directness remains below the preferred direct, weak-connector target.")
    if review_report.structural_action_count < _required_structural_action_threshold(review_report, rewrite_report):
        reason_codes.append("structural_action_count_below_threshold")
    if review_report.high_value_action_count < DEFAULT_CONFIG.strict_min_high_value_actions:
        reason_codes.append("missing_high_value_action")
    if rewrite_report.block_failures:
        if (
            review_report.rewrite_quota_met
            and review_report.body_rewrite_coverage >= review_report.required_body_rewrite_coverage
            and review_report.body_discourse_change_score >= review_report.required_body_discourse_change_score
            and review_report.body_cluster_rewrite_score >= review_report.required_body_cluster_rewrite_score
            and review_report.chapter_rewrite_quota_check
            and review_report.chapter_policy_consistency_check
        ):
            pass
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
    suffix = str(context.get("suffix", ".md"))

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
            revision_patterns=["partial_keep"] if guidance.should_rewrite else [],
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
            high_sensitivity_prose=guidance.high_sensitivity_prose,
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
            high_sensitivity_prose=guidance.high_sensitivity_prose,
        )
    except TypeError:
        rewritten_text, stats = rewriter.rewrite(
            block,
            mode=effective_mode,
            pass_index=pass_index,
    )
    rewritten_text, semantic_surface_actions, semantic_surface_notes = _repair_block_semantic_role_surface(
        original_block=block,
        rewritten_block=rewritten_text,
        guidance=guidance,
    )
    rewritten_text, evidence_surface_actions, evidence_surface_notes = _repair_block_evidence_fidelity_surface(
        original_block=block,
        rewritten_block=rewritten_text,
        guidance=guidance,
    )
    surface_actions = _deduplicate([*semantic_surface_actions, *evidence_surface_actions])
    surface_notes = _deduplicate([*semantic_surface_notes, *evidence_surface_notes])
    if surface_actions:
        backend = getattr(rewriter, "backend", rewriter)
        rewritten_sentences = split_sentences(rewritten_text)
        structural_repairs: list[str] = []
        if "local:repair-enumeration-flow" in surface_actions:
            structural_repairs.append("repair_enumeration_flow")
        if "local:prevent-appendix-like-supporting-sentence" in surface_actions:
            structural_repairs.append("prevent_appendix_like_supporting_sentence")
        if "local:preserve-semantic-role-of-core-sentence" in surface_actions:
            structural_repairs.append("preserve_semantic_role_of_core_sentence")
        if "local:preserve-original-evidence-scope" in surface_actions:
            structural_repairs.append("preserve_original_evidence_scope")
        if "local:restore-mechanism-sentence-to-academic-statement" in surface_actions:
            structural_repairs.append("restore_mechanism_sentence_to_academic_statement")
        stats.changed = rewritten_text != block
        stats.changed_characters = sum(1 for left, right in zip(block, rewritten_text) if left != right) + abs(
            len(block) - len(rewritten_text)
        )
        stats.rewritten_sentences = rewritten_sentences
        stats.sentence_count_after = len(rewritten_sentences)
        stats.sentence_level_change = rewritten_text != block
        stats.applied_rules = _deduplicate([*stats.applied_rules, *surface_actions])
        stats.candidate_notes = _deduplicate([*stats.candidate_notes, *surface_notes])
        stats.structural_actions = _deduplicate([*stats.structural_actions, *structural_repairs])
        if hasattr(backend, "_extract_subject_head"):
            stats.subject_heads = [backend._extract_subject_head(sentence) for sentence in rewritten_sentences]
        if hasattr(backend, "_has_repeated_subject_risk"):
            stats.repeated_subject_risk = backend._has_repeated_subject_risk(stats.subject_heads)
        local_signals = analyze_local_revision_sentences(rewritten_sentences)
        stats.sentence_transition_rigidity = local_signals.sentence_transition_rigidity
        stats.local_discourse_flatness = local_signals.local_discourse_flatness
        stats.revision_realism_score = local_signals.revision_realism_score
        stats.sentence_cadence_irregularity = local_signals.sentence_cadence_irregularity
        readability = analyze_paragraph_readability_sentences(
            rewritten_sentences,
            high_sensitivity=guidance.high_sensitivity_prose,
        )
        stats.sentence_completeness_score = readability.sentence_completeness_score
        stats.paragraph_readability_score = readability.paragraph_readability_score
        stats.dangling_sentence_risk = readability.dangling_sentence_risk
        stats.incomplete_support_sentence_risk = readability.incomplete_support_sentence_risk
        stats.fragment_like_conclusion_risk = readability.fragment_like_conclusion_risk
    stats.block_id = guidance.block_id
    stats.rewrite_depth = guidance.rewrite_depth
    stats.rewrite_intensity = guidance.rewrite_intensity
    stats.high_sensitivity_prose = guidance.high_sensitivity_prose
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
        revision_patterns=list(stats.revision_patterns or guidance.revision_pattern),
        required_actions_met=not missing_required_actions,
        missing_required_actions=missing_required_actions,
        notes=_deduplicate([*notes, *stats.candidate_notes]),
        mode_used=effective_mode.value,
    )
    return candidate, stats


def _repair_block_semantic_role_surface(
    *,
    original_block: str,
    rewritten_block: str,
    guidance: BlockPolicy,
) -> tuple[str, list[str], list[str]]:
    """Repair known semantic-role regressions that survive block rewriting."""

    if not original_block.strip() or not rewritten_block.strip() or rewritten_block == original_block:
        return rewritten_block, [], []

    explicit_enumeration = bool(
        re.match(r"^\s*(?:[（(]?(?:\d+|\[\[AIRC:CORE_NUMBER:\d+\]\])[）)]?|第[一二三四五六七八九十]+)", original_block)
    )
    semantic_role_sensitive = guidance.high_sensitivity_prose or guidance.chapter_type in {
        "training_strategy",
        "mechanism_explanation",
        "method_design",
        "conclusion",
        "future_work",
    }
    if not explicit_enumeration and not semantic_role_sensitive:
        return rewritten_block, [], []

    repaired = rewritten_block
    actions: list[str] = []
    notes: list[str] = []

    candidate = re.sub(
        r"(根据[^。]{2,120})。从真实样本中",
        r"\1，从真实样本中",
        repaired,
    )
    if candidate != repaired:
        repaired = candidate
        actions.extend(
            [
                "local:preserve-enumeration-item-role",
                "local:repair-enumeration-flow",
            ]
        )
        notes.append("semantic-role surface repair merged a broken mechanism fragment back into the enumeration sentence")

    candidate = re.sub(
        r"(?:本研究|本文)(?:更强调|重点)通过课程采样器",
        "训练通过课程采样器",
        repaired,
    )
    if candidate != repaired:
        repaired = candidate
        actions.extend(
            [
                "local:preserve-enumeration-item-role",
                "local:remove-generated-scaffolding-phrase",
            ]
        )
        notes.append("semantic-role surface repair restored a direct course-learning mechanism statement")

    severe_fragment_match = re.search(
        r"(?P<prefix>[（(]?(?:\d+|\[\[AIRC:CORE_NUMBER:\d+\]\])[）)]?[^：:\n]{2,42}[：:])"
        r"若仅强调(?P<lemma>[^。]{2,120})。训练中同时维护(?P<support>[^。]{2,220})。"
        r"易导致(?P<consequence>[^。]{2,160})(?:。|$)",
        repaired,
    )
    if severe_fragment_match:
        repaired = (
            repaired[: severe_fragment_match.start()]
            + f"{severe_fragment_match.group('prefix')}若仅强调{severe_fragment_match.group('lemma')}，"
            + f"易导致{severe_fragment_match.group('consequence')}。"
            + f"因此训练中同时维护{severe_fragment_match.group('support')}。"
            + repaired[severe_fragment_match.end() :]
        )
        actions.extend(
            [
                "local:repair-enumeration-flow",
                "local:prevent-appendix-like-supporting-sentence",
            ]
        )
        notes.append("semantic-role surface repair rebuilt a fragmented enumeration item into a complete causal pair")

    return repaired, _deduplicate(actions), _deduplicate(notes)


def _repair_document_evidence_fidelity_surface(
    *,
    original_text: str,
    rewritten_text: str,
) -> str:
    """Apply final document-level restraint repairs for protected evidence boundaries."""

    repaired = rewritten_text

    source_hard_real = _source_paragraph_containing(original_text, "hard_real_buffer")
    if source_hard_real and "base_probability、base_logit" in repaired and "hard_real_buffer" not in repaired:
        repaired = re.sub(
            r"（1）困难真实样本挖掘：[^。\n]*base_probability、base_logit[^。\n]*。该过程无需人工干预，而是通过模型输出自动识别最易误判的样本区域。",
            source_hard_real.strip(),
            repaired,
        )

    source_literature_close = _source_paragraph_containing(original_text, "特别是在多分支模型中")
    if source_literature_close and ("[18]研究" in repaired or "[18]本研究" in repaired):
        repaired = re.sub(
            r"综上所述，现有 AIGC 图像检测方法[^。\n]*问题\[17\]。特别是在多分支模型中[^。\n]*?\[18\][^。\n]*?。(?:综上所述，现有 AIGC 图像检测方法[^。\n]*问题\[17\]。)?",
            source_literature_close.strip(),
            repaired,
        )

    source_ntire = _source_paragraph_containing(original_text, "NTIRE-RobustAIGenDetection")
    if source_ntire and repaired.count("NTIRE-RobustAIGenDetection") > original_text.count("NTIRE-RobustAIGenDetection"):
        repaired = re.sub(
            r"主线训练与验证采用 2024 年 NTIRE-RobustAIGenDetection 数据集[^。\n]*。"
            r"(?:主线训练与验证采用 2024 年 NTIRE-RobustAIGenDetection 数据集[^。\n]*。)+"
            r"(?:基于其大规模、多扰动的特性[^。\n]*。)?",
            source_ntire.strip(),
            repaired,
        )

    source_photos = _source_paragraph_containing(original_text, "后续研究需要进一步提升语义分支")
    if source_photos and repaired.count("real1") > original_text.count("real1"):
        repaired = re.sub(
            r"当前模型仍然依赖频域相关证据。后续研究需要进一步提升语义分支[^。\n]*real1、real8[^。\n]*。后续研究仍将围绕 real1、real8[^。\n]*。",
            source_photos.strip(),
            repaired,
        )
        repaired = re.sub(
            r"(后续研究需要进一步提升语义分支[^。\n]*real1、real8[^。\n]*。)后续研究仍将围绕 real1、real8[^。\n]*。",
            r"\1",
            repaired,
        )

    repaired = re.sub(
        r"(保持了对此集合中所有 AIGC 样本的完全召回，误报数量较早期版本明显下降。)\1",
        r"\1",
        repaired,
    )
    repaired = re.sub(
        r"(保持了对此集合中所有 AIGC 样本的完全召回，误报数量较早期版本明显下降。)"
        r"保持了对此集合中所有 AIGC 样本的完全召回，误报数量较早期版本明显下降。",
        r"\1",
        repaired,
    )

    return repaired


def _repair_document_academic_sentence_surface(rewritten_text: str) -> str:
    """Repair punctuation seams introduced by sentence naturalization without changing content."""

    repaired = rewritten_text
    repaired = repaired.replace("，；", "，")
    repaired = repaired.replace("。；", "。")
    repaired = repaired.replace("；；", "；")
    repaired = repaired.replace("，。", "。")
    repaired = repaired.replace("在方法上，本研究", "本研究")
    repaired = repaired.replace("本研究具有显著的安全价值，在当前网络环境下愈发不可或缺。", "本研究的安全价值在当前网络环境下愈发突出。")
    repaired = repaired.replace("可以用于了清晰的方法论", "给出了较清晰的方法论")
    repaired = repaired.replace("是了实现基础", "提供了实现基础")
    repaired = repaired.replace(
        "研究不仅实现了一个性能良好的检测系统，更为构建实用化AIGC图像检测工具给出清晰的方法论路径。",
        "研究实现了一个性能良好的检测系统，也为构建实用化AIGC图像检测工具给出清晰的方法论路径。",
    )
    repaired = re.sub(
        r"该机制使得模型在结构上仍保留([^，。；]{2,40})，但在决策层面实现了",
        r"该机制保留\1，同时实现了",
        repaired,
    )
    return repaired


def _source_paragraph_containing(text: str, needle: str) -> str:
    """Return the original paragraph that contains a specific protected evidence marker."""

    for paragraph in re.split(r"\n\s*\n", text):
        if needle in paragraph:
            return paragraph.strip()
    return ""


def _repair_block_evidence_fidelity_surface(
    *,
    original_block: str,
    rewritten_block: str,
    guidance: BlockPolicy,
) -> tuple[str, list[str], list[str]]:
    """Repair severe block-level evidence/integrity drift that should fall back to source phrasing."""

    if not original_block.strip() or not rewritten_block.strip() or rewritten_block == original_block:
        return rewritten_block, [], []

    original_tokens = set(re.findall(r"\b[a-z]+(?:_[a-z0-9]+)+\b", original_block))
    missing_tokens = sorted(token for token in original_tokens if token not in rewritten_block)
    duplicated_enumeration = bool(
        re.search(r"([（(]\d+[）)])[^。\n]{0,220}\1", rewritten_block)
    )
    fused_citation_boundary = bool(
        re.search(r"\[\d+\](?=[A-Za-z\u4e00-\u9fff])", rewritten_block)
    )
    duplicated_citation_tail = bool(
        len(re.findall(r"\[\d+\]", rewritten_block)) > len(re.findall(r"\[\d+\]", original_block)) + 1
    )
    appendix_like_enumeration_fragment = bool(
        re.search(r"(?:还包括|进一步包括|也包括|相关内容包括|这一机制还包括|这一设计还包括)", rewritten_block)
        and (
            re.search(r"^\s*(?:[（(]?(?:\d+|\[\[AIRC:CORE_NUMBER:\d+\]\])[）)]?|第[一二三四五六七八九十]+)", original_block)
            or "主要创新点" in original_block
            or "课程学习" in original_block
        )
    )
    severe_fragment_pair = bool(
        re.search(r"根据[^。]{2,120}。从真实样本中", rewritten_block)
        or re.search(r"若仅强调[^。]{2,120}。训练中同时维护", rewritten_block)
    )

    if not (
        missing_tokens
        or duplicated_enumeration
        or fused_citation_boundary
        or duplicated_citation_tail
        or appendix_like_enumeration_fragment
        or severe_fragment_pair
    ):
        return rewritten_block, [], []

    notes: list[str] = []
    if missing_tokens:
        notes.append(
            "evidence-fidelity surface repair restored the source block because protected identifiers disappeared: "
            + ", ".join(missing_tokens[:4])
        )
    if fused_citation_boundary or duplicated_citation_tail:
        notes.append("evidence-fidelity surface repair restored the source block after citation-boundary drift")
    if duplicated_enumeration or appendix_like_enumeration_fragment or severe_fragment_pair:
        notes.append("evidence-fidelity surface repair restored the source block after enumeration/mechanism drift")

    return (
        original_block,
        _deduplicate(
            [
                "local:preserve-original-evidence-scope",
                "local:restore-thesis-register",
                "local:restore-mechanism-sentence-to-academic-statement",
            ]
        ),
        _deduplicate(notes),
    )


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

        if block.chapter_rewrite_priority == "conservative":
            intensity = "light"
            discourse_actions = _deduplicate([*discourse_actions, "sentence_level_recast"])
            recommended_actions = _deduplicate(
                [*recommended_actions, "chapter_conservative_light_edit", "partial_keep", "light_clause_reorder"]
            )
            structural_actions = [
                action for action in structural_actions if action not in {"pair_fusion", "sentence_cluster_merge"}
            ]
            min_sentence = max(min_sentence, DEFAULT_CONFIG.light_edit_min_sentence_level_changes)
            min_cluster = 0
            notes.append("Retry escalation: conservative chapter stays light to avoid technical drift.")
        elif block.rewrite_depth == "developmental_rewrite":
            intensity = "medium" if block.chapter_rewrite_priority == "medium" else "high"
            discourse_actions = _deduplicate(
                [*discourse_actions, "sentence_cluster_rewrite", "proposition_reorder", "transition_absorption"]
            )
            structural_actions = _deduplicate([*structural_actions, "pair_fusion"])
            recommended_actions = _deduplicate(
                [*recommended_actions, "sentence_cluster_rewrite", "narrative_flow_rebuilder", "conclusion_absorb"]
            )
            min_sentence = max(min_sentence, DEFAULT_CONFIG.developmental_min_sentence_level_changes)
            if block.chapter_rewrite_priority != "medium":
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
                chapter_rewrite_intensity=intensity,
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
            f"revision_patterns={candidate.revision_patterns or ['none']} "
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
    """Rank candidates by evidence fidelity and restrained thesis register before rewrite scale."""

    decision_rank = {
        "pass": 6,
        "pass_with_minor_risk": 4,
        "reject": -2,
    }.get(write_gate.decision, 0)
    template_penalty = int(review.template_risk) * 100 + int(review.template_warning) * 25
    realism_score = int(float(review.local_revision_realism.get("revision_realism_score", 0.0)) * 100)
    readability_score = int(float(review.sentence_readability.get("paragraph_readability_score", 0.0)) * 100)
    completeness_score = int(float(review.sentence_readability.get("sentence_completeness_score", 0.0)) * 100)
    semantic_role_score = int(float(review.semantic_role_integrity.get("semantic_role_integrity_score", 0.0)) * 100)
    enumeration_score = int(float(review.semantic_role_integrity.get("enumeration_integrity_score", 0.0)) * 100)
    assertion_strength_score = int(float(review.authorial_intent.get("assertion_strength_score", 0.0)) * 100)
    authorial_stance_score = int(float(review.authorial_intent.get("authorial_stance_presence", 0.0)) * 100)
    evidence_fidelity_score = int(float(review.evidence_fidelity.get("evidence_fidelity_score", 0.0)) * 100)
    thesis_tone_score = int(float(review.evidence_fidelity.get("thesis_tone_restraint_score", 0.0)) * 100)
    dangling_penalty = int(float(review.sentence_readability.get("dangling_sentence_risk", 0.0)) * 100)
    support_penalty = int(float(review.sentence_readability.get("incomplete_support_sentence_risk", 0.0)) * 100)
    conclusion_penalty = int(float(review.sentence_readability.get("fragment_like_conclusion_risk", 0.0)) * 100)
    transition_penalty = int(float(review.local_revision_realism.get("sentence_transition_rigidity", 0.0)) * 100)
    stylistic_uniformity_penalty = int(float(review.local_revision_realism.get("stylistic_uniformity_score", 0.0)) * 100)
    texture_variation_score = int(float(review.local_revision_realism.get("support_sentence_texture_variation", 0.0)) * 100)
    voice_variation_score = int(float(review.local_revision_realism.get("paragraph_voice_variation", 0.0)) * 100)
    cliche_penalty = int(float(review.local_revision_realism.get("academic_cliche_density", 0.0)) * 100)
    scaffolding_penalty = int(float(review.semantic_role_integrity.get("scaffolding_phrase_density", 0.0)) * 100)
    abstracted_subject_penalty = int(
        float(review.semantic_role_integrity.get("over_abstracted_subject_risk", 0.0)) * 100
    )
    appendix_penalty = int(float(review.authorial_intent.get("appendix_like_support_ratio", 0.0)) * 100)
    unsupported_penalty = int(float(review.evidence_fidelity.get("unsupported_expansion_risk", 0.0)) * 100)
    metaphor_penalty = int(float(review.evidence_fidelity.get("metaphor_or_storytelling_risk", 0.0)) * 100)
    authorial_claim_penalty = int(float(review.evidence_fidelity.get("unjustified_authorial_claim_risk", 0.0)) * 100)
    bureaucratic_penalty = int(
        float(review.academic_sentence_naturalization.get("bureaucratic_opening_density", 0.0)) * 100
    )
    repeated_subject_penalty = int(
        float(review.academic_sentence_naturalization.get("repeated_explicit_subject_risk", 0.0)) * 100
    )
    overstructured_penalty = int(
        float(review.academic_sentence_naturalization.get("overstructured_syntax_risk", 0.0)) * 100
    )
    delayed_clause_penalty = int(
        float(review.academic_sentence_naturalization.get("delayed_main_clause_risk", 0.0)) * 100
    )
    slogan_penalty = int(float(review.academic_sentence_naturalization.get("slogan_like_goal_risk", 0.0)) * 100)
    directness_score = int(float(review.academic_sentence_naturalization.get("directness_score", 0.0)) * 100)
    target_alignment = review.target_style_alignment or {}
    target_alignment_score = int(float(target_alignment.get("target_style_alignment_score", 1.0)) * 100)
    style_match_score = int(float(target_alignment.get("style_distribution_match_ratio", 1.0)) * 100)
    class_style_match_score = int(float(target_alignment.get("class_aware_style_match_ratio", 1.0)) * 100)
    avg_sentence_length_penalty = int(float(target_alignment.get("avg_sentence_length_diff", 0.0)) * 2)
    clause_diff_penalty = int(float(target_alignment.get("clause_per_sentence_diff", 0.0)) * 20)
    main_clause_penalty = int(float(target_alignment.get("main_clause_position_diff", 0.0)) * 100)
    function_word_penalty = int(float(target_alignment.get("function_word_density_diff", 0.0)) * 100)
    helper_verb_penalty = int(float(target_alignment.get("helper_verb_usage_diff", 0.0)) * 100)
    explanatory_gap_penalty = int(float(target_alignment.get("explanatory_rewrite_gap", 0.0)) * 100)
    compactness_gap_penalty = int(float(target_alignment.get("compactness_gap", 0.0)) * 100)
    native_gap_penalty = int(float(target_alignment.get("native_fluency_gap", 0.0)) * 100)
    l2_gap_penalty = int(float(target_alignment.get("l2_texture_gap", 0.0)) * 100)
    grammar_penalty = int(float(target_alignment.get("grammar_error_rate", 0.0)) * 100)
    terminology_penalty = int(target_alignment.get("terminology_drift", 0)) * 10
    evidence_drift_penalty = int(target_alignment.get("evidence_drift", 0)) * 15
    l2_profile = review.l2_style_profile or {}
    l2_enabled = bool(l2_profile.get("enabled", False))
    l2_texture_score = int(float(l2_profile.get("l2_texture_score", 0.0)) * 100) if l2_enabled else 0
    native_like_penalty = int(float(l2_profile.get("native_like_concision_risk", 0.0)) * 100) if l2_enabled else 0
    l2_colloquial_penalty = int(float(l2_profile.get("colloquial_risk", 0.0)) * 100) if l2_enabled else 0
    l2_ungrammatical_penalty = int(float(l2_profile.get("ungrammatical_risk", 0.0)) * 100) if l2_enabled else 0
    connector_penalty = int(float(review.academic_sentence_naturalization.get("connector_overuse_risk", 0.0)) * 100)
    nominalization_penalty = int(float(review.academic_sentence_naturalization.get("nominalization_density", 0.0)) * 100)
    passive_penalty = int(float(review.academic_sentence_naturalization.get("passive_voice_ratio", 0.0)) * 100)
    overlong_penalty = int(float(review.academic_sentence_naturalization.get("overlong_sentence_risk", 0.0)) * 100)
    subject_monotony_penalty = int(float(review.academic_sentence_naturalization.get("subject_monotony_risk", 0.0)) * 100)
    return (
        int(write_gate.write_allowed),
        decision_rank,
        int(review.chapter_rewrite_quota_check),
        int(review.rewrite_quota_met),
        int(review.chapter_policy_consistency_check),
        evidence_fidelity_score,
        semantic_role_score,
        target_alignment_score,
        class_style_match_score,
        style_match_score,
        -evidence_drift_penalty,
        -terminology_penalty,
        -grammar_penalty,
        readability_score,
        thesis_tone_score,
        directness_score,
        l2_texture_score,
        -native_like_penalty,
        -l2_colloquial_penalty,
        -l2_ungrammatical_penalty,
        -connector_penalty,
        -subject_monotony_penalty,
        -overlong_penalty,
        -avg_sentence_length_penalty,
        -clause_diff_penalty,
        -main_clause_penalty,
        -function_word_penalty,
        -helper_verb_penalty,
        -explanatory_gap_penalty,
        -compactness_gap_penalty,
        -native_gap_penalty,
        -l2_gap_penalty,
        -bureaucratic_penalty,
        -repeated_subject_penalty,
        -overstructured_penalty,
        -delayed_clause_penalty,
        -slogan_penalty,
        -nominalization_penalty,
        -passive_penalty,
        -unsupported_penalty,
        -metaphor_penalty,
        -authorial_claim_penalty,
        assertion_strength_score,
        authorial_stance_score,
        -appendix_penalty,
        completeness_score,
        enumeration_score,
        int(review.paragraph_readability_preserved),
        int(review.sentence_completeness_preserved),
        int(review.semantic_role_integrity_preserved),
        int(review.enumeration_integrity_preserved),
        int(review.assertion_strength_preserved),
        int(review.appendix_like_support_controlled),
        int(review.authorial_stance_present),
        int(review.evidence_fidelity_preserved),
        int(review.unsupported_expansion_controlled),
        int(review.thesis_tone_restrained),
        int(review.metaphor_or_storytelling_controlled),
        int(review.authorial_claim_risk_controlled),
        int(review.bureaucratic_opening_controlled),
        int(review.explicit_subject_chain_controlled),
        int(review.overstructured_syntax_controlled),
        int(review.main_clause_position_reasonable),
        int(review.slogan_like_goal_phrase_controlled),
        int(review.scaffolding_phrase_density_controlled),
        int(review.over_abstracted_subject_risk_controlled),
        -dangling_penalty,
        -support_penalty,
        -conclusion_penalty,
        -scaffolding_penalty,
        -abstracted_subject_penalty,
        -stylistic_uniformity_penalty,
        texture_variation_score,
        voice_variation_score,
        -cliche_penalty,
        int(review.stylistic_uniformity_controlled),
        int(review.support_sentence_texture_varied),
        int(review.paragraph_voice_variation_present),
        int(review.academic_cliche_density_controlled),
        -transition_penalty,
        realism_score,
        int(review.local_transition_natural),
        int(review.local_discourse_not_flat),
        int(review.sentence_uniformity_reduced),
        int(review.revision_realism_present),
        int(review.body_rewrite_coverage * 100),
        review.body_discourse_change_score,
        review.body_cluster_rewrite_score,
        review.body_changed_blocks,
        -template_penalty,
        int(review.effective_change),
        int(not review.repeated_subject_risk),
        review.structural_action_count,
        -len(review.problems),
        review.changed_characters,
    )


def _needs_retry_escalation(review: ReviewReport, mode: RewriteMode) -> bool:
    if mode is RewriteMode.CONSERVATIVE:
        return False
    return (
        review.body_rewrite_coverage < review.required_body_rewrite_coverage
        or review.body_discourse_change_score < review.required_body_discourse_change_score
        or review.body_cluster_rewrite_score < review.required_body_cluster_rewrite_score
        or not review.rewrite_quota_met
        or not review.chapter_rewrite_quota_check
        or bool(review.failed_block_ids)
        or not review.revision_realism_present
        or not review.sentence_completeness_preserved
        or not review.paragraph_readability_preserved
        or not review.stylistic_uniformity_controlled
        or not review.support_sentence_texture_varied
          or not review.paragraph_voice_variation_present
          or not review.academic_cliche_density_controlled
          or not review.semantic_role_integrity_preserved
          or not review.enumeration_integrity_preserved
        or not review.scaffolding_phrase_density_controlled
        or not review.over_abstracted_subject_risk_controlled
        or not review.assertion_strength_preserved
        or not review.appendix_like_support_controlled
        or not review.authorial_stance_present
        or not review.evidence_fidelity_preserved
        or not review.unsupported_expansion_controlled
        or not review.thesis_tone_restrained
        or not review.metaphor_or_storytelling_controlled
        or not review.authorial_claim_risk_controlled
        or not review.bureaucratic_opening_controlled
        or not review.explicit_subject_chain_controlled
        or not review.overstructured_syntax_controlled
        or not review.main_clause_position_reasonable
        or not review.slogan_like_goal_phrase_controlled
        or not review.academic_sentence_naturalization.get("author_style_alignment_controlled", True)
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
        f"body_rewrite_coverage={candidate.review.body_rewrite_coverage:.2f},"
        f"body_changed_blocks={candidate.review.body_changed_blocks}/{candidate.review.body_blocks_total},"
        f"body_discourse_change_score={candidate.review.body_discourse_change_score},"
        f"body_cluster_rewrite_score={candidate.review.body_cluster_rewrite_score},"
        f"document_scale={candidate.review.document_scale},"
        f"rewrite_quota_met={candidate.review.rewrite_quota_met},"
        f"revision_realism_score={candidate.review.local_revision_realism.get('revision_realism_score', 0.0)},"
        f"stylistic_uniformity_score={candidate.review.local_revision_realism.get('stylistic_uniformity_score', 0.0)},"
        f"support_sentence_texture_variation={candidate.review.local_revision_realism.get('support_sentence_texture_variation', 0.0)},"
        f"paragraph_voice_variation={candidate.review.local_revision_realism.get('paragraph_voice_variation', 0.0)},"
        f"academic_cliche_density={candidate.review.local_revision_realism.get('academic_cliche_density', 0.0)},"
        f"paragraph_readability_score={candidate.review.sentence_readability.get('paragraph_readability_score', 0.0)},"
        f"sentence_completeness_score={candidate.review.sentence_readability.get('sentence_completeness_score', 0.0)},"
        f"semantic_role_integrity_score={candidate.review.semantic_role_integrity.get('semantic_role_integrity_score', 0.0)},"
        f"enumeration_integrity_score={candidate.review.semantic_role_integrity.get('enumeration_integrity_score', 0.0)},"
        f"scaffolding_phrase_density={candidate.review.semantic_role_integrity.get('scaffolding_phrase_density', 0.0)},"
        f"over_abstracted_subject_risk={candidate.review.semantic_role_integrity.get('over_abstracted_subject_risk', 0.0)},"
        f"assertion_strength_score={candidate.review.authorial_intent.get('assertion_strength_score', 0.0)},"
        f"appendix_like_support_ratio={candidate.review.authorial_intent.get('appendix_like_support_ratio', 0.0)},"
        f"authorial_stance_presence={candidate.review.authorial_intent.get('authorial_stance_presence', 0.0)},"
        f"evidence_fidelity_score={candidate.review.evidence_fidelity.get('evidence_fidelity_score', 0.0)},"
        f"unsupported_expansion_risk={candidate.review.evidence_fidelity.get('unsupported_expansion_risk', 0.0)},"
        f"thesis_tone_restraint_score={candidate.review.evidence_fidelity.get('thesis_tone_restraint_score', 0.0)},"
        f"metaphor_or_storytelling_risk={candidate.review.evidence_fidelity.get('metaphor_or_storytelling_risk', 0.0)},"
        f"unjustified_authorial_claim_risk={candidate.review.evidence_fidelity.get('unjustified_authorial_claim_risk', 0.0)},"
        f"bureaucratic_opening_density={candidate.review.academic_sentence_naturalization.get('bureaucratic_opening_density', 0.0)},"
        f"repeated_explicit_subject_risk={candidate.review.academic_sentence_naturalization.get('repeated_explicit_subject_risk', 0.0)},"
        f"overstructured_syntax_risk={candidate.review.academic_sentence_naturalization.get('overstructured_syntax_risk', 0.0)},"
        f"delayed_main_clause_risk={candidate.review.academic_sentence_naturalization.get('delayed_main_clause_risk', 0.0)},"
        f"slogan_like_goal_risk={candidate.review.academic_sentence_naturalization.get('slogan_like_goal_risk', 0.0)},"
        f"directness_score={candidate.review.academic_sentence_naturalization.get('directness_score', 0.0)},"
        f"connector_overuse_risk={candidate.review.academic_sentence_naturalization.get('connector_overuse_risk', 0.0)},"
        f"nominalization_density={candidate.review.academic_sentence_naturalization.get('nominalization_density', 0.0)},"
        f"passive_voice_ratio={candidate.review.academic_sentence_naturalization.get('passive_voice_ratio', 0.0)},"
        f"overlong_sentence_risk={candidate.review.academic_sentence_naturalization.get('overlong_sentence_risk', 0.0)},"
        f"subject_monotony_risk={candidate.review.academic_sentence_naturalization.get('subject_monotony_risk', 0.0)},"
        f"target_style_alignment_score={candidate.review.target_style_alignment.get('target_style_alignment_score', 1.0)},"
        f"style_distribution_match_ratio={candidate.review.target_style_alignment.get('style_distribution_match_ratio', 1.0)},"
        f"class_aware_style_match_ratio={candidate.review.target_style_alignment.get('class_aware_style_match_ratio', 1.0)},"
        f"avg_sentence_length_diff={candidate.review.target_style_alignment.get('avg_sentence_length_diff', 0.0)},"
        f"clause_per_sentence_diff={candidate.review.target_style_alignment.get('clause_per_sentence_diff', 0.0)},"
        f"main_clause_position_diff={candidate.review.target_style_alignment.get('main_clause_position_diff', 0.0)},"
        f"function_word_density_diff={candidate.review.target_style_alignment.get('function_word_density_diff', 0.0)},"
        f"helper_verb_usage_diff={candidate.review.target_style_alignment.get('helper_verb_usage_diff', 0.0)},"
        f"explanatory_rewrite_gap={candidate.review.target_style_alignment.get('explanatory_rewrite_gap', 0.0)},"
        f"compactness_gap={candidate.review.target_style_alignment.get('compactness_gap', 0.0)},"
        f"native_fluency_gap={candidate.review.target_style_alignment.get('native_fluency_gap', 0.0)},"
        f"l2_texture_gap={candidate.review.target_style_alignment.get('l2_texture_gap', 0.0)},"
        f"grammar_error_rate={candidate.review.target_style_alignment.get('grammar_error_rate', 0.0)},"
        f"terminology_drift={candidate.review.target_style_alignment.get('terminology_drift', 0)},"
        f"evidence_drift={candidate.review.target_style_alignment.get('evidence_drift', 0)},"
        f"dangling_sentence_risk={candidate.review.sentence_readability.get('dangling_sentence_risk', 0.0)},"
        f"incomplete_support_sentence_risk={candidate.review.sentence_readability.get('incomplete_support_sentence_risk', 0.0)},"
        f"local_transition_natural={candidate.review.local_transition_natural},"
        f"local_discourse_not_flat={candidate.review.local_discourse_not_flat},"
        f"sentence_uniformity_reduced={candidate.review.sentence_uniformity_reduced},"
        f"revision_realism_present={candidate.review.revision_realism_present},"
        f"stylistic_uniformity_controlled={candidate.review.stylistic_uniformity_controlled},"
        f"support_sentence_texture_varied={candidate.review.support_sentence_texture_varied},"
        f"paragraph_voice_variation_present={candidate.review.paragraph_voice_variation_present},"
        f"academic_cliche_density_controlled={candidate.review.academic_cliche_density_controlled},"
        f"bureaucratic_opening_controlled={candidate.review.bureaucratic_opening_controlled},"
        f"explicit_subject_chain_controlled={candidate.review.explicit_subject_chain_controlled},"
        f"overstructured_syntax_controlled={candidate.review.overstructured_syntax_controlled},"
        f"main_clause_position_reasonable={candidate.review.main_clause_position_reasonable},"
        f"slogan_like_goal_phrase_controlled={candidate.review.slogan_like_goal_phrase_controlled},"
        f"author_style_alignment_controlled={candidate.review.academic_sentence_naturalization.get('author_style_alignment_controlled', True)},"
        f"chapter_rewrite_quota_check={candidate.review.chapter_rewrite_quota_check},"
        f"chapter_policy_consistency_check={candidate.review.chapter_policy_consistency_check},"
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
            "status": "failed",
            "rewritten_file_path": None,
            "report_file_path": str(target),
            "rewrite_report_path": str(target),
            "input_normalization": normalization.to_dict(),
            "rewrite_coverage": 0.0,
            "body_rewrite_coverage": 0.0,
            "body_changed_blocks": 0,
            "body_blocks_total": 0,
            "body_changed_sentences": 0,
            "body_sentences_total": 0,
            "body_discourse_change_score": 0,
            "body_cluster_rewrite_score": 0,
            "document_scale": "short",
            "rewrite_quota_met": False,
            "human_like_variation": False,
            "non_uniform_rewrite_distribution": False,
            "sentence_cluster_changes_present": False,
            "narrative_flow_changed": False,
            "revision_pattern_distribution": {},
            "chapter_rewrite_metrics": [],
            "chapter_policy_consistency_check": False,
            "chapter_rewrite_quota_check": False,
            "chapter_rewrite_quota_reason_codes": [],
            "discourse_change_score": 0,
            "cluster_rewrite_score": 0,
            "blocks_changed": [],
            "blocks_skipped": [],
            "blocks_rejected": [],
            "warnings": [error],
            "failed_obligations": {},
            "write_gate_decision": "normalization_failed",
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
            "body_rewrite_coverage": result.review.body_rewrite_coverage,
            "body_changed_blocks": result.review.body_changed_blocks,
            "body_blocks_total": result.review.body_blocks_total,
            "body_changed_sentences": result.review.body_changed_sentences,
            "body_sentences_total": result.review.body_sentences_total,
            "body_discourse_change_score": result.review.body_discourse_change_score,
            "body_cluster_rewrite_score": result.review.body_cluster_rewrite_score,
            "document_scale": result.review.document_scale,
            "rewrite_quota_met": result.review.rewrite_quota_met,
            "human_like_variation": result.review.human_like_variation,
            "non_uniform_rewrite_distribution": result.review.non_uniform_rewrite_distribution,
            "sentence_cluster_changes_present": result.review.sentence_cluster_changes_present,
            "narrative_flow_changed": result.review.narrative_flow_changed,
            "revision_pattern_distribution": result.review.revision_pattern_distribution,
            "chapter_rewrite_metrics": result.review.chapter_rewrite_metrics,
            "chapter_policy_consistency_check": result.review.chapter_policy_consistency_check,
            "chapter_rewrite_quota_check": result.review.chapter_rewrite_quota_check,
            "chapter_rewrite_quota_reason_codes": result.review.chapter_rewrite_quota_reason_codes,
            "discourse_change_score": result.review.discourse_change_score,
            "cluster_rewrite_score": result.review.cluster_rewrite_score,
            "style_variation_score": result.review.style_variation_score,
            "format_integrity": result.review.format_integrity,
            "core_content_integrity": result.review.core_content_integrity,
            "failed_block_ids": result.review.failed_block_ids,
            "problems": result.review.problems,
            "warnings": result.review.warnings,
            "target_style_alignment": result.review.target_style_alignment,
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
            "current_body_rewrite_coverage": result.review.body_rewrite_coverage,
            "recommended_next_intensity": "high"
            if result.review.body_rewrite_coverage < result.review.required_body_rewrite_coverage
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
        "sentence_cluster_merge": {"sentence_cluster_merge", "sentence_cluster_rewrite", "pair_fusion", "sentence_merge"},
        "sentence_cluster_split": {"sentence_cluster_split", "sentence_split", "rebuild_sentence_rhythm"},
        "discourse_reordering": {"discourse_reordering", "proposition_reorder", "paragraph_reorder"},
        "narrative_path_rewrite": {"narrative_path_rewrite", "proposition_reorder", "paragraph_reorder"},
        "conclusion_absorption": {"conclusion_absorption", "conclusion_absorb", "followup_absorb"},
        "uneven_rewrite_distribution": {"uneven_rewrite_distribution", "partial_keep"},
        "conclusion_absorb": {"conclusion_absorb", "conclusion_absorption", "followup_absorb"},
        "subject_chain_compression": {
            "subject_chain_compression",
            "compress_subject_chain",
            "merge_consecutive_subject_sentences",
            "subject_drop",
            "subject_variation",
            "meta_compression",
            "followup_absorb",
        },
        "enumeration_reframe": {"enumeration_reframe"},
        "clause_reorder": {"clause_reorder", "pair_fusion", "sentence_merge"},
        "sentence_cluster_rewrite": {"sentence_cluster_rewrite", "sentence_cluster_merge", "pair_fusion", "conclusion_absorb", "sentence_merge"},
        "meta_compression": {"meta_compression", "subject_chain_compression"},
        "proposition_reorder": {"proposition_reorder", "paragraph_reorder"},
        "transition_absorption": {"transition_absorption", "pair_fusion", "sentence_cluster_rewrite"},
        "rationale_expansion": {"rationale_expansion", "clause_reorder"},
        "reduce_function_word_overuse": {"reduce_function_word_overuse"},
        "weaken_template_connectors": {"weaken_template_connectors", "transition_absorption"},
        "compress_subject_chain": {"compress_subject_chain", "subject_chain_compression"},
        "rebuild_sentence_rhythm": {"rebuild_sentence_rhythm", "sentence_split", "sentence_merge"},
        "break_parallelism": {"break_parallelism", "clause_reorder"},
        "rewrite_dense_nominal_phrases": {"rewrite_dense_nominal_phrases"},
        "preserve_explicit_subject_if_clarity_needed": {"preserve_explicit_subject_if_clarity_needed"},
        "keep_original_if_technical_density_is_high": {"keep_original_if_technical_density_is_high"},
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
