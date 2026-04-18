from __future__ import annotations

import argparse
from pathlib import Path

from .config import SKILL_PRESETS, RewriteMode
from .io_utils import FileOperationError
from .pipeline import guide_file, review_file, rewrite_file, run_file, suggest_file, write_file
from .skill_protocol import SkillInputSchema, build_execution_plan
from .validator import ValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airc",
        description="Agent-first academic revision helper: guide first, then review, then write only after integrity checks pass.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Public production interface. Run guide -> controlled rewrite -> review -> write and emit a JSON rewrite report.",
    )
    run_parser.add_argument("path", help="Local path to a .md, .txt, .docx, or experimental .doc file.")
    run_parser.add_argument(
        "--preset",
        choices=sorted(SKILL_PRESETS),
        default="academic_natural",
        help="User-level execution preset.",
    )
    run_parser.add_argument("--output", help="Optional rewritten output file path.")
    run_parser.add_argument("--report", help="Optional JSON rewrite report path.")
    run_parser.add_argument("--dry-run", action="store_true", help="Run all checks but do not write output or report files.")
    run_parser.add_argument(
        "--keep-normalized",
        action="store_true",
        help="Keep the intermediate normalized markdown file for doc/docx/txt inputs.",
    )
    run_parser.add_argument(
        "--debug-rewrite",
        action="store_true",
        help="Include rewrite debug traces in the console output.",
    )

    guide_parser = subparsers.add_parser(
        "guide",
        help="Primary entrypoint. Scan a markdown or text file and return block policy plus agent-facing rewrite guidance.",
    )
    guide_parser.add_argument("path", help="Local path to a .md or .txt file.")
    guide_parser.add_argument(
        "--as-agent-context",
        action="store_true",
        help="Render the guidance as compact agent-facing context with policies, forbidden actions, and the final self-checklist.",
    )

    review_parser = subparsers.add_parser(
        "review",
        help="Review an agent-supplied candidate against AIRC guidance, integrity checks, and the write gate.",
    )
    review_parser.add_argument("original", help="Original local .md or .txt file.")
    review_parser.add_argument("candidate", help="Candidate rewritten .md or .txt file.")
    review_parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RewriteMode],
        default=RewriteMode.BALANCED.value,
        help="Review mode used to judge rewrite depth expectations.",
    )

    write_parser = subparsers.add_parser(
        "write",
        help="Write an already reviewed candidate to a new file only if it passes the write gate.",
    )
    write_parser.add_argument("original", help="Original local .md or .txt file.")
    write_parser.add_argument("candidate", help="Candidate rewritten .md or .txt file.")
    write_parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RewriteMode],
        default=RewriteMode.BALANCED.value,
        help="Review mode used before writing.",
    )
    write_parser.add_argument("--output", help="Optional output file path.")
    write_parser.add_argument("--dry-run", action="store_true", help="Review and diff only; do not write a file.")

    rewrite_parser = subparsers.add_parser(
        "rewrite",
        help="Convenience preview mode. Run guide -> controlled rewrite -> review in one command without writing files.",
    )
    rewrite_parser.add_argument("path", help="Local path to a .md or .txt file.")
    rewrite_parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RewriteMode],
        default=RewriteMode.BALANCED.value,
        help="Revision intensity.",
    )
    rewrite_parser.add_argument("--dry-run", action="store_true", help="Preview the diff without writing a file.")
    rewrite_parser.add_argument("--output", help="Optional output file path.")
    rewrite_parser.add_argument(
        "--strict-mode",
        dest="strict_mode",
        action="store_true",
        default=True,
        help="Enable strict structural enforcement. This is on by default.",
    )
    rewrite_parser.add_argument(
        "--debug-rewrite",
        action="store_true",
        help="Print block policies, subject heads, structural actions, repeated-subject risk, reviewer decisions, and write-gate traces.",
    )

    suggest_parser = subparsers.add_parser("suggest", help="Generate author suggestions for vague paragraphs.")
    suggest_parser.add_argument("path", help="Local path to a .md or .txt file.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            result = run_file(
                path=Path(args.path),
                preset=args.preset,
                output_path=args.output,
                report_path=args.report,
                dry_run=args.dry_run,
                debug_rewrite=args.debug_rewrite,
                keep_intermediate=args.keep_normalized,
            )
            _print_run_summary(result)
            return 0 if result.output_written or args.dry_run else 1

        if args.command == "guide":
            result = guide_file(Path(args.path))
            if args.as_agent_context:
                _print_agent_context(result)
            else:
                _print_guide_summary(result)
            return 0

        if args.command == "review":
            result = review_file(
                original_path=Path(args.original),
                candidate_path=Path(args.candidate),
                mode=RewriteMode.from_value(args.mode),
            )
            _print_review_summary(result)
            return 0 if result.review.decision in {"pass", "pass_with_minor_risk"} else 1

        if args.command == "write":
            result = write_file(
                original_path=Path(args.original),
                candidate_path=Path(args.candidate),
                output_path=args.output,
                mode=RewriteMode.from_value(args.mode),
                dry_run=args.dry_run,
            )
            _print_write_summary(result)
            return 0 if result.output_written or args.dry_run else 1

        if args.command == "rewrite":
            result = rewrite_file(
                path=Path(args.path),
                mode=RewriteMode.from_value(args.mode),
                dry_run=True,
                output_path=args.output,
                debug_rewrite=args.debug_rewrite,
                strict_mode=args.strict_mode,
            )
            if args.debug_rewrite and result.debug_log:
                print("Rewrite debug trace:")
                for line in result.debug_log:
                    print(f"- {line}")
            print(f"Requested mode: {result.requested_mode.value}")
            print(f"Final mode used: {result.mode_used.value}")
            print(f"Reviewer decision: {result.review.decision}")
            print(f"Format integrity: {'pass' if result.review.format_integrity else 'fail'}")
            print(f"Core content integrity: {'pass' if result.review.core_content_integrity else 'fail'}")
            print(f"English spacing integrity: {'pass' if result.review.english_spacing_integrity else 'fail'}")
            print(f"Placeholder integrity: {'pass' if result.review.placeholder_integrity else 'fail'}")
            print(f"Effective change: {'true' if result.effective_change else 'false'}")
            print(f"Discourse change score: {result.review.discourse_change_score}")
            print(f"Cluster rewrite score: {result.review.cluster_rewrite_score}")
            print(f"Style variation score: {result.review.style_variation_score}")
            print(f"Rewrite coverage: {result.review.rewrite_coverage:.2f}")
            print("Output written: no")
            print(f"Output path: {result.output_path}")
            print(f"Overall guidance risk: {result.guidance.document_risk}")
            print(f"Write gate decision: {result.write_gate.decision}")
            if result.selected_candidate_reason:
                print(f"Selected candidate reason: {result.selected_candidate_reason}")
            if result.skipped_write_reason:
                print(f"Skipped write reason: {result.skipped_write_reason}")
            if result.review.failed_block_ids:
                print(f"Failed block ids: {result.review.failed_block_ids}")

            warnings = result.review.warnings + result.review.problems + result.write_gate.warnings
            if warnings:
                print("Warnings:")
                for item in warnings:
                    print(f"- {item}")

            print("Output file was not written. Use the write command after the candidate passes review.")
            print(result.diff or "No visible diff was produced in preview mode.")
            return 0 if result.review.decision in {"pass", "pass_with_minor_risk"} else 1

        if args.command == "suggest":
            result = suggest_file(Path(args.path))
            if not result.suggestions:
                print("No obvious vague paragraphs were detected.")
                return 0

            print(f"Suggestions for: {result.source_path}")
            for suggestion in result.suggestions:
                print(f"{suggestion.paragraph_index}. {suggestion.excerpt}")
                for item in suggestion.suggestions:
                    print(f"   - {item}")
            return 0

        parser.print_help()
        return 1
    except (ValidationError, FileOperationError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1


def _print_guide_summary(result) -> None:
    print(f"Source path: {result.source_path}")
    print(f"Document risk: {result.document_risk}")
    print(f"Write gate decision: {result.write_gate_decision}")
    print(f"Do not touch blocks: {len(result.do_not_touch_blocks)}")
    print(f"High risk blocks: {len(result.high_risk_blocks)}")
    print(f"Light edit blocks: {len(result.light_edit_blocks)}")
    print(f"Rewritable blocks: {len(result.rewritable_blocks)}")
    print("Format integrity status:")
    for key, value in result.format_integrity_status.items():
        print(f"- {key}: {value}")
    if result.core_protected_terms:
        print("Core protected terms:")
        for term in result.core_protected_terms[:15]:
            print(f"- {term}")
    print("Block policies:")
    for block in result.block_policies:
        print(f"{block.block_id}. [{block.edit_policy}] {block.block_type}")
        if block.preview:
            print(f"   preview: {block.preview}")
        if block.notes:
            print(f"   notes: {', '.join(block.notes)}")
        print(f"   rewrite depth: {block.rewrite_depth}")
        print(f"   rewrite intensity: {block.rewrite_intensity}")
        if block.required_structural_actions:
            print(f"   required: {', '.join(block.required_structural_actions)}")
        if block.required_discourse_actions:
            print(f"   required discourse: {', '.join(block.required_discourse_actions)}")
        print(
            f"   minimum changes: sentence>={block.required_minimum_sentence_level_changes}, "
            f"cluster>={block.required_minimum_cluster_changes}"
        )
        if block.optional_actions:
            print(f"   optional: {', '.join(block.optional_actions)}")
        if block.recommended_actions:
            print(f"   actions: {', '.join(block.recommended_actions)}")
        if block.forbidden_actions:
            print(f"   forbidden: {', '.join(block.forbidden_actions)}")


def _print_agent_context(result) -> None:
    schema = SkillInputSchema.from_path(result.source_path or "", preset="academic_natural")
    plan = build_execution_plan(result, schema)
    print(plan.agent_instruction)
    print("")
    print("AIRC agent context")
    print(f"- Document risk: {result.document_risk}")
    print(f"- Write gate preconditions: {'; '.join(result.write_gate_preconditions)}")
    print("- Core protected patterns:")
    for item in result.core_protected_patterns:
        print(f"  - {item}")
    print("- Format protected patterns:")
    for item in result.format_protected_patterns:
        print(f"  - {item}")
    print("- Naturalness priorities:")
    for item in result.naturalness_priorities:
        print(f"  - {item}")
    print("- Do not touch blocks:")
    for block in result.do_not_touch_blocks[:40]:
        print(f"  - Block {block.block_id} ({block.block_type}): {block.preview or '[blank]'}")
        print(f"    Required: {', '.join(block.required_structural_actions) if block.required_structural_actions else 'none'}")
        print(f"    Recommended: {', '.join(block.recommended_actions) if block.recommended_actions else 'keep_original'}")
        print(f"    Forbidden: {', '.join(block.forbidden_actions) if block.forbidden_actions else 'none'}")
    print("- High risk blocks:")
    for block in result.high_risk_blocks[:40]:
        print(f"  - Block {block.block_id}: {block.preview or '[blank]'}")
        print(f"    Required: {', '.join(block.required_structural_actions) if block.required_structural_actions else 'none'}")
        print(f"    Recommended: {', '.join(block.recommended_actions) if block.recommended_actions else 'keep_original'}")
        print(f"    Forbidden: {', '.join(block.forbidden_actions) if block.forbidden_actions else 'none'}")
    print("- Rewritable blocks:")
    for block in [*result.light_edit_blocks, *result.rewritable_blocks][:60]:
        print(f"  - Block {block.block_id} ({block.edit_policy}): {block.preview or '[blank]'}")
        print(f"    Rewrite depth: {block.rewrite_depth}")
        print(f"    Rewrite intensity: {block.rewrite_intensity}")
        print(f"    Required: {', '.join(block.required_structural_actions) if block.required_structural_actions else 'none'}")
        print(
            f"    Required discourse: {', '.join(block.required_discourse_actions) if block.required_discourse_actions else 'none'}"
        )
        print(
            f"    Minimum changes: sentence>={block.required_minimum_sentence_level_changes}, "
            f"cluster>={block.required_minimum_cluster_changes}"
        )
        print(f"    Recommended: {', '.join(block.recommended_actions) if block.recommended_actions else 'keep_original'}")
        print(f"    Forbidden: {', '.join(block.forbidden_actions) if block.forbidden_actions else 'none'}")
    print("- Agent notes:")
    for note in result.agent_notes:
        print(f"  - {note}")
    print("- Final self-checklist:")
    checklist = [
        "Titles and heading hierarchy unchanged.",
        "Formulas, citations, technical terms, numbers, paths, captions, placeholders unchanged.",
        "No broken English spacing, no '..', no '：。', no extra trailing spaces or broken linebreaks.",
        "Repeated subject chains reduced without weakening reference clarity.",
        "If a rewrite sounds stiff, keep the original sentence.",
    ]
    for item in checklist:
        print(f"  - {item}")


def _print_run_summary(result) -> None:
    review = result.rewrite_result.review
    gate = result.rewrite_result.write_gate
    print("AIRC public run summary")
    print(f"Source path: {result.source_path}")
    print(f"Preset: {result.execution_plan.preset}")
    print(f"Mode: {result.execution_plan.mode}")
    print(f"Reviewer decision: {review.decision}")
    print(f"Write gate decision: {gate.decision}")
    print(f"Output written: {'yes' if result.output_written else 'no'}")
    print(f"Output path: {result.output_path}")
    print(f"Rewrite report written: {'yes' if result.report_written else 'no'}")
    print(f"Rewrite report path: {result.report_path}")
    print(f"Original input type: {result.input_normalization.original_type}")
    print(f"Normalized input type: {result.input_normalization.normalized_type}")
    print(f"Converter used: {result.input_normalization.converter_used}")
    print(f"Pandoc used: {'yes' if result.input_normalization.pandoc_used else 'no'}")
    print(f"Normalization success: {'yes' if result.input_normalization.normalization_success else 'no'}")
    print(f"Rewrite coverage: {review.rewrite_coverage:.2f}")
    print(f"Discourse change score: {review.discourse_change_score}")
    print(f"Cluster rewrite score: {review.cluster_rewrite_score}")
    print(f"Blocks changed: {result.output_schema.blocks_changed}")
    print(f"Blocks skipped: {result.output_schema.blocks_skipped}")
    if result.rewrite_result.skipped_write_reason:
        print(f"Skipped write reason: {result.rewrite_result.skipped_write_reason}")
    if gate.reason_codes:
        print("Failure reason codes:" if not gate.write_allowed else "Write gate reason codes:")
        for reason in gate.reason_codes:
            print(f"- {reason}")
    if result.output_schema.warnings:
        print("Warnings:")
        for item in result.output_schema.warnings:
            print(f"- {item}")
    if result.rewrite_result.debug_log:
        print("Rewrite debug trace:")
        for line in result.rewrite_result.debug_log:
            print(f"- {line}")


def _print_review_summary(result) -> None:
    print(f"Original path: {result.original_path}")
    print(f"Candidate path: {result.candidate_path}")
    print(f"Reviewer decision: {result.review.decision}")
    print(f"Write gate decision: {result.write_gate.decision}")
    print(f"Format integrity: {'pass' if result.review.format_integrity else 'fail'}")
    print(f"Core content integrity: {'pass' if result.review.core_content_integrity else 'fail'}")
    print(f"English spacing integrity: {'pass' if result.review.english_spacing_integrity else 'fail'}")
    print(f"Placeholder integrity: {'pass' if result.review.placeholder_integrity else 'fail'}")
    print(f"Effective change: {'true' if result.review.effective_change else 'false'}")
    print(f"Naturalness score: {result.review.naturalness_score}")
    print(f"Discourse change score: {result.review.discourse_change_score}")
    print(f"Cluster rewrite score: {result.review.cluster_rewrite_score}")
    print(f"Style variation score: {result.review.style_variation_score}")
    print(f"Rewrite coverage: {result.review.rewrite_coverage:.2f}")
    if result.review.failed_block_ids:
        print(f"Failed block ids: {result.review.failed_block_ids}")
    if result.review.problems or result.review.warnings or result.write_gate.warnings:
        print("Warnings:")
        for item in [*result.review.problems, *result.review.warnings, *result.write_gate.warnings]:
            print(f"- {item}")


def _print_write_summary(result) -> None:
    print(f"Original path: {result.original_path}")
    print(f"Candidate path: {result.candidate_path}")
    print(f"Reviewer decision: {result.review.decision}")
    print(f"Write gate decision: {result.write_gate.decision}")
    print(f"Format integrity: {'pass' if result.review.format_integrity else 'fail'}")
    print(f"Core content integrity: {'pass' if result.review.core_content_integrity else 'fail'}")
    print(f"Discourse change score: {result.review.discourse_change_score}")
    print(f"Cluster rewrite score: {result.review.cluster_rewrite_score}")
    print(f"Style variation score: {result.review.style_variation_score}")
    print(f"Rewrite coverage: {result.review.rewrite_coverage:.2f}")
    print(f"Output written: {'yes' if result.output_written else 'no'}")
    print(f"Output path: {result.output_path}")
    if result.review.failed_block_ids:
        print(f"Failed block ids: {result.review.failed_block_ids}")
    if result.skipped_write_reason:
        print(f"Skipped write reason: {result.skipped_write_reason}")
    if result.write_gate.warnings or result.review.problems or result.review.warnings:
        print("Warnings:")
        for item in [*result.review.problems, *result.review.warnings, *result.write_gate.warnings]:
            print(f"- {item}")
    if result.dry_run:
        print(result.diff or "No visible diff was produced in dry-run mode.")


if __name__ == "__main__":
    raise SystemExit(main())
