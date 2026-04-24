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
        description="AIRC: Artificial Intelligence Rewrite Content. Guide first, review, then write only after integrity checks pass.",
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
    run_parser.add_argument(
        "--target-style-file",
        help="Optional reference file used for target-style fitting without copying its wording.",
    )
    run_parser.add_argument(
        "--json-report",
        action="store_true",
        default=True,
        help="Emit a machine-readable JSON report. Enabled by default for non-dry runs.",
    )
    run_parser.add_argument(
        "--agent-context",
        action="store_true",
        help="Print the generated agent execution context after the run summary.",
    )
    run_parser.add_argument(
        "--max-retry-passes",
        type=int,
        help="Override the preset retry budget for block-level coverage/discourse retry.",
    )
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
                target_style_file=args.target_style_file,
                dry_run=args.dry_run,
                debug_rewrite=args.debug_rewrite,
                keep_intermediate=args.keep_normalized,
                emit_agent_context=args.agent_context,
                emit_json_report=args.json_report,
                max_retry_passes=args.max_retry_passes,
            )
            _print_run_summary(result)
            if args.agent_context:
                print("")
                print("AIRC agent execution context")
                print(result.agent_instructions)
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
            print(f"Body rewrite coverage: {result.review.body_rewrite_coverage:.2f}")
            print(f"Body changed blocks: {result.review.body_changed_blocks}/{result.review.body_blocks_total}")
            print(f"Document scale: {result.review.document_scale}")
            print(f"Rewrite quota met: {'yes' if result.review.rewrite_quota_met else 'no'}")
            print(f"Human-like variation: {'yes' if result.review.human_like_variation else 'no'}")
            print(f"Non-uniform rewrite distribution: {'yes' if result.review.non_uniform_rewrite_distribution else 'no'}")
            print(f"Sentence cluster changes present: {'yes' if result.review.sentence_cluster_changes_present else 'no'}")
            print(f"Narrative flow changed: {'yes' if result.review.narrative_flow_changed else 'no'}")
            _print_academic_sentence_naturalization_metrics(result.review)
            _print_sentence_readability_metrics(result.review)
            _print_chapter_metrics(result.review)
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
    print(f"Body blocks: {result.body_blocks_total}")
    print(f"Body sentences: {result.body_sentences_total}")
    print(f"Document scale: {result.document_scale}")
    if result.chapter_policy_summary:
        print("Chapter policy summary:")
        for item in result.chapter_policy_summary:
            print(
                f"- {item.get('chapter_title')}: type={item.get('chapter_type')}, "
                f"priority={item.get('chapter_rewrite_priority')}, "
                f"intensity={item.get('chapter_rewrite_intensity')}, quota={item.get('chapter_rewrite_quota')}"
            )
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
        print(f"   chapter type: {block.chapter_type}")
        print(f"   chapter priority: {block.chapter_rewrite_priority}")
        print(f"   chapter quota: {block.chapter_rewrite_quota}")
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
    print(f"- Document scale: {result.document_scale}")
    print(f"- Body blocks: {result.body_blocks_total}")
    print(f"- Body sentences: {result.body_sentences_total}")
    print("- Chapter policy summary:")
    for item in result.chapter_policy_summary:
        print(
            f"  - {item.get('chapter_title')}: type={item.get('chapter_type')}, "
            f"priority={item.get('chapter_rewrite_priority')}, quota={item.get('chapter_rewrite_quota')}"
        )
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
        print(f"    Chapter: {block.chapter_title} ({block.chapter_type}, {block.chapter_rewrite_priority})")
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
    print(f"Status: {result.output_schema.status}")
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
    print(f"Body rewrite coverage: {review.body_rewrite_coverage:.2f}")
    print(f"Body changed blocks: {review.body_changed_blocks}/{review.body_blocks_total}")
    print(f"Document scale: {review.document_scale}")
    print(f"Rewrite quota met: {'yes' if review.rewrite_quota_met else 'no'}")
    print(f"Human-like variation: {'yes' if review.human_like_variation else 'no'}")
    print(f"Non-uniform rewrite distribution: {'yes' if review.non_uniform_rewrite_distribution else 'no'}")
    print(f"Sentence cluster changes present: {'yes' if review.sentence_cluster_changes_present else 'no'}")
    print(f"Narrative flow changed: {'yes' if review.narrative_flow_changed else 'no'}")
    print(f"Local transition natural: {'yes' if review.local_transition_natural else 'no'}")
    print(f"Local discourse not flat: {'yes' if review.local_discourse_not_flat else 'no'}")
    print(f"Sentence uniformity reduced: {'yes' if review.sentence_uniformity_reduced else 'no'}")
    print(f"Revision realism present: {'yes' if review.revision_realism_present else 'no'}")
    print(f"Stylistic uniformity controlled: {'yes' if review.stylistic_uniformity_controlled else 'no'}")
    print(f"Support sentence texture varied: {'yes' if review.support_sentence_texture_varied else 'no'}")
    print(f"Paragraph voice variation present: {'yes' if review.paragraph_voice_variation_present else 'no'}")
    print(f"Academic cliche density controlled: {'yes' if review.academic_cliche_density_controlled else 'no'}")
    print(f"Revision realism score: {float(review.local_revision_realism.get('revision_realism_score', 0.0)):.2f}")
    print(
        "Sentence transition rigidity: "
        f"{float(review.local_revision_realism.get('sentence_transition_rigidity', 0.0)):.2f}"
    )
    print(
        "Local discourse flatness: "
        f"{float(review.local_revision_realism.get('local_discourse_flatness', 0.0)):.2f}"
    )
    print(
        "Sentence cadence irregularity: "
        f"{float(review.local_revision_realism.get('sentence_cadence_irregularity', 0.0)):.2f}"
    )
    print(
        "Stylistic uniformity score: "
        f"{float(review.local_revision_realism.get('stylistic_uniformity_score', 0.0)):.2f}"
    )
    print(
        "Support sentence texture variation: "
        f"{float(review.local_revision_realism.get('support_sentence_texture_variation', 0.0)):.2f}"
    )
    print(
        "Paragraph voice variation: "
        f"{float(review.local_revision_realism.get('paragraph_voice_variation', 0.0)):.2f}"
    )
    print(
        "Academic cliche density: "
        f"{float(review.local_revision_realism.get('academic_cliche_density', 0.0)):.2f}"
    )
    print(f"Semantic role integrity preserved: {'yes' if review.semantic_role_integrity_preserved else 'no'}")
    print(f"Enumeration integrity preserved: {'yes' if review.enumeration_integrity_preserved else 'no'}")
    print(
        "Scaffolding phrase density controlled: "
        f"{'yes' if review.scaffolding_phrase_density_controlled else 'no'}"
    )
    print(
        "Over-abstracted subject risk controlled: "
        f"{'yes' if review.over_abstracted_subject_risk_controlled else 'no'}"
    )
    print(
        "Semantic role integrity score: "
        f"{float(review.semantic_role_integrity.get('semantic_role_integrity_score', 0.0)):.2f}"
    )
    print(
        "Enumeration integrity score: "
        f"{float(review.semantic_role_integrity.get('enumeration_integrity_score', 0.0)):.2f}"
    )
    print(
        "Scaffolding phrase density: "
        f"{float(review.semantic_role_integrity.get('scaffolding_phrase_density', 0.0)):.2f}"
    )
    print(
        "Over-abstracted subject risk: "
        f"{float(review.semantic_role_integrity.get('over_abstracted_subject_risk', 0.0)):.2f}"
    )
    print(f"Assertion strength preserved: {'yes' if review.assertion_strength_preserved else 'no'}")
    print(f"Appendix-like support controlled: {'yes' if review.appendix_like_support_controlled else 'no'}")
    print(f"Authorial stance present: {'yes' if review.authorial_stance_present else 'no'}")
    print(
        "Assertion strength score: "
        f"{float(review.authorial_intent.get('assertion_strength_score', 0.0)):.2f}"
    )
    print(
        "Appendix-like support ratio: "
        f"{float(review.authorial_intent.get('appendix_like_support_ratio', 0.0)):.2f}"
    )
    print(
        "Authorial stance presence: "
        f"{float(review.authorial_intent.get('authorial_stance_presence', 0.0)):.2f}"
    )
    print(f"Evidence fidelity preserved: {'yes' if review.evidence_fidelity_preserved else 'no'}")
    print(f"Unsupported expansion controlled: {'yes' if review.unsupported_expansion_controlled else 'no'}")
    print(f"Thesis tone restrained: {'yes' if review.thesis_tone_restrained else 'no'}")
    print(
        "Metaphor/storytelling controlled: "
        f"{'yes' if review.metaphor_or_storytelling_controlled else 'no'}"
    )
    print(
        "Authorial claim risk controlled: "
        f"{'yes' if review.authorial_claim_risk_controlled else 'no'}"
    )
    print(
        "Evidence fidelity score: "
        f"{float(review.evidence_fidelity.get('evidence_fidelity_score', 0.0)):.2f}"
    )
    print(
        "Unsupported expansion risk: "
        f"{float(review.evidence_fidelity.get('unsupported_expansion_risk', 0.0)):.2f}"
    )
    print(
        "Thesis tone restraint score: "
        f"{float(review.evidence_fidelity.get('thesis_tone_restraint_score', 0.0)):.2f}"
    )
    print(
        "Metaphor/storytelling risk: "
        f"{float(review.evidence_fidelity.get('metaphor_or_storytelling_risk', 0.0)):.2f}"
    )
    print(
        "Unjustified authorial claim risk: "
        f"{float(review.evidence_fidelity.get('unjustified_authorial_claim_risk', 0.0)):.2f}"
    )
    _print_academic_sentence_naturalization_metrics(review)
    _print_sentence_readability_metrics(review)
    _print_chapter_metrics(review)
    print(f"Discourse change score: {review.discourse_change_score}")
    print(f"Cluster rewrite score: {review.cluster_rewrite_score}")
    print(f"Blocks changed: {result.output_schema.blocks_changed}")
    print(f"Blocks skipped: {result.output_schema.blocks_skipped}")
    print(f"Blocks rejected: {result.output_schema.blocks_rejected}")
    if result.output_schema.failed_obligations:
        print("Failed obligations:")
        for block_id, obligations in result.output_schema.failed_obligations.items():
            print(f"- Block {block_id}: {', '.join(obligations)}")
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
    print(f"Body rewrite coverage: {result.review.body_rewrite_coverage:.2f}")
    print(f"Body changed blocks: {result.review.body_changed_blocks}/{result.review.body_blocks_total}")
    print(f"Document scale: {result.review.document_scale}")
    print(f"Rewrite quota met: {'yes' if result.review.rewrite_quota_met else 'no'}")
    print(f"Human-like variation: {'yes' if result.review.human_like_variation else 'no'}")
    print(f"Non-uniform rewrite distribution: {'yes' if result.review.non_uniform_rewrite_distribution else 'no'}")
    print(f"Sentence cluster changes present: {'yes' if result.review.sentence_cluster_changes_present else 'no'}")
    print(f"Narrative flow changed: {'yes' if result.review.narrative_flow_changed else 'no'}")
    _print_academic_sentence_naturalization_metrics(result.review)
    _print_sentence_readability_metrics(result.review)
    _print_chapter_metrics(result.review)
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
    print(f"Body rewrite coverage: {result.review.body_rewrite_coverage:.2f}")
    print(f"Body changed blocks: {result.review.body_changed_blocks}/{result.review.body_blocks_total}")
    print(f"Document scale: {result.review.document_scale}")
    print(f"Rewrite quota met: {'yes' if result.review.rewrite_quota_met else 'no'}")
    print(f"Human-like variation: {'yes' if result.review.human_like_variation else 'no'}")
    print(f"Non-uniform rewrite distribution: {'yes' if result.review.non_uniform_rewrite_distribution else 'no'}")
    print(f"Sentence cluster changes present: {'yes' if result.review.sentence_cluster_changes_present else 'no'}")
    print(f"Narrative flow changed: {'yes' if result.review.narrative_flow_changed else 'no'}")
    _print_academic_sentence_naturalization_metrics(result.review)
    _print_sentence_readability_metrics(result.review)
    _print_chapter_metrics(result.review)
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


def _print_chapter_metrics(review) -> None:
    print(f"Chapter rewrite quota met: {'yes' if review.chapter_rewrite_quota_check else 'no'}")
    print(f"Chapter policy consistent: {'yes' if review.chapter_policy_consistency_check else 'no'}")
    if not review.chapter_rewrite_metrics:
        return
    print("Chapter metrics:")
    for metric in review.chapter_rewrite_metrics:
        print(
            f"- {metric.get('chapter_title')}: "
            f"type={metric.get('chapter_type')}, "
            f"priority={metric.get('chapter_rewrite_priority')}, "
            f"coverage={float(metric.get('chapter_rewrite_coverage', 0.0)):.2f}, "
            f"changed={metric.get('chapter_changed_blocks')}/{metric.get('chapter_blocks_total')}, "
            f"discourse={metric.get('chapter_discourse_change_score')}, "
            f"cluster={metric.get('chapter_cluster_rewrite_score')}, "
            f"quota={'yes' if metric.get('chapter_rewrite_quota_met') else 'no'}"
        )


def _print_academic_sentence_naturalization_metrics(review) -> None:
    metrics = getattr(review, "academic_sentence_naturalization", {}) or {}
    print(f"Bureaucratic opening controlled: {'yes' if review.bureaucratic_opening_controlled else 'no'}")
    print(f"Explicit subject chain controlled: {'yes' if review.explicit_subject_chain_controlled else 'no'}")
    print(f"Overstructured syntax controlled: {'yes' if review.overstructured_syntax_controlled else 'no'}")
    print(f"Main clause position reasonable: {'yes' if review.main_clause_position_reasonable else 'no'}")
    print(f"Slogan-like goal phrase controlled: {'yes' if review.slogan_like_goal_phrase_controlled else 'no'}")
    print(f"Bureaucratic opening density: {float(metrics.get('bureaucratic_opening_density', 0.0)):.2f}")
    print(f"Repeated explicit subject risk: {float(metrics.get('repeated_explicit_subject_risk', 0.0)):.2f}")
    print(f"Overstructured syntax risk: {float(metrics.get('overstructured_syntax_risk', 0.0)):.2f}")
    print(f"Delayed main clause risk: {float(metrics.get('delayed_main_clause_risk', 0.0)):.2f}")
    print(f"Slogan-like goal risk: {float(metrics.get('slogan_like_goal_risk', 0.0)):.2f}")
    print(f"Author style alignment controlled: {'yes' if metrics.get('author_style_alignment_controlled', True) else 'no'}")
    print(f"Directness score: {float(metrics.get('directness_score', 0.0)):.2f}")
    print(f"Connector overuse risk: {float(metrics.get('connector_overuse_risk', 0.0)):.2f}")
    print(f"Nominalization density: {float(metrics.get('nominalization_density', 0.0)):.2f}")
    print(f"Passive voice ratio: {float(metrics.get('passive_voice_ratio', 0.0)):.2f}")
    print(f"Overlong sentence risk: {float(metrics.get('overlong_sentence_risk', 0.0)):.2f}")
    print(f"Subject monotony risk: {float(metrics.get('subject_monotony_risk', 0.0)):.2f}")
    l2 = getattr(review, "l2_style_profile", {}) or {}
    print(f"L2 profile enabled: {'yes' if l2.get('enabled', False) else 'no'}")
    print(f"L2 texture present: {'yes' if l2.get('l2_texture_present', True) else 'no'}")
    print(f"L2 texture score: {float(l2.get('l2_texture_score', 0.0)):.2f}")
    print(f"L2 function word density: {float(l2.get('function_word_density', 0.0)):.2f}")
    print(f"L2 native-like concision risk: {float(l2.get('native_like_concision_risk', 0.0)):.2f}")
    print(f"L2 colloquial risk: {float(l2.get('colloquial_risk', 0.0)):.2f}")
    print(f"L2 ungrammatical risk: {float(l2.get('ungrammatical_risk', 0.0)):.2f}")
    alignment = getattr(review, "target_style_alignment", {}) or {}
    print(f"Target style alignment enabled: {'yes' if alignment.get('enabled', False) else 'no'}")
    print(f"Target style alignment score: {float(alignment.get('target_style_alignment_score', 1.0)):.2f}")
    print(f"Style distribution match ratio: {float(alignment.get('style_distribution_match_ratio', 1.0)):.2f}")
    print(f"Average sentence length diff: {float(alignment.get('avg_sentence_length_diff', 0.0)):.2f}")
    print(f"Clause per sentence diff: {float(alignment.get('clause_per_sentence_diff', 0.0)):.2f}")
    print(f"Main clause position diff: {float(alignment.get('main_clause_position_diff', 0.0)):.2f}")
    print(f"Function word density diff: {float(alignment.get('function_word_density_diff', 0.0)):.2f}")
    print(f"Helper verb usage diff: {float(alignment.get('helper_verb_usage_diff', 0.0)):.2f}")
    print(f"Explanatory rewrite gap: {float(alignment.get('explanatory_rewrite_gap', 0.0)):.2f}")
    print(f"Compactness gap: {float(alignment.get('compactness_gap', 0.0)):.2f}")
    print(f"Native fluency gap: {float(alignment.get('native_fluency_gap', 0.0)):.2f}")
    print(f"L2 texture gap: {float(alignment.get('l2_texture_gap', 0.0)):.2f}")
    print(f"Grammar error rate: {float(alignment.get('grammar_error_rate', 0.0)):.2f}")
    print(f"Terminology drift: {int(alignment.get('terminology_drift', 0))}")
    print(f"Evidence drift: {int(alignment.get('evidence_drift', 0))}")


def _print_sentence_readability_metrics(review) -> None:
    metrics = getattr(review, "sentence_readability", {}) or {}
    print(f"Sentence completeness preserved: {'yes' if review.sentence_completeness_preserved else 'no'}")
    print(f"Paragraph readability preserved: {'yes' if review.paragraph_readability_preserved else 'no'}")
    print(f"No dangling support sentences: {'yes' if review.no_dangling_support_sentences else 'no'}")
    print(
        "No fragment-like conclusion sentences: "
        f"{'yes' if review.no_fragment_like_conclusion_sentences else 'no'}"
    )
    print(f"Sentence completeness score: {float(metrics.get('sentence_completeness_score', 0.0)):.2f}")
    print(f"Paragraph readability score: {float(metrics.get('paragraph_readability_score', 0.0)):.2f}")
    print(f"Dangling sentence risk: {float(metrics.get('dangling_sentence_risk', 0.0)):.2f}")
    print(f"Incomplete support sentence risk: {float(metrics.get('incomplete_support_sentence_risk', 0.0)):.2f}")
    print(f"Fragment-like conclusion risk: {float(metrics.get('fragment_like_conclusion_risk', 0.0)):.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
