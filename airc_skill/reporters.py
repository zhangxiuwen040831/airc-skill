from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .config import DEFAULT_CONFIG
from .models import ReviewReport, WriteGateDecision
from .skill_protocol import SkillExecutionPlan, SkillInputSchema, SkillOutputSchema


def build_failure_transparency(
    *,
    output_schema: SkillOutputSchema,
    review: ReviewReport,
    block_failures: list[str],
) -> dict[str, Any]:
    suggested_retry_intensity = (
        "high"
        if (
            review.body_rewrite_coverage < review.required_body_rewrite_coverage
            or review.body_discourse_change_score < review.required_body_discourse_change_score
            or review.body_cluster_rewrite_score < review.required_body_cluster_rewrite_score
            or not review.rewrite_quota_met
            or not review.chapter_rewrite_quota_check
            or not review.paragraph_opening_style_valid
            or not review.paragraph_skeleton_consistent
            or not review.revision_realism_present
            or not review.sentence_completeness_preserved
            or not review.paragraph_readability_preserved
        )
        else "current"
    )
    if output_schema.blocks_rejected:
        suggested_next_step = "Retry rejected blocks with the suggested intensity and required actions."
    elif output_schema.write_allowed:
        suggested_next_step = "No retry required; candidate passed the write gate."
    else:
        suggested_next_step = "Review warnings and retry with higher rewrite intensity if the output is too conservative."

    return {
        "failed_block_ids": list(output_schema.blocks_rejected),
        "failed_obligations": output_schema.failed_obligations,
        "current_rewrite_coverage": review.rewrite_coverage,
        "current_body_rewrite_coverage": review.body_rewrite_coverage,
        "current_body_changed_blocks": review.body_changed_blocks,
        "current_body_blocks_total": review.body_blocks_total,
        "document_scale": review.document_scale,
        "rewrite_quota_met": review.rewrite_quota_met,
        "rewrite_quota_reason_codes": list(review.rewrite_quota_reason_codes),
        "human_like_variation": review.human_like_variation,
        "non_uniform_rewrite_distribution": review.non_uniform_rewrite_distribution,
        "sentence_cluster_changes_present": review.sentence_cluster_changes_present,
        "narrative_flow_changed": review.narrative_flow_changed,
        "revision_pattern_distribution": dict(review.revision_pattern_distribution),
        "human_noise_marks": list(review.human_noise_marks),
        "chapter_rewrite_metrics": list(review.chapter_rewrite_metrics),
        "chapter_policy_consistency_check": review.chapter_policy_consistency_check,
        "chapter_rewrite_quota_check": review.chapter_rewrite_quota_check,
        "chapter_rewrite_quota_reason_codes": list(review.chapter_rewrite_quota_reason_codes),
        "paragraph_topic_sentence_preserved": review.paragraph_topic_sentence_preserved,
        "paragraph_opening_style_valid": review.paragraph_opening_style_valid,
        "paragraph_skeleton_consistent": review.paragraph_skeleton_consistent,
        "no_dangling_opening_sentence": review.no_dangling_opening_sentence,
        "topic_sentence_not_demoted_to_mid_paragraph": review.topic_sentence_not_demoted_to_mid_paragraph,
        "paragraph_skeleton_review": dict(review.paragraph_skeleton_review),
        "local_transition_natural": review.local_transition_natural,
        "local_discourse_not_flat": review.local_discourse_not_flat,
        "sentence_uniformity_reduced": review.sentence_uniformity_reduced,
        "revision_realism_present": review.revision_realism_present,
        "stylistic_uniformity_controlled": review.stylistic_uniformity_controlled,
        "support_sentence_texture_varied": review.support_sentence_texture_varied,
        "paragraph_voice_variation_present": review.paragraph_voice_variation_present,
        "academic_cliche_density_controlled": review.academic_cliche_density_controlled,
        "local_revision_realism": dict(review.local_revision_realism),
        "sentence_completeness_preserved": review.sentence_completeness_preserved,
        "paragraph_readability_preserved": review.paragraph_readability_preserved,
        "no_dangling_support_sentences": review.no_dangling_support_sentences,
        "no_fragment_like_conclusion_sentences": review.no_fragment_like_conclusion_sentences,
        "sentence_readability": dict(review.sentence_readability),
        "semantic_role_integrity_preserved": review.semantic_role_integrity_preserved,
        "enumeration_integrity_preserved": review.enumeration_integrity_preserved,
        "scaffolding_phrase_density_controlled": review.scaffolding_phrase_density_controlled,
        "over_abstracted_subject_risk_controlled": review.over_abstracted_subject_risk_controlled,
        "semantic_role_integrity": dict(review.semantic_role_integrity),
        "assertion_strength_preserved": review.assertion_strength_preserved,
        "appendix_like_support_controlled": review.appendix_like_support_controlled,
        "authorial_stance_present": review.authorial_stance_present,
        "authorial_intent": dict(review.authorial_intent),
        "evidence_fidelity_preserved": review.evidence_fidelity_preserved,
        "unsupported_expansion_controlled": review.unsupported_expansion_controlled,
        "thesis_tone_restrained": review.thesis_tone_restrained,
        "metaphor_or_storytelling_controlled": review.metaphor_or_storytelling_controlled,
        "authorial_claim_risk_controlled": review.authorial_claim_risk_controlled,
        "evidence_fidelity": dict(review.evidence_fidelity),
        "bureaucratic_opening_controlled": review.bureaucratic_opening_controlled,
        "explicit_subject_chain_controlled": review.explicit_subject_chain_controlled,
        "overstructured_syntax_controlled": review.overstructured_syntax_controlled,
        "main_clause_position_reasonable": review.main_clause_position_reasonable,
        "slogan_like_goal_phrase_controlled": review.slogan_like_goal_phrase_controlled,
        "academic_sentence_naturalization": dict(review.academic_sentence_naturalization),
        "l2_style_profile": dict(review.l2_style_profile),
        "current_discourse_change_score": review.discourse_change_score,
        "current_cluster_rewrite_score": review.cluster_rewrite_score,
        "target_style_alignment": dict(review.target_style_alignment),
        "suggested_next_step": suggested_next_step,
        "suggested_retry_intensity": suggested_retry_intensity,
        "block_failures": list(block_failures),
    }


def build_json_report(
    *,
    schema: SkillInputSchema,
    execution_plan: SkillExecutionPlan,
    output_schema: SkillOutputSchema,
    review: ReviewReport,
    write_gate: WriteGateDecision,
    input_normalization: dict[str, Any],
    candidate_scores: list[str],
    block_failures: list[str],
) -> dict[str, Any]:
    return {
        "input_schema": schema.to_dict(),
        "input_normalization": input_normalization,
        "execution_plan": execution_plan.to_dict(),
        "output_schema": output_schema.to_dict(),
        "review": {
            "decision": review.decision,
            "effective_change": review.effective_change,
            "rewrite_coverage": review.rewrite_coverage,
            "body_rewrite_coverage": review.body_rewrite_coverage,
            "body_changed_blocks": review.body_changed_blocks,
            "body_blocks_total": review.body_blocks_total,
            "body_changed_sentences": review.body_changed_sentences,
            "body_sentences_total": review.body_sentences_total,
            "body_discourse_change_score": review.body_discourse_change_score,
            "body_cluster_rewrite_score": review.body_cluster_rewrite_score,
            "document_scale": review.document_scale,
            "rewrite_quota_met": review.rewrite_quota_met,
            "rewrite_quota_reason_codes": review.rewrite_quota_reason_codes,
            "human_like_variation": review.human_like_variation,
            "non_uniform_rewrite_distribution": review.non_uniform_rewrite_distribution,
            "sentence_cluster_changes_present": review.sentence_cluster_changes_present,
            "narrative_flow_changed": review.narrative_flow_changed,
            "revision_pattern_distribution": review.revision_pattern_distribution,
            "human_noise_marks": review.human_noise_marks,
            "chapter_rewrite_metrics": review.chapter_rewrite_metrics,
            "chapter_policy_consistency_check": review.chapter_policy_consistency_check,
            "chapter_rewrite_quota_check": review.chapter_rewrite_quota_check,
            "chapter_rewrite_quota_reason_codes": review.chapter_rewrite_quota_reason_codes,
            "paragraph_topic_sentence_preserved": review.paragraph_topic_sentence_preserved,
            "paragraph_opening_style_valid": review.paragraph_opening_style_valid,
            "paragraph_skeleton_consistent": review.paragraph_skeleton_consistent,
            "no_dangling_opening_sentence": review.no_dangling_opening_sentence,
            "topic_sentence_not_demoted_to_mid_paragraph": review.topic_sentence_not_demoted_to_mid_paragraph,
            "paragraph_skeleton_review": review.paragraph_skeleton_review,
            "local_transition_natural": review.local_transition_natural,
            "local_discourse_not_flat": review.local_discourse_not_flat,
            "sentence_uniformity_reduced": review.sentence_uniformity_reduced,
            "revision_realism_present": review.revision_realism_present,
            "stylistic_uniformity_controlled": review.stylistic_uniformity_controlled,
            "support_sentence_texture_varied": review.support_sentence_texture_varied,
            "paragraph_voice_variation_present": review.paragraph_voice_variation_present,
            "academic_cliche_density_controlled": review.academic_cliche_density_controlled,
            "local_revision_realism": review.local_revision_realism,
            "sentence_completeness_preserved": review.sentence_completeness_preserved,
            "paragraph_readability_preserved": review.paragraph_readability_preserved,
            "no_dangling_support_sentences": review.no_dangling_support_sentences,
            "no_fragment_like_conclusion_sentences": review.no_fragment_like_conclusion_sentences,
              "sentence_readability": review.sentence_readability,
              "semantic_role_integrity_preserved": review.semantic_role_integrity_preserved,
              "enumeration_integrity_preserved": review.enumeration_integrity_preserved,
              "scaffolding_phrase_density_controlled": review.scaffolding_phrase_density_controlled,
              "over_abstracted_subject_risk_controlled": review.over_abstracted_subject_risk_controlled,
              "semantic_role_integrity": review.semantic_role_integrity,
              "assertion_strength_preserved": review.assertion_strength_preserved,
              "appendix_like_support_controlled": review.appendix_like_support_controlled,
              "authorial_stance_present": review.authorial_stance_present,
              "authorial_intent": review.authorial_intent,
              "evidence_fidelity_preserved": review.evidence_fidelity_preserved,
              "unsupported_expansion_controlled": review.unsupported_expansion_controlled,
              "thesis_tone_restrained": review.thesis_tone_restrained,
              "metaphor_or_storytelling_controlled": review.metaphor_or_storytelling_controlled,
              "authorial_claim_risk_controlled": review.authorial_claim_risk_controlled,
              "evidence_fidelity": review.evidence_fidelity,
              "bureaucratic_opening_controlled": review.bureaucratic_opening_controlled,
              "explicit_subject_chain_controlled": review.explicit_subject_chain_controlled,
              "overstructured_syntax_controlled": review.overstructured_syntax_controlled,
              "main_clause_position_reasonable": review.main_clause_position_reasonable,
              "slogan_like_goal_phrase_controlled": review.slogan_like_goal_phrase_controlled,
              "academic_sentence_naturalization": review.academic_sentence_naturalization,
              "l2_style_profile": review.l2_style_profile,
              "target_style_alignment": review.target_style_alignment,
              "discourse_change_score": review.discourse_change_score,
            "cluster_rewrite_score": review.cluster_rewrite_score,
            "style_variation_score": review.style_variation_score,
            "format_integrity": review.format_integrity,
            "core_content_integrity": review.core_content_integrity,
            "failed_block_ids": review.failed_block_ids,
            "problems": review.problems,
            "warnings": review.warnings,
            "natural_revision_checklist": review.natural_revision_checklist,
        },
        "write_gate": {
            "write_allowed": write_gate.write_allowed,
            "decision": write_gate.decision,
            "reason_codes": write_gate.reason_codes,
            "warnings": write_gate.warnings,
        },
        "failure_transparency": build_failure_transparency(
            output_schema=output_schema,
            review=review,
            block_failures=block_failures,
        ),
        "human_report": build_human_report(
            output_schema=output_schema,
            review=review,
            write_gate=write_gate,
        ),
        "candidate_scores": list(candidate_scores),
        "agent_instructions": execution_plan.agent_instructions,
        "agent_context": execution_plan.agent_instruction,
    }


def build_human_report(
    *,
    output_schema: SkillOutputSchema,
    review: ReviewReport,
    write_gate: WriteGateDecision,
) -> str:
    lines = [
        "AIRC Revision Report",
        f"Status: {output_schema.status}",
        f"Write gate: {write_gate.decision}",
        f"Output written: {'yes' if output_schema.write_allowed else 'no'}",
        f"Rewrite coverage: {review.rewrite_coverage:.2f}",
        f"Body rewrite coverage: {review.body_rewrite_coverage:.2f}",
        f"Body changed blocks: {review.body_changed_blocks}/{review.body_blocks_total}",
        f"Body changed sentences: {review.body_changed_sentences}/{review.body_sentences_total}",
        f"Document scale: {review.document_scale}",
        f"Rewrite quota met: {'yes' if review.rewrite_quota_met else 'no'}",
        f"Human-like variation: {'yes' if review.human_like_variation else 'no'}",
        f"Non-uniform rewrite distribution: {'yes' if review.non_uniform_rewrite_distribution else 'no'}",
        f"Sentence cluster changes present: {'yes' if review.sentence_cluster_changes_present else 'no'}",
        f"Narrative flow changed: {'yes' if review.narrative_flow_changed else 'no'}",
        f"Paragraph topic sentence preserved: {'yes' if review.paragraph_topic_sentence_preserved else 'no'}",
        f"Paragraph opening style valid: {'yes' if review.paragraph_opening_style_valid else 'no'}",
        f"Paragraph skeleton consistent: {'yes' if review.paragraph_skeleton_consistent else 'no'}",
        f"No dangling opening sentence: {'yes' if review.no_dangling_opening_sentence else 'no'}",
        f"Topic sentence not demoted: {'yes' if review.topic_sentence_not_demoted_to_mid_paragraph else 'no'}",
        f"Local transition natural: {'yes' if review.local_transition_natural else 'no'}",
        f"Local discourse not flat: {'yes' if review.local_discourse_not_flat else 'no'}",
        f"Sentence uniformity reduced: {'yes' if review.sentence_uniformity_reduced else 'no'}",
        f"Revision realism present: {'yes' if review.revision_realism_present else 'no'}",
        f"Stylistic uniformity controlled: {'yes' if review.stylistic_uniformity_controlled else 'no'}",
        f"Support sentence texture varied: {'yes' if review.support_sentence_texture_varied else 'no'}",
        f"Paragraph voice variation present: {'yes' if review.paragraph_voice_variation_present else 'no'}",
        f"Academic cliche density controlled: {'yes' if review.academic_cliche_density_controlled else 'no'}",
        f"Revision realism score: {float(review.local_revision_realism.get('revision_realism_score', 0.0)):.2f}",
        f"Sentence transition rigidity: {float(review.local_revision_realism.get('sentence_transition_rigidity', 0.0)):.2f}",
        f"Local discourse flatness: {float(review.local_revision_realism.get('local_discourse_flatness', 0.0)):.2f}",
        f"Sentence cadence irregularity: {float(review.local_revision_realism.get('sentence_cadence_irregularity', 0.0)):.2f}",
        f"Stylistic uniformity score: {float(review.local_revision_realism.get('stylistic_uniformity_score', 0.0)):.2f}",
        f"Support sentence texture variation: {float(review.local_revision_realism.get('support_sentence_texture_variation', 0.0)):.2f}",
        f"Paragraph voice variation: {float(review.local_revision_realism.get('paragraph_voice_variation', 0.0)):.2f}",
        f"Academic cliche density: {float(review.local_revision_realism.get('academic_cliche_density', 0.0)):.2f}",
        f"Sentence completeness preserved: {'yes' if review.sentence_completeness_preserved else 'no'}",
        f"Paragraph readability preserved: {'yes' if review.paragraph_readability_preserved else 'no'}",
        f"No dangling support sentences: {'yes' if review.no_dangling_support_sentences else 'no'}",
        f"No fragment-like conclusion sentences: {'yes' if review.no_fragment_like_conclusion_sentences else 'no'}",
        f"Sentence completeness score: {float(review.sentence_readability.get('sentence_completeness_score', 0.0)):.2f}",
        f"Paragraph readability score: {float(review.sentence_readability.get('paragraph_readability_score', 0.0)):.2f}",
        f"Semantic role integrity preserved: {'yes' if review.semantic_role_integrity_preserved else 'no'}",
        f"Enumeration integrity preserved: {'yes' if review.enumeration_integrity_preserved else 'no'}",
        f"Scaffolding phrase density controlled: {'yes' if review.scaffolding_phrase_density_controlled else 'no'}",
        f"Over-abstracted subject risk controlled: {'yes' if review.over_abstracted_subject_risk_controlled else 'no'}",
        f"Semantic role integrity score: {float(review.semantic_role_integrity.get('semantic_role_integrity_score', 0.0)):.2f}",
        f"Enumeration integrity score: {float(review.semantic_role_integrity.get('enumeration_integrity_score', 0.0)):.2f}",
        f"Scaffolding phrase density: {float(review.semantic_role_integrity.get('scaffolding_phrase_density', 0.0)):.2f}",
        f"Over-abstracted subject risk: {float(review.semantic_role_integrity.get('over_abstracted_subject_risk', 0.0)):.2f}",
        f"Assertion strength preserved: {'yes' if review.assertion_strength_preserved else 'no'}",
        f"Appendix-like support controlled: {'yes' if review.appendix_like_support_controlled else 'no'}",
        f"Authorial stance present: {'yes' if review.authorial_stance_present else 'no'}",
        f"Assertion strength score: {float(review.authorial_intent.get('assertion_strength_score', 0.0)):.2f}",
        f"Appendix-like support ratio: {float(review.authorial_intent.get('appendix_like_support_ratio', 0.0)):.2f}",
        f"Authorial stance presence: {float(review.authorial_intent.get('authorial_stance_presence', 0.0)):.2f}",
        f"Evidence fidelity preserved: {'yes' if review.evidence_fidelity_preserved else 'no'}",
        f"Unsupported expansion controlled: {'yes' if review.unsupported_expansion_controlled else 'no'}",
        f"Thesis tone restrained: {'yes' if review.thesis_tone_restrained else 'no'}",
        f"Metaphor/storytelling controlled: {'yes' if review.metaphor_or_storytelling_controlled else 'no'}",
        f"Authorial claim risk controlled: {'yes' if review.authorial_claim_risk_controlled else 'no'}",
        f"Evidence fidelity score: {float(review.evidence_fidelity.get('evidence_fidelity_score', 0.0)):.2f}",
        f"Unsupported expansion risk: {float(review.evidence_fidelity.get('unsupported_expansion_risk', 0.0)):.2f}",
        f"Thesis tone restraint score: {float(review.evidence_fidelity.get('thesis_tone_restraint_score', 0.0)):.2f}",
        f"Metaphor/storytelling risk: {float(review.evidence_fidelity.get('metaphor_or_storytelling_risk', 0.0)):.2f}",
        f"Unjustified authorial claim risk: {float(review.evidence_fidelity.get('unjustified_authorial_claim_risk', 0.0)):.2f}",
        f"Bureaucratic opening controlled: {'yes' if review.bureaucratic_opening_controlled else 'no'}",
        f"Explicit subject chain controlled: {'yes' if review.explicit_subject_chain_controlled else 'no'}",
        f"Overstructured syntax controlled: {'yes' if review.overstructured_syntax_controlled else 'no'}",
        f"Main clause position reasonable: {'yes' if review.main_clause_position_reasonable else 'no'}",
        f"Slogan-like goal phrase controlled: {'yes' if review.slogan_like_goal_phrase_controlled else 'no'}",
        f"Bureaucratic opening density: {float(review.academic_sentence_naturalization.get('bureaucratic_opening_density', 0.0)):.2f}",
        f"Repeated explicit subject risk: {float(review.academic_sentence_naturalization.get('repeated_explicit_subject_risk', 0.0)):.2f}",
        f"Overstructured syntax risk: {float(review.academic_sentence_naturalization.get('overstructured_syntax_risk', 0.0)):.2f}",
        f"Delayed main clause risk: {float(review.academic_sentence_naturalization.get('delayed_main_clause_risk', 0.0)):.2f}",
        f"Slogan-like goal risk: {float(review.academic_sentence_naturalization.get('slogan_like_goal_risk', 0.0)):.2f}",
        f"Author style alignment controlled: {'yes' if review.academic_sentence_naturalization.get('author_style_alignment_controlled', True) else 'no'}",
        f"Directness score: {float(review.academic_sentence_naturalization.get('directness_score', 0.0)):.2f}",
        f"Connector overuse risk: {float(review.academic_sentence_naturalization.get('connector_overuse_risk', 0.0)):.2f}",
        f"Nominalization density: {float(review.academic_sentence_naturalization.get('nominalization_density', 0.0)):.2f}",
        f"Passive voice ratio: {float(review.academic_sentence_naturalization.get('passive_voice_ratio', 0.0)):.2f}",
        f"Overlong sentence risk: {float(review.academic_sentence_naturalization.get('overlong_sentence_risk', 0.0)):.2f}",
        f"Subject monotony risk: {float(review.academic_sentence_naturalization.get('subject_monotony_risk', 0.0)):.2f}",
        f"L2 profile enabled: {'yes' if review.l2_style_profile.get('enabled', False) else 'no'}",
        f"L2 texture present: {'yes' if review.l2_style_profile.get('l2_texture_present', True) else 'no'}",
        f"L2 texture score: {float(review.l2_style_profile.get('l2_texture_score', 0.0)):.2f}",
        f"L2 function word density: {float(review.l2_style_profile.get('function_word_density', 0.0)):.2f}",
        f"L2 native-like concision risk: {float(review.l2_style_profile.get('native_like_concision_risk', 0.0)):.2f}",
        f"L2 colloquial risk: {float(review.l2_style_profile.get('colloquial_risk', 0.0)):.2f}",
        f"L2 ungrammatical risk: {float(review.l2_style_profile.get('ungrammatical_risk', 0.0)):.2f}",
        f"Target style alignment enabled: {'yes' if review.target_style_alignment.get('enabled', False) else 'no'}",
        f"Target style alignment score: {float(review.target_style_alignment.get('target_style_alignment_score', 1.0)):.2f}",
        f"Style distribution match ratio: {float(review.target_style_alignment.get('style_distribution_match_ratio', 1.0)):.2f}",
        f"Class-aware style match ratio: {float(review.target_style_alignment.get('class_aware_style_match_ratio', 1.0)):.2f}",
        f"Average sentence length diff: {float(review.target_style_alignment.get('avg_sentence_length_diff', 0.0)):.2f}",
        f"Clause per sentence diff: {float(review.target_style_alignment.get('clause_per_sentence_diff', 0.0)):.2f}",
        f"Main clause position diff: {float(review.target_style_alignment.get('main_clause_position_diff', 0.0)):.2f}",
        f"Function word density diff: {float(review.target_style_alignment.get('function_word_density_diff', 0.0)):.2f}",
        f"Helper verb usage diff: {float(review.target_style_alignment.get('helper_verb_usage_diff', 0.0)):.2f}",
        f"Explanatory rewrite gap: {float(review.target_style_alignment.get('explanatory_rewrite_gap', 0.0)):.2f}",
        f"Compactness gap: {float(review.target_style_alignment.get('compactness_gap', 0.0)):.2f}",
        f"Native fluency gap: {float(review.target_style_alignment.get('native_fluency_gap', 0.0)):.2f}",
        f"L2 texture gap: {float(review.target_style_alignment.get('l2_texture_gap', 0.0)):.2f}",
        f"Grammar error rate: {float(review.target_style_alignment.get('grammar_error_rate', 0.0)):.2f}",
        f"Terminology drift: {int(review.target_style_alignment.get('terminology_drift', 0))}",
        f"Evidence drift: {int(review.target_style_alignment.get('evidence_drift', 0))}",
        f"Chapter quota met: {'yes' if review.chapter_rewrite_quota_check else 'no'}",
        f"Chapter policy consistent: {'yes' if review.chapter_policy_consistency_check else 'no'}",
        f"Discourse change score: {review.discourse_change_score}",
        f"Cluster rewrite score: {review.cluster_rewrite_score}",
        f"Changed blocks: {output_schema.blocks_changed or ['none']}",
        f"Skipped blocks: {output_schema.blocks_skipped or ['none']}",
        f"Rejected blocks: {output_schema.blocks_rejected or ['none']}",
    ]
    if output_schema.failed_obligations:
        lines.append("Failed obligations:")
        for block_id, obligations in output_schema.failed_obligations.items():
            lines.append(f"- Block {block_id}: {', '.join(obligations)}")
    if review.natural_revision_checklist:
        lines.append("Natural revision checklist:")
        for key, passed in review.natural_revision_checklist.items():
            lines.append(f"- {key}: {'pass' if passed else 'fail'}")
    if review.chapter_rewrite_metrics:
        lines.append("Chapter rewrite metrics:")
        for metric in review.chapter_rewrite_metrics:
            lines.append(
                "- "
                f"{metric.get('chapter_title')}: "
                f"type={metric.get('chapter_type')}, "
                f"priority={metric.get('chapter_rewrite_priority')}, "
                f"coverage={float(metric.get('chapter_rewrite_coverage', 0.0)):.2f}, "
                f"changed={metric.get('chapter_changed_blocks')}/{metric.get('chapter_blocks_total')}, "
                f"discourse={metric.get('chapter_discourse_change_score')}, "
                f"cluster={metric.get('chapter_cluster_rewrite_score')}, "
                f"quota={'pass' if metric.get('chapter_rewrite_quota_met') else 'fail'}"
            )
    worst_classes = review.target_style_alignment.get("worst_alignment_classes", [])
    if worst_classes:
        lines.append("Worst target-style classes:")
        for item in worst_classes:
            lines.append(
                "- "
                f"{item.get('class_name')}: "
                f"match={float(item.get('class_style_match_ratio', 0.0)):.2f}, "
                f"paragraphs={item.get('paragraph_count', 0)}, "
                f"sentence_gap={float(item.get('sentence_length_gap', 0.0)):.2f}, "
                f"function_gap={float(item.get('function_word_gap', 0.0)):.2f}, "
                f"explanatory_gap={float(item.get('explanatory_gap', 0.0)):.2f}, "
                f"native_gap={float(item.get('native_fluency_gap', 0.0)):.2f}"
            )
    warnings = [*output_schema.warnings, *write_gate.warnings]
    if warnings:
        lines.append("Warnings:")
        for warning in dict.fromkeys(warnings):
            lines.append(f"- {warning}")
    return "\n".join(lines)


def dataclass_to_json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return {str(dataclass_to_json_safe(key)): dataclass_to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [dataclass_to_json_safe(item) for item in value]
    return value
