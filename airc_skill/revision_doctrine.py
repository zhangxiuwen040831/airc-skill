"""Public doctrine for AIRC.

The doctrine turns AIRC's style position into a machine-readable contract for
external agents. It is about academic revision quality, not adversarial goals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .natural_revision_profile import ACADEMIC_NATURAL_STUDENTLIKE, get_natural_revision_profile


@dataclass(frozen=True)
class RevisionDoctrine:
    public_name: str
    positioning: tuple[str, ...]
    thesis_draft_observations: tuple[str, ...]
    body_rewrite_methods: tuple[str, ...]
    block_strategies: dict[str, tuple[str, ...]]
    style_profile: str
    natural_actions: tuple[str, ...]
    forbidden_patterns: tuple[str, ...]
    final_checklist: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ACADEMIC_NATURAL_REVISION_DOCTRINE = RevisionDoctrine(
    public_name="AIRC",
    positioning=(
        "agent-first revision workflow",
        "academic natural revision",
        "developmental academic rewrite",
        "body-only rewrite quota system",
        "human revision simulation layer",
        "chapter-aware rewrite policy",
        "structure-preserving revision",
        "natural academic polish",
    ),
    thesis_draft_observations=(
        "Chinese thesis drafts often repeat meta subjects such as 本研究 / 本文 across adjacent sentences.",
        "English titles, English abstracts, figures, formulas, citations, numbers, paths, and checkpoint references are high-preserve zones.",
        "Background, significance, risk analysis, and ordinary method-explanation paragraphs usually benefit from sentence-cluster restructuring.",
        "Terminology-dense or citation/number-dense paragraphs should be simplified conservatively rather than broadly rewritten.",
        "Conclusion and transition sentences should often be absorbed into the previous argument instead of remaining as 因此，本研究-style followups.",
        "Rewrite coverage is measured only over ordinary body prose; headings, images, captions, formulas, references, paths, and technical-dense blocks are excluded.",
        "Long papers require broad body reconstruction across blocks rather than a small number of locally polished paragraphs.",
        "Human revision is uneven: some blocks are compressed, some expanded or reordered, and a few sentences may remain close to the source for clarity.",
        "Human editors allocate effort by chapter function: background, significance, review, analysis, conclusion, and future work open up more than definitions, losses, metrics, setup, and deployment details.",
    ),
    body_rewrite_methods=(
        "chapter_aware_rewrite_priority",
        "chapter_rewrite_quota_check",
        "compress_subject_chain",
        "compress_meta_discourse",
        "weaken_template_connectors",
        "reduce_function_word_overuse",
        "rebuild_sentence_rhythm",
        "break_parallelism",
        "sentence_cluster_rewrite",
        "sentence_cluster_merge",
        "sentence_cluster_split",
        "discourse_reordering",
        "narrative_path_rewrite",
        "conclusion_absorption",
        "uneven_rewrite_distribution",
        "body_wide_discourse_rebuild",
        "conclusion_absorb",
        "rewrite_dense_nominal_phrases",
    ),
    block_strategies={
        "do_not_touch": (
            "preserve verbatim",
            "perform format checks only",
            "never apply developmental rewrite",
        ),
        "high_risk": (
            "default to original wording",
            "allow only minimal cleanup",
            "do not expand or generalize technical content",
        ),
        "light_edit": (
            "perform sentence-level polish",
            "reduce repeated subjects and connectors when safe",
            "do not reorder paragraph-level logic",
        ),
        "rewritable": (
            "perform developmental academic rewrite",
            "execute sentence-cluster restructuring",
            "show discourse-level change rather than word replacement",
            "rebuild the explanation path across sentence groups",
            "apply non-uniform revision patterns such as compress, expand, reorder, partial_keep, and rewrite_all",
        ),
        "chapter_high_priority": (
            "rewrite background, significance, review, analysis, conclusion, and future-work narration more deeply",
            "meet chapter-level coverage, cluster, and discourse quotas",
            "rebuild explanation path while preserving claims",
        ),
        "chapter_medium_priority": (
            "improve method, mechanism, training, dataset, system architecture, and workflow narration around protected terms",
            "allow sentence-cluster changes without drifting technical detail",
            "meet moderate chapter coverage",
        ),
        "chapter_conservative_priority": (
            "keep problem definitions, losses, metrics, experiment setup, and deployment details light",
            "prefer partial keep and connector cleanup",
            "reject broad reconstruction that risks technical drift",
        ),
    },
    style_profile=ACADEMIC_NATURAL_STUDENTLIKE.name,
    natural_actions=ACADEMIC_NATURAL_STUDENTLIKE.required_actions,
    forbidden_patterns=ACADEMIC_NATURAL_STUDENTLIKE.forbidden_markers,
    final_checklist=ACADEMIC_NATURAL_STUDENTLIKE.checklist,
)


def doctrine_for_agent_context() -> dict[str, Any]:
    return ACADEMIC_NATURAL_REVISION_DOCTRINE.to_dict()


def doctrine_for_style(style_name: str) -> dict[str, Any]:
    """Return the public doctrine with the requested style profile attached."""

    profile = get_natural_revision_profile(style_name)
    payload = ACADEMIC_NATURAL_REVISION_DOCTRINE.to_dict()
    payload["style_profile"] = profile.name
    payload["style_profile_description"] = profile.description
    payload["natural_actions"] = list(profile.required_actions)
    payload["forbidden_patterns"] = list(profile.forbidden_markers)
    payload["final_checklist"] = profile.checklist
    if profile.name == "zh_academic_l2_mild":
        payload["positioning"] = [
            *payload["positioning"],
            "optional mild Chinese L2 academic texture",
            "restrained non-native-like academic phrasing without grammar errors",
        ]
    return payload
