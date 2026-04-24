"""Shared doctrine for natural academic revision.

This module is intentionally descriptive and deterministic. It gives agents a
stable style profile without encouraging colloquial, web-style, or adversarial
rewriting goals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NaturalRevisionProfile:
    name: str
    description: str
    style_principles: tuple[str, ...]
    required_actions: tuple[str, ...]
    optional_actions: tuple[str, ...]
    forbidden_markers: tuple[str, ...]
    checklist: dict[str, tuple[str, ...]]


ACADEMIC_NATURAL_STUDENTLIKE = NaturalRevisionProfile(
    name="academic_natural_studentlike",
    description=(
        "Revise ordinary Chinese academic narration as if a careful undergraduate "
        "or graduate student repeatedly polished their own paper: natural, formal, "
        "restrained, and faithful to the original content."
    ),
    style_principles=(
        "Only change narration, sentence organization, paragraph flow, and naturalness.",
        "Keep facts, claims, terms, citations, numbers, formulas, captions, placeholders, and Markdown structure stable.",
        "Reduce template-heavy wording by deleting, absorbing, or merging it rather than replacing it with another template.",
        "Prefer clear subjects, but compress repeated 本研究 / 本文 chains when context already identifies the actor.",
        "Use longer and shorter sentences in a natural rhythm; do not split or merge just for variety.",
        "Simulate human revision behavior: some sentence groups are compressed, some expanded, some reordered, and some partly kept.",
        "Avoid uniform rewrite distribution; a whole paper should not look as if every sentence was edited by the same mechanical rule.",
        "Allocate rewrite intensity by chapter function: open up narrative chapters and keep technical-dense chapters controlled.",
        "Keep the tone academic and restrained; do not use web-style, self-media, or chatty expressions.",
    ),
    required_actions=(
        "reduce_function_word_overuse",
        "weaken_template_connectors",
        "compress_meta_discourse",
        "compress_subject_chain",
        "rebuild_sentence_rhythm",
        "break_parallelism",
        "sentence_cluster_merge",
        "sentence_cluster_split",
        "discourse_reordering",
        "narrative_path_rewrite",
        "conclusion_absorption",
        "uneven_rewrite_distribution",
        "rewrite_dense_nominal_phrases",
        "preserve_explicit_subject_if_clarity_needed",
        "keep_original_if_technical_density_is_high",
        "apply_chapter_aware_rewrite_policy",
    ),
    optional_actions=(
        "sentence_cluster_rewrite",
        "transition_absorption",
        "conclusion_absorb",
        "enumeration_reframe",
        "rationale_expansion",
        "partial_keep",
        "rewrite_all",
    ),
    forbidden_markers=(
        "这块",
        "这边",
        "大家",
        "我们",
        "里头",
        "儿化",
        "超",
        "超级",
        "真的",
        "其实",
        "大白话",
    ),
    checklist={
        "core_consistency": (
            "original meaning preserved",
            "facts unchanged",
            "core claims unchanged",
        ),
        "academic_register": (
            "objective and restrained tone",
            "no emotional wording",
            "no excessive colloquialization",
        ),
        "naturalness": (
            "mechanical connectors reduced",
            "template sentences reduced",
            "repeated subjects reduced where clarity allows",
            "parallel sentence skeletons broken up when they feel mechanical",
            "sentence rhythm varies naturally",
            "meta discourse compressed where it repeats the next sentence",
            "narrative flow rebuilt in ordinary body narration",
            "human-like variation is present across blocks",
            "rewrite distribution is non-uniform rather than mechanically even",
            "sentence-cluster changes are visible in long documents",
            "chapter-level rewrite intensity matches section function",
            "developmental rewrite applied to rewritable body blocks",
        ),
        "wording": (
            "function-word overuse reduced when safe",
            "empty verbs such as 进行 / 构建 / 实现 / 提升 are not piled up",
            "unnecessary 的 / 了 / 并 / 而 are cleaned when meaning stays stable",
        ),
        "structure": (
            "paragraph count preserved",
            "Markdown structure preserved",
            "headings, formulas, captions, placeholders, paths, terms, citations, and numbers unchanged",
        ),
    },
)

ZH_ACADEMIC_L2_MILD = NaturalRevisionProfile(
    name="zh_academic_l2_mild",
    description=(
        "Revise Chinese academic prose as a serious non-native or less fluent Chinese writer: "
        "formal, careful, slightly explanatory, mildly redundant, and still grammatically complete."
    ),
    style_principles=(
        "Preserve the original facts, terms, citations, numbers, formulas, headings, paths, and Markdown structure.",
        "Keep a restrained academic tone; do not add outside background, commentary, or web-style language.",
        "Do not seek native-level concision or polish; allow mild explanatory wording when it remains grammatical.",
        "Let some function words and process words remain or increase mildly, such as 的, 了, 来, 进行, 会, 能够, 还有, 通过……来.",
        "Allow small local redundancy and explanatory phrasing, but never broken grammar or meaningless repetition.",
        "Use 本研究 / 模型 / 系统 / 分支 as default academic subjects; use 我们 only when the source style supports it.",
        "Keep technical-dense, formula, metric, path, checkpoint, and citation-heavy blocks conservative.",
    ),
    required_actions=(
        "expand_compact_academic_clause",
        "increase_function_word_density_mildly",
        "soften_native_like_concision",
        "allow_explanatory_rephrasing",
        "inject_mild_l2_texture",
        "avoid_too_fluent_native_polish",
        "preserve_original_evidence_scope",
        "keep_original_if_technical_density_is_high",
    ),
    optional_actions=(
        "split_overpacked_sentence_mildly",
        "retain_mild_repetition_when_clear",
        "use_plain_academic_subject",
        "explain_nominal_phrase_as_process",
    ),
    forbidden_markers=(
        "这块",
        "这边",
        "大家",
        "超",
        "超级",
        "真的",
        "其实",
        "大白话",
        "搞定",
        "拉满",
        "靠谱",
    ),
    checklist={
        "core_consistency": (
            "facts unchanged",
            "technical terms preserved",
            "citations, numbers, formulas, paths, captions, placeholders, and Markdown structure unchanged",
        ),
        "mild_l2_texture": (
            "some explanatory wording is present",
            "function-word density is mildly higher than the default academic_natural profile",
            "native-like compression is softened in ordinary narrative prose",
        ),
        "quality_bounds": (
            "no colloquial or web-style wording",
            "no intentionally broken grammar",
            "no dangling sentence fragments",
            "no unsupported expansion",
        ),
    },
)


FUNCTION_WORD_ACTIONS = (
    "reduce_function_word_overuse",
    "rewrite_dense_nominal_phrases",
)

NATURALNESS_ACTIONS = ACADEMIC_NATURAL_STUDENTLIKE.required_actions
COLLOQUIAL_FORBIDDEN_MARKERS = ACADEMIC_NATURAL_STUDENTLIKE.forbidden_markers
NATURAL_REVISION_CHECKLIST = ACADEMIC_NATURAL_STUDENTLIKE.checklist

STYLE_PROFILES = {
    ACADEMIC_NATURAL_STUDENTLIKE.name: ACADEMIC_NATURAL_STUDENTLIKE,
    ZH_ACADEMIC_L2_MILD.name: ZH_ACADEMIC_L2_MILD,
}


def get_natural_revision_profile(name: str) -> NaturalRevisionProfile:
    """Return a configured natural revision profile by stable style name."""

    normalized = (name or ACADEMIC_NATURAL_STUDENTLIKE.name).strip().lower()
    if normalized in {"academic_natural", ACADEMIC_NATURAL_STUDENTLIKE.name}:
        return ACADEMIC_NATURAL_STUDENTLIKE
    try:
        return STYLE_PROFILES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(STYLE_PROFILES))
        raise ValueError(f"Unsupported natural revision profile: {name}. Supported profiles: {supported}") from exc
