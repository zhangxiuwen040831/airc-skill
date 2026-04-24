from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RewriteMode(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    STRONG = "strong"

    @classmethod
    def from_value(cls, value: str) -> "RewriteMode":
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unsupported rewrite mode: {value}")


@dataclass(frozen=True)
class SkillPreset:
    name: str
    mode: RewriteMode
    target_style: str
    rewrite_scope: str
    preservation_level: str
    default_rewrite_intensity: str
    min_rewrite_coverage: float
    force_cluster_rewrite: bool
    max_retry_passes: int
    description: str


@dataclass(frozen=True)
class AppConfig:
    max_file_size_bytes: int = 5 * 1024 * 1024
    min_chunk_chars: int = 800
    max_chunk_chars: int = 1500
    supported_suffixes: tuple[str, ...] = (".md", ".txt")
    candidate_encodings: tuple[str, ...] = ("utf-8-sig", "utf-8", "gb18030", "big5")
    meaningful_rewrite_length: int = 120
    balanced_min_changed_chars: int = 12
    strong_min_changed_chars: int = 20
    max_rewrite_passes: int = 3
    recent_candidate_window: int = 4
    repeated_pattern_limit: int = 2
    repeated_opener_limit: int = 2
    paragraph_template_family_limit: int = 1
    document_template_family_warning_limit: int = 2
    document_template_family_block_limit: int = 3
    rule_like_family_warning_limit: int = 2
    rule_like_family_block_limit: int = 3
    structural_gate_length: int = 120
    structural_gate_length_long: int = 220
    balanced_min_structural_actions: int = 1
    balanced_min_structural_actions_long: int = 2
    strong_min_structural_actions: int = 2
    strong_min_structural_actions_long: int = 3
    light_edit_min_sentence_level_changes: int = 1
    developmental_min_sentence_level_changes: int = 2
    developmental_min_cluster_changes: int = 1
    light_edit_min_discourse_score: int = 2
    developmental_min_discourse_score: int = 5
    developmental_min_cluster_score: int = 1
    rewrite_coverage_pass_threshold: float = 0.60
    rewrite_coverage_minor_threshold: float = 0.40
    document_scale_medium_body_chars: int = 3000
    document_scale_long_body_chars: int = 8000
    document_scale_very_long_body_chars: int = 15000
    body_rewrite_coverage_short_threshold: float = 0.40
    body_rewrite_coverage_medium_threshold: float = 0.50
    body_rewrite_coverage_long_threshold: float = 0.60
    body_rewrite_coverage_very_long_threshold: float = 0.60
    body_changed_block_ratio_short_threshold: float = 0.0
    body_changed_block_ratio_medium_threshold: float = 0.30
    body_changed_block_ratio_long_threshold: float = 0.45
    body_changed_block_ratio_very_long_threshold: float = 0.50
    body_discourse_short_threshold: int = 2
    body_discourse_medium_threshold: int = 3
    body_discourse_long_threshold: int = 5
    body_discourse_very_long_threshold: int = 5
    body_cluster_short_threshold: int = 0
    body_cluster_medium_threshold: int = 1
    body_cluster_long_threshold: int = 2
    body_cluster_very_long_threshold: int = 2
    human_variation_min_patterns: int = 2
    human_variation_min_changed_block_spread: int = 2
    style_variation_warning_score: int = 4
    style_variation_reject_score: int = 2
    strict_mode_default: bool = True
    strict_min_high_value_actions: int = 1
    strict_template_blacklist: tuple[str, ...] = (
        "从……角度看",
        "在此基础上",
        "这也意味着",
        "进一步来看",
        "综上",
    )
    strict_template_surface_markers: tuple[str, ...] = (
        "从",
        "在此基础上",
        "这也意味着",
        "进一步来看",
        "综上",
    )
    strict_blacklist_repeat_limit: int = 1


DEFAULT_CONFIG = AppConfig()


SKILL_PRESETS: dict[str, SkillPreset] = {
    "academic_natural": SkillPreset(
        name="academic_natural",
        mode=RewriteMode.BALANCED,
        target_style="academic_natural",
        rewrite_scope="full",
        preservation_level="strict",
        default_rewrite_intensity="medium",
        min_rewrite_coverage=DEFAULT_CONFIG.rewrite_coverage_pass_threshold,
        force_cluster_rewrite=True,
        max_retry_passes=DEFAULT_CONFIG.max_rewrite_passes,
        description="Default production preset: formal academic narration with strong cluster-level rewriting.",
    ),
    "aggressive_rewrite": SkillPreset(
        name="aggressive_rewrite",
        mode=RewriteMode.STRONG,
        target_style="academic_natural",
        rewrite_scope="full",
        preservation_level="strict",
        default_rewrite_intensity="high",
        min_rewrite_coverage=DEFAULT_CONFIG.rewrite_coverage_pass_threshold,
        force_cluster_rewrite=True,
        max_retry_passes=DEFAULT_CONFIG.max_rewrite_passes,
        description="Use for ordinary Chinese body paragraphs that need broad developmental rewriting.",
    ),
    "conservative": SkillPreset(
        name="conservative",
        mode=RewriteMode.CONSERVATIVE,
        target_style="concise",
        rewrite_scope="body_only",
        preservation_level="strict",
        default_rewrite_intensity="light",
        min_rewrite_coverage=DEFAULT_CONFIG.rewrite_coverage_minor_threshold,
        force_cluster_rewrite=False,
        max_retry_passes=1,
        description="Use when only light edits are allowed and developmental rewrite is not desired.",
    ),
    "body_only": SkillPreset(
        name="body_only",
        mode=RewriteMode.BALANCED,
        target_style="academic_natural",
        rewrite_scope="body_only",
        preservation_level="strict",
        default_rewrite_intensity="medium",
        min_rewrite_coverage=DEFAULT_CONFIG.rewrite_coverage_pass_threshold,
        force_cluster_rewrite=True,
        max_retry_passes=DEFAULT_CONFIG.max_rewrite_passes,
        description="Rewrite ordinary body narration while keeping headings, abstracts, captions, references, and appendices conservative.",
    ),
    "zh_academic_l2_mild": SkillPreset(
        name="zh_academic_l2_mild",
        mode=RewriteMode.BALANCED,
        target_style="zh_academic_l2_mild",
        rewrite_scope="full",
        preservation_level="strict",
        default_rewrite_intensity="medium",
        min_rewrite_coverage=DEFAULT_CONFIG.rewrite_coverage_pass_threshold,
        force_cluster_rewrite=True,
        max_retry_passes=DEFAULT_CONFIG.max_rewrite_passes,
        description=(
            "Optional style preset: restrained Chinese academic prose with mild non-native texture, "
            "slightly explanatory wording, and strict evidence/format preservation."
        ),
    ),
}


def get_skill_preset(name: str) -> SkillPreset:
    normalized = name.strip().lower()
    try:
        return SKILL_PRESETS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(SKILL_PRESETS))
        raise ValueError(f"Unsupported AIRC preset: {name}. Supported presets: {supported}") from exc


def fallback_modes(mode: RewriteMode) -> list[RewriteMode]:
    if mode is RewriteMode.STRONG:
        return [RewriteMode.STRONG, RewriteMode.BALANCED, RewriteMode.CONSERVATIVE]
    if mode is RewriteMode.BALANCED:
        return [RewriteMode.BALANCED, RewriteMode.CONSERVATIVE]
    return [RewriteMode.CONSERVATIVE]
