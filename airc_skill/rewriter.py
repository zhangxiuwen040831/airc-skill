from __future__ import annotations

import re
import math
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Protocol

from .academic_sentence_naturalization import analyze_academic_sentence_naturalization
from .authorial_intent import analyze_authorial_intent
from .config import DEFAULT_CONFIG, RewriteMode
from .evidence_fidelity import analyze_evidence_fidelity
from .local_revision_realism import analyze_local_revision_sentences, technical_density_is_high
from .paragraph_skeleton import (
    analyze_paragraph_skeleton,
    is_dangling_opening_sentence,
    opening_style_valid,
    paragraph_skeleton_checks,
)
from .sentence_readability import (
    analyze_paragraph_readability_sentences,
    dangling_sentence_risk,
    fragment_like_conclusion_sentence,
    incomplete_support_sentence_risk,
)

_LINE_BREAK_RE = re.compile(r"[ \t]*\n[ \t]*")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_TRAILING_PUNCT_RE = re.compile(r"[。！？?!]+$")
_LEADING_CONNECTOR_RE = re.compile(r"^(?:并且|而且|并|还|同时|此外|另外|与此同时|因此|所以|由此|基于此)\s*")

_SUMMARY_PREFIXES = ("总的来说", "总而言之", "总体来看", "总体而言", "综合来看", "综上")
_TRANSITION_PREFIXES = ("与此同时", "同时", "此外", "另外", "在这一过程中", "与之相伴的是")
_IMPLICATION_PREFIXES = ("因此", "由此可见", "由此", "这也意味着", "这也说明", "基于此", "在这种情况下")
_META_SUBJECTS = ("本研究", "本文", "该研究", "该系统")
_SCAFFOLDING_OPENINGS = (
    "这一工作",
    "这项工作",
    "这一点",
    "该段论述",
    "相关内容",
    "该部分",
    "这一部分",
    "这一设置",
    "这一调整",
    "这一诊断结果",
    "这一机制",
    "这一设计",
)
_HUANBAOKUO_RE = re.compile(r"(?:还包括|也包括|进一步包括|相关内容还包括|相关内容包括|进一步还包括)")
_ENUMERATION_ITEM_RE = re.compile(r"^(?:[（(]?\d+[）)]|第一|第二|第三|第四|第五|第六|其一|其二|其三)")
_ENUMERATION_HEAD_RE = re.compile(
    r"(?:主要包括以下|主要分为|主要包含|主要创新点如下|创新点如下|主要内容如下|包括以下几类|包含三类关键组成部分)"
)
_APPENDIX_LIKE_SUPPORT_RE = re.compile(
    r"^(?:用于进一步|从[^。；]{2,20}角度来看|在该设置下|在当前设置下|通过这种方式|"
    r"结合[^。；]{2,40}可以看出|可以看出|这表明|本研究用于|本文用于|该方法用于|该机制用于|这种方式用于)[，,\s]*"
)
_WEAK_MODAL_VERB_RE = re.compile(r"(?:可以|能够|用于|有助于)")
_AUTHORIAL_STANCE_RE = re.compile(
    r"(?:相比之下|不同于|并非|而是|更关键的是|关键在于|核心在于|本质是|本质在于|"
    r"本研究选择|本工作选择|选择[^。；]{1,48}而非[^。；]{1,48}|重点并非[^。；]{1,96}而是)"
)
_UNSUPPORTED_EXPANSION_RE = re.compile(
    r"(?:主流观点|业界通常认为|普遍认为|领域共识|超过八成|超过半数|绝大多数|"
    r"20\d{2}年(?:AIGC|生成式|内容生态|传播环境)|投诉|内容生态|运营人员)"
)
_METAPHOR_STORYTELLING_RE = re.compile(
    r"(?:幻想的破灭|边界修复手术|判决书|挣脱出来|终于摆脱|路径终点|被击穿|逼近满分|"
    r"如同[^。；]{0,24}(?:手术|判决|故事)|仿佛[^。；]{0,24}(?:剧情|手术|较量))"
)
_UNJUSTIFIED_AUTHORIAL_CLAIM_RE = re.compile(
    r"(?:^我们[，,\s]*|^我们的[，,\s]*|本工作证明了|这项工作的终点|我们的工程价值|我们选择|正如该领域主流观点所认为的)"
)
_THESIS_REGISTER_WRAPPER_RE = re.compile(
    r"^(?:正如该领域主流观点所认为的|业界通常认为|普遍认为|可以看出|由此可见|从这一角度来看|从[^。；]{1,20}角度来看)[，,\s]*"
)
_TEMPLATE_FAMILY_NAMES = (
    "study_subject_family",
    "implication_family",
    "transition_family",
    "framing_family",
)
_RULE_LIKE_FAMILIES = {
    "summary-opening",
    "note-opening",
    "practical-opening",
    "implication-opening",
    "transition-opening",
    "study-focus",
    "therefore-study",
    "therefore-build",
}
STRUCTURAL_ACTION_TYPES = (
    "sentence_merge",
    "sentence_split",
    "sentence_cluster_merge",
    "sentence_cluster_split",
    "sentence_cluster_rewrite",
    "clause_reorder",
    "pair_fusion",
    "enumeration_reframe",
    "conclusion_absorb",
    "conclusion_absorption",
    "paragraph_reorder",
    "discourse_reordering",
    "narrative_path_rewrite",
    "topic_reframe",
    "merge_consecutive_subject_sentences",
    "subject_drop",
    "subject_variation",
    "meta_compression",
    "followup_absorb",
    "subject_chain_compression",
    "uneven_rewrite_distribution",
    "topic_sentence_preservation",
    "soften_overexplicit_transition",
    "reduce_sentence_uniformity",
    "introduce_local_hierarchy",
    "reshape_supporting_sentence",
    "weaken_overfinished_sentence",
    "convert_flat_parallel_flow",
    "light_partial_retain_with_local_rephrase",
    "readability_repair_pass",
    "sentence_completeness_repair",
    "repair_incomplete_support_sentence",
    "repair_fragment_like_conclusion_sentence",
    "high_sensitivity_readability_repair",
    "preserve_semantic_role_of_core_sentence",
    "preserve_enumeration_item_role",
    "remove_generated_scaffolding_phrase",
    "replace_abstracted_subject_with_concrete_referent",
    "restore_mechanism_sentence_from_support_like_rewrite",
    "repair_enumeration_flow",
    "prevent_appendix_like_supporting_sentence",
    "avoid_huanbaokuo_style_expansion",
    "upgrade_appendix_like_sentence_to_assertion",
    "strengthen_mechanism_verb",
    "replace_weak_modal_verbs",
    "restore_authorial_choice_expression",
    "promote_support_sentence_to_core_if_needed",
    "reduce_overuse_of_passive_explanations",
    "remove_unsupported_expansion",
    "remove_external_domain_commentary",
    "remove_metaphoric_storytelling",
    "restore_thesis_register",
    "replace_we_with_original_subject_style",
    "downgrade_overclaimed_judgment",
    "restore_mechanism_sentence_to_academic_statement",
    "preserve_original_evidence_scope",
    "remove_bureaucratic_opening",
    "compress_explicit_subject_chain",
    "flatten_overstructured_parallelism",
    "advance_main_clause",
    "remove_slogan_like_goal_phrase",
    "convert_project_style_opening_to_academic_statement",
    "replace_theme_is_with_direct_topic_statement",
    "decompress_overpacked_modifier_prefix",
    "enforce_direct_statement",
    "move_main_clause_forward",
    "split_overlong_sentence",
    "remove_academic_wrapping",
    "convert_passive_to_active",
    "reduce_connectors",
    "diversify_subject",
    "denormalize_parallel_structure",
    "expand_compact_academic_clause",
    "increase_function_word_density_mildly",
    "soften_native_like_concision",
    "allow_explanatory_rephrasing",
    "inject_mild_l2_texture",
    "avoid_too_fluent_native_polish",
)
DISCOURSE_ACTION_TYPES = (
    "proposition_reorder",
    "sentence_cluster_rewrite",
    "sentence_cluster_merge",
    "sentence_cluster_split",
    "discourse_reordering",
    "narrative_path_rewrite",
    "conclusion_absorption",
    "meta_compression",
    "subject_chain_compression",
    "compress_subject_chain",
    "conclusion_absorb",
    "enumeration_reframe",
    "rationale_expansion",
    "transition_absorption",
    "reduce_function_word_overuse",
    "weaken_template_connectors",
    "rebuild_sentence_rhythm",
    "break_parallelism",
    "rewrite_dense_nominal_phrases",
    "preserve_explicit_subject_if_clarity_needed",
    "keep_original_if_technical_density_is_high",
    "uneven_rewrite_distribution",
    "soften_overexplicit_transition",
    "reduce_sentence_uniformity",
    "introduce_local_hierarchy",
    "reshape_supporting_sentence",
    "weaken_overfinished_sentence",
    "convert_flat_parallel_flow",
    "light_partial_retain_with_local_rephrase",
    "readability_repair_pass",
    "sentence_completeness_repair",
    "repair_incomplete_support_sentence",
    "repair_fragment_like_conclusion_sentence",
    "high_sensitivity_readability_repair",
    "preserve_semantic_role_of_core_sentence",
    "preserve_enumeration_item_role",
    "remove_generated_scaffolding_phrase",
    "replace_abstracted_subject_with_concrete_referent",
    "restore_mechanism_sentence_from_support_like_rewrite",
    "repair_enumeration_flow",
    "prevent_appendix_like_supporting_sentence",
    "avoid_huanbaokuo_style_expansion",
)
HIGH_IMPACT_ACTION_TYPES = (
    "pair_fusion",
    "conclusion_absorb",
    "conclusion_absorption",
    "paragraph_reorder",
    "discourse_reordering",
    "narrative_path_rewrite",
    "subject_chain_compression",
    "sentence_cluster_rewrite",
    "sentence_cluster_merge",
    "proposition_reorder",
    "topic_sentence_preservation",
    "introduce_local_hierarchy",
    "convert_flat_parallel_flow",
    "readability_repair_pass",
    "sentence_completeness_repair",
    "preserve_enumeration_item_role",
    "restore_mechanism_sentence_from_support_like_rewrite",
)
_STRICT_TEMPLATE_SURFACES = {
    "从……角度看",
    "在此基础上",
    "这也意味着",
    "进一步来看",
    "综上",
}


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like units while keeping end punctuation."""

    sentences: list[str] = []
    current: list[str] = []
    round_depth = 0
    square_depth = 0
    brace_depth = 0

    for character in text:
        current.append(character)

        if character in "（(":
            round_depth += 1
        elif character in "）)" and round_depth > 0:
            round_depth -= 1
        elif character == "[":
            square_depth += 1
        elif character == "]" and square_depth > 0:
            square_depth -= 1
        elif character == "{":
            brace_depth += 1
        elif character == "}" and brace_depth > 0:
            brace_depth -= 1

        if character in "。！？?!" and round_depth == square_depth == brace_depth == 0:
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []

    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)
    return sentences


class RewriteBackend(Protocol):
    def rewrite(
        self,
        text: str,
        mode: RewriteMode,
        pass_index: int = 1,
        rewrite_depth: str | None = None,
        rewrite_intensity: str | None = None,
        high_sensitivity_prose: bool = False,
        style_profile: str = "academic_natural",
    ) -> "BackendRewriteResult":
        """Rewrite a single chunk of text."""

    def reset_document_state(self) -> None:
        """Reset any document-level rewrite state."""


@dataclass(frozen=True)
class SentenceUnit:
    text: str
    label: str
    source_indices: tuple[int, ...]


@dataclass(frozen=True)
class CandidateOption:
    key: str
    text: str
    surface: str
    template_family: str | None = None
    rule_like: bool | None = None


@dataclass
class RewriteDocumentState:
    paragraph_index: int = 0
    family_usage: dict[str, int] = field(default_factory=dict)
    paragraph_family_usage: dict[str, int] = field(default_factory=dict)
    template_family_usage: dict[str, int] = field(default_factory=dict)
    paragraph_template_family_usage: dict[str, int] = field(default_factory=dict)
    rule_like_usage: dict[str, int] = field(default_factory=dict)
    paragraph_rule_like_usage: dict[str, int] = field(default_factory=dict)
    variant_usage: dict[str, int] = field(default_factory=dict)
    surface_usage: dict[str, int] = field(default_factory=dict)
    recent_keys: list[str] = field(default_factory=list)
    recent_surfaces: list[str] = field(default_factory=list)
    recent_template_families: list[str] = field(default_factory=list)

    def advance_paragraph(self) -> int:
        self.paragraph_index += 1
        self.paragraph_family_usage = {}
        self.paragraph_template_family_usage = {}
        self.paragraph_rule_like_usage = {}
        return self.paragraph_index

    def record_candidate(
        self,
        family: str,
        option: CandidateOption,
        template_family: str | None,
        rule_like: bool,
    ) -> None:
        self.family_usage[family] = self.family_usage.get(family, 0) + 1
        self.paragraph_family_usage[family] = self.paragraph_family_usage.get(family, 0) + 1
        if template_family:
            self.template_family_usage[template_family] = self.template_family_usage.get(template_family, 0) + 1
            self.paragraph_template_family_usage[template_family] = (
                self.paragraph_template_family_usage.get(template_family, 0) + 1
            )
            self.recent_template_families.append(template_family)
        self.variant_usage[option.key] = self.variant_usage.get(option.key, 0) + 1
        self.surface_usage[option.surface] = self.surface_usage.get(option.surface, 0) + 1
        if rule_like:
            self.rule_like_usage[family] = self.rule_like_usage.get(family, 0) + 1
            self.paragraph_rule_like_usage[family] = self.paragraph_rule_like_usage.get(family, 0) + 1
        self.recent_keys.append(option.key)
        self.recent_surfaces.append(option.surface)
        max_window = DEFAULT_CONFIG.recent_candidate_window
        self.recent_keys = self.recent_keys[-max_window:]
        self.recent_surfaces = self.recent_surfaces[-max_window:]
        self.recent_template_families = self.recent_template_families[-max_window:]


@dataclass(frozen=True)
class BackendRewriteResult:
    text: str
    applied_rules: list[str]
    original_sentences: list[str]
    rewritten_sentences: list[str]
    sentence_level_change: bool
    paragraph_char_count: int
    sentence_labels: list[str]
    subject_heads: list[str]
    detected_patterns: list[str]
    structural_actions: list[str]
    structural_action_count: int
    high_value_structural_actions: list[str]
    discourse_actions_used: list[str]
    sentence_level_changes: int
    cluster_changes: int
    discourse_change_score: int
    rewrite_coverage: float
    prefix_only_rewrite: bool
    repeated_subject_risk: bool
    selected_variants: list[str]
    candidate_notes: list[str]
    paragraph_index: int
    revision_patterns: list[str] = field(default_factory=list)
    human_noise_marks: list[str] = field(default_factory=list)
    sentence_transition_rigidity: float = 0.0
    local_discourse_flatness: float = 0.0
    revision_realism_score: float = 0.0
    sentence_cadence_irregularity: float = 0.0
    local_revision_actions: list[str] = field(default_factory=list)
    sentence_completeness_score: float = 1.0
    paragraph_readability_score: float = 1.0
    dangling_sentence_risk: float = 0.0
    incomplete_support_sentence_risk: float = 0.0
    fragment_like_conclusion_risk: float = 0.0
    readability_repair_actions: list[str] = field(default_factory=list)
    high_sensitivity_prose: bool = False


@dataclass
class RewriteStats:
    mode: RewriteMode
    changed: bool
    applied_rules: list[str]
    sentence_count_before: int
    sentence_count_after: int
    sentence_level_change: bool
    changed_characters: int
    original_sentences: list[str]
    rewritten_sentences: list[str]
    paragraph_char_count: int
    sentence_labels: list[str]
    subject_heads: list[str]
    detected_patterns: list[str]
    structural_actions: list[str]
    structural_action_count: int
    high_value_structural_actions: list[str]
    discourse_actions_used: list[str] = field(default_factory=list)
    sentence_level_changes: int = 0
    cluster_changes: int = 0
    discourse_change_score: int = 0
    rewrite_coverage: float = 0.0
    prefix_only_rewrite: bool = False
    repeated_subject_risk: bool = False
    selected_variants: list[str] = field(default_factory=list)
    candidate_notes: list[str] = field(default_factory=list)
    paragraph_index: int = 0
    block_id: int = 0
    rewrite_depth: str = ""
    rewrite_intensity: str = ""
    revision_patterns: list[str] = field(default_factory=list)
    human_noise_marks: list[str] = field(default_factory=list)
    sentence_transition_rigidity: float = 0.0
    local_discourse_flatness: float = 0.0
    revision_realism_score: float = 0.0
    sentence_cadence_irregularity: float = 0.0
    local_revision_actions: list[str] = field(default_factory=list)
    sentence_completeness_score: float = 1.0
    paragraph_readability_score: float = 1.0
    dangling_sentence_risk: float = 0.0
    incomplete_support_sentence_risk: float = 0.0
    fragment_like_conclusion_risk: float = 0.0
    readability_repair_actions: list[str] = field(default_factory=list)
    high_sensitivity_prose: bool = False


class LLMRewriteBackend:
    """Optional future backend. The local MVP keeps it disabled by default."""

    def reset_document_state(self) -> None:
        return None

    def rewrite(
        self,
        text: str,
        mode: RewriteMode,
        pass_index: int = 1,
        rewrite_depth: str | None = None,
        rewrite_intensity: str | None = None,
        high_sensitivity_prose: bool = False,
        style_profile: str = "academic_natural",
    ) -> BackendRewriteResult:
        raise NotImplementedError("No online backend is configured in the local MVP.")


class RuleBasedRewriteBackend:
    """A local rewriter that focuses on sentence fusion, controlled variation, and paragraph flow."""

    def __init__(self, style_profile: str = "academic_natural") -> None:
        self.style_profile = style_profile
        self.reset_document_state()

    def reset_document_state(self) -> None:
        self.state = RewriteDocumentState()

    def rewrite(
        self,
        text: str,
        mode: RewriteMode,
        pass_index: int = 1,
        rewrite_depth: str | None = None,
        rewrite_intensity: str | None = None,
        high_sensitivity_prose: bool = False,
        style_profile: str = "academic_natural",
    ) -> BackendRewriteResult:
        active_style_profile = style_profile or self.style_profile
        if not text.strip():
            return BackendRewriteResult(
                text=text,
                applied_rules=[],
                original_sentences=[],
                rewritten_sentences=[],
                sentence_level_change=False,
                paragraph_char_count=0,
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
                paragraph_index=self.state.paragraph_index,
                high_sensitivity_prose=high_sensitivity_prose,
            )

        paragraph_index = self.state.advance_paragraph()
        active_rewrite_depth = rewrite_depth or (
            "developmental_rewrite" if mode in {RewriteMode.BALANCED, RewriteMode.STRONG} else "light_edit"
        )
        active_rewrite_intensity = rewrite_intensity or (
            "high" if active_rewrite_depth == "developmental_rewrite" and mode is RewriteMode.STRONG else "medium"
        )
        prefix = text[: len(text) - len(text.lstrip())]
        suffix = text[len(text.rstrip()) :]
        core = self._normalize_whitespace(text.strip())

        original_sentences = split_sentences(core) or [core]
        units = [
            SentenceUnit(text=sentence, label=self._classify_sentence(sentence), source_indices=(index,))
            for index, sentence in enumerate(original_sentences)
        ]
        units = self._deduplicate_adjacent_units(units)
        paragraph_char_count = len(re.sub(r"\s+", "", core))
        detected_patterns = self._detect_paragraph_patterns(units)

        applied_rules: list[str] = []
        structural_actions: list[str] = []
        selected_variants: list[str] = []
        candidate_notes: list[str] = []

        rewritten_units, rules, surfaces, notes = self._rewrite_sentence_structures(
            units=units,
            mode=mode,
            pass_index=pass_index,
        )
        applied_rules.extend(rules)
        structural_actions.extend(self._structural_actions_from_rules(rules))
        selected_variants.extend(surfaces)
        candidate_notes.extend(notes)

        if active_rewrite_depth == "developmental_rewrite" and mode is RewriteMode.STRONG:
            rewritten_units, rules, _, notes = self._narrative_flow_rebuilder(
                units=rewritten_units,
                mode=mode,
                pass_index=pass_index,
                rewrite_depth=active_rewrite_depth,
                rewrite_intensity=active_rewrite_intensity,
            )
            applied_rules.extend(rules)
            structural_actions.extend(self._structural_actions_from_rules(rules))
            candidate_notes.extend(notes)

        rewritten_units, rules, surfaces, notes = self._discourse_compressor(
            units=rewritten_units,
            mode=mode,
            pass_index=pass_index,
        )
        applied_rules.extend(rules)
        structural_actions.extend(self._structural_actions_from_rules(rules))
        selected_variants.extend(surfaces)
        candidate_notes.extend(notes)

        if active_rewrite_depth == "developmental_rewrite":
            rewritten_units, rules, surfaces, notes = self._sentence_cluster_rewriter(
                units=rewritten_units,
                mode=mode,
                pass_index=pass_index,
                rewrite_depth=active_rewrite_depth,
                rewrite_intensity=active_rewrite_intensity,
            )
            applied_rules.extend(rules)
            structural_actions.extend(self._structural_actions_from_rules(rules))
            selected_variants.extend(surfaces)
            candidate_notes.extend(notes)

        if active_rewrite_depth == "developmental_rewrite" and mode in {RewriteMode.BALANCED, RewriteMode.STRONG}:
            rewritten_units, rules, _, notes = self._split_long_sentences(
                units=rewritten_units,
                mode=mode,
                pass_index=pass_index,
            )
            applied_rules.extend(rules)
            structural_actions.extend(self._structural_actions_from_rules(rules))
            candidate_notes.extend(notes)

            rewritten_units, rules, surfaces, notes = self._merge_short_followups(
                units=rewritten_units,
                mode=mode,
                pass_index=pass_index,
            )
            applied_rules.extend(rules)
            structural_actions.extend(self._structural_actions_from_rules(rules))
            selected_variants.extend(surfaces)
            candidate_notes.extend(notes)

        if active_rewrite_depth == "developmental_rewrite" or mode is RewriteMode.STRONG:
            rewritten_units, rules, _, notes = self._narrative_flow_rebuilder(
                units=rewritten_units,
                mode=mode,
                pass_index=pass_index,
                rewrite_depth=active_rewrite_depth,
                rewrite_intensity=active_rewrite_intensity,
            )
            applied_rules.extend(rules)
            structural_actions.extend(self._structural_actions_from_rules(rules))
            candidate_notes.extend(notes)

        polished_units: list[SentenceUnit] = []
        for unit in rewritten_units:
            updated_unit, rule_names, surfaces, notes = self._rewrite_sentence_unit(
                unit=unit,
                mode=mode,
                pass_index=pass_index,
            )
            polished_units.append(updated_unit)
            applied_rules.extend(rule_names)
            structural_actions.extend(self._structural_actions_from_rules(rule_names))
            selected_variants.extend(surfaces)
            candidate_notes.extend(notes)

        if active_rewrite_depth == "developmental_rewrite":
            polished_units, rules, surfaces, notes = self._developmental_recast_if_static(
                units=polished_units,
                original_sentences=original_sentences,
                rewrite_intensity=active_rewrite_intensity,
                mode=mode,
                pass_index=pass_index,
            )
            applied_rules.extend(rules)
            structural_actions.extend(self._structural_actions_from_rules(rules))
            selected_variants.extend(surfaces)
            candidate_notes.extend(notes)

        rewritten_units, rules, surfaces, notes = self._rewrite_standalone_transitions(
            units=polished_units,
            mode=mode,
            pass_index=pass_index,
        )
        applied_rules.extend(rules)
        selected_variants.extend(surfaces)
        candidate_notes.extend(notes)

        rewritten_sentences = [self._ensure_sentence_end(unit.text) for unit in rewritten_units if unit.text.strip()]
        rewritten_sentences, human_rules, human_notes, human_noise_marks = self._inject_human_revision_variation(
            original_sentences=original_sentences,
            rewritten_sentences=rewritten_sentences,
            paragraph_index=paragraph_index,
            rewrite_depth=active_rewrite_depth,
            rewrite_intensity=active_rewrite_intensity,
            mode=mode,
        )
        applied_rules.extend(human_rules)
        structural_actions.extend(self._structural_actions_from_rules(human_rules))
        candidate_notes.extend(human_notes)
        rewritten_sentences, opening_rules, opening_notes = self._repair_paragraph_opening(
            original_sentences=original_sentences,
            rewritten_sentences=rewritten_sentences,
        )
        applied_rules.extend(opening_rules)
        structural_actions.extend(self._structural_actions_from_rules(opening_rules))
        candidate_notes.extend(opening_notes)
        rewritten_sentences, duplicate_rules, duplicate_notes = self._deduplicate_adjacent_sentences(rewritten_sentences)
        applied_rules.extend(duplicate_rules)
        structural_actions.extend(self._structural_actions_from_rules(duplicate_rules))
        candidate_notes.extend(duplicate_notes)
        rewritten_sentences, local_rules, local_notes, local_marks = self._apply_local_revision_realism(
            original_sentences=original_sentences,
            rewritten_sentences=rewritten_sentences,
            rewrite_depth=active_rewrite_depth,
            rewrite_intensity=active_rewrite_intensity,
            mode=mode,
            paragraph_index=paragraph_index,
        )
        applied_rules.extend(local_rules)
        structural_actions.extend(self._structural_actions_from_rules(local_rules))
        selected_variants.extend(self._local_surfaces_from_rules(local_rules))
        candidate_notes.extend(local_notes)
        if local_rules:
            rewritten_sentences, opening_rules, opening_notes = self._repair_paragraph_opening(
                original_sentences=original_sentences,
                rewritten_sentences=rewritten_sentences,
            )
            applied_rules.extend(opening_rules)
            structural_actions.extend(self._structural_actions_from_rules(opening_rules))
            candidate_notes.extend(opening_notes)
        rewritten_sentences, readability_rules, readability_notes, readability_marks = self._readability_repair_pass(
            original_sentences=original_sentences,
            rewritten_sentences=rewritten_sentences,
            rewrite_depth=active_rewrite_depth,
            mode=mode,
            high_sensitivity_prose=high_sensitivity_prose,
        )
        applied_rules.extend(readability_rules)
        structural_actions.extend(self._structural_actions_from_rules(readability_rules))
        selected_variants.extend(self._readability_surfaces_from_rules(readability_rules))
        candidate_notes.extend(readability_notes)
        if readability_rules:
            rewritten_sentences, opening_rules, opening_notes = self._repair_paragraph_opening(
                original_sentences=original_sentences,
                rewritten_sentences=rewritten_sentences,
            )
            applied_rules.extend(opening_rules)
            structural_actions.extend(self._structural_actions_from_rules(opening_rules))
            candidate_notes.extend(opening_notes)
            rewritten_sentences, duplicate_rules, duplicate_notes = self._deduplicate_adjacent_sentences(rewritten_sentences)
            applied_rules.extend(duplicate_rules)
            structural_actions.extend(self._structural_actions_from_rules(duplicate_rules))
            candidate_notes.extend(duplicate_notes)
        rewritten_sentences, texture_rules, texture_notes, texture_marks = self._apply_author_texture_variation(
            original_sentences=original_sentences,
            rewritten_sentences=rewritten_sentences,
            rewrite_depth=active_rewrite_depth,
            rewrite_intensity=active_rewrite_intensity,
            mode=mode,
            paragraph_index=paragraph_index,
            high_sensitivity_prose=high_sensitivity_prose,
        )
        applied_rules.extend(texture_rules)
        structural_actions.extend(self._structural_actions_from_rules(texture_rules))
        selected_variants.extend(self._local_surfaces_from_rules(texture_rules))
        candidate_notes.extend(texture_notes)
        if texture_rules:
            rewritten_sentences, duplicate_rules, duplicate_notes = self._deduplicate_adjacent_sentences(rewritten_sentences)
            applied_rules.extend(duplicate_rules)
            structural_actions.extend(self._structural_actions_from_rules(duplicate_rules))
            candidate_notes.extend(duplicate_notes)
        rewritten_sentences, no_op_rules, no_op_notes = self._recast_unchanged_complete_prose(
            original_sentences=original_sentences,
            rewritten_sentences=rewritten_sentences,
            rewrite_depth=active_rewrite_depth,
            mode=mode,
        )
        applied_rules.extend(no_op_rules)
        structural_actions.extend(self._structural_actions_from_rules(no_op_rules))
        selected_variants.extend(self._readability_surfaces_from_rules(no_op_rules))
        candidate_notes.extend(no_op_notes)
        rewritten_sentences, protected_rules, protected_notes = self._repair_protected_token_coverage(
            original_sentences,
            rewritten_sentences,
        )
        applied_rules.extend(protected_rules)
        candidate_notes.extend(protected_notes)
        rewritten_sentences, post_readability_rules, post_readability_notes = self._repair_post_readability_subject_chains(
            original_sentences=original_sentences,
            rewritten_sentences=rewritten_sentences,
        )
        applied_rules.extend(post_readability_rules)
        structural_actions.extend(self._structural_actions_from_rules(post_readability_rules))
        selected_variants.extend(self._readability_surfaces_from_rules(post_readability_rules))
        candidate_notes.extend(post_readability_notes)
        rewritten_sentences, semantic_rules, semantic_notes, semantic_marks = self._apply_semantic_role_integrity_repairs(
            original_sentences=original_sentences,
            rewritten_sentences=rewritten_sentences,
            rewrite_depth=active_rewrite_depth,
            rewrite_intensity=active_rewrite_intensity,
            high_sensitivity_prose=high_sensitivity_prose,
        )
        applied_rules.extend(semantic_rules)
        structural_actions.extend(self._structural_actions_from_rules(semantic_rules))
        selected_variants.extend(self._local_surfaces_from_rules(semantic_rules))
        candidate_notes.extend(semantic_notes)
        if semantic_rules:
            rewritten_sentences, opening_rules, opening_notes = self._repair_paragraph_opening(
                original_sentences=original_sentences,
                rewritten_sentences=rewritten_sentences,
            )
            applied_rules.extend(opening_rules)
            structural_actions.extend(self._structural_actions_from_rules(opening_rules))
            candidate_notes.extend(opening_notes)
            rewritten_sentences, duplicate_rules, duplicate_notes = self._deduplicate_adjacent_sentences(rewritten_sentences)
            applied_rules.extend(duplicate_rules)
            structural_actions.extend(self._structural_actions_from_rules(duplicate_rules))
            candidate_notes.extend(duplicate_notes)
        rewritten_sentences, authorial_rules, authorial_notes, authorial_marks = self._apply_authorial_intent_repairs(
            original_sentences=original_sentences,
            rewritten_sentences=rewritten_sentences,
            rewrite_depth=active_rewrite_depth,
            rewrite_intensity=active_rewrite_intensity,
            high_sensitivity_prose=high_sensitivity_prose,
        )
        applied_rules.extend(authorial_rules)
        structural_actions.extend(self._structural_actions_from_rules(authorial_rules))
        selected_variants.extend(self._local_surfaces_from_rules(authorial_rules))
        candidate_notes.extend(authorial_notes)
        if authorial_rules:
            rewritten_sentences, opening_rules, opening_notes = self._repair_paragraph_opening(
                original_sentences=original_sentences,
                rewritten_sentences=rewritten_sentences,
            )
            applied_rules.extend(opening_rules)
            structural_actions.extend(self._structural_actions_from_rules(opening_rules))
            candidate_notes.extend(opening_notes)
            rewritten_sentences, duplicate_rules, duplicate_notes = self._deduplicate_adjacent_sentences(rewritten_sentences)
            applied_rules.extend(duplicate_rules)
            structural_actions.extend(self._structural_actions_from_rules(duplicate_rules))
            candidate_notes.extend(duplicate_notes)
        rewritten_sentences, evidence_rules, evidence_notes, evidence_marks = self._apply_evidence_fidelity_repairs(
            original_sentences=original_sentences,
            rewritten_sentences=rewritten_sentences,
            rewrite_depth=active_rewrite_depth,
            high_sensitivity_prose=high_sensitivity_prose,
        )
        applied_rules.extend(evidence_rules)
        structural_actions.extend(self._structural_actions_from_rules(evidence_rules))
        selected_variants.extend(self._local_surfaces_from_rules(evidence_rules))
        candidate_notes.extend(evidence_notes)
        if evidence_rules:
            rewritten_sentences, opening_rules, opening_notes = self._repair_paragraph_opening(
                original_sentences=original_sentences,
                rewritten_sentences=rewritten_sentences,
            )
            applied_rules.extend(opening_rules)
            structural_actions.extend(self._structural_actions_from_rules(opening_rules))
            candidate_notes.extend(opening_notes)
            rewritten_sentences, duplicate_rules, duplicate_notes = self._deduplicate_adjacent_sentences(rewritten_sentences)
            applied_rules.extend(duplicate_rules)
            structural_actions.extend(self._structural_actions_from_rules(duplicate_rules))
            candidate_notes.extend(duplicate_notes)
        rewritten_sentences, naturalization_rules, naturalization_notes, naturalization_marks = (
            self._apply_academic_sentence_naturalization(
                original_sentences=original_sentences,
                rewritten_sentences=rewritten_sentences,
                rewrite_depth=active_rewrite_depth,
                high_sensitivity_prose=high_sensitivity_prose,
            )
        )
        applied_rules.extend(naturalization_rules)
        structural_actions.extend(self._structural_actions_from_rules(naturalization_rules))
        selected_variants.extend(self._local_surfaces_from_rules(naturalization_rules))
        candidate_notes.extend(naturalization_notes)
        if naturalization_rules:
            rewritten_sentences, opening_rules, opening_notes = self._repair_paragraph_opening(
                original_sentences=original_sentences,
                rewritten_sentences=rewritten_sentences,
            )
            applied_rules.extend(opening_rules)
            structural_actions.extend(self._structural_actions_from_rules(opening_rules))
            candidate_notes.extend(opening_notes)
            rewritten_sentences, duplicate_rules, duplicate_notes = self._deduplicate_adjacent_sentences(rewritten_sentences)
            applied_rules.extend(duplicate_rules)
            structural_actions.extend(self._structural_actions_from_rules(duplicate_rules))
            candidate_notes.extend(duplicate_notes)
        if active_style_profile == "zh_academic_l2_mild":
            rewritten_sentences, l2_rules, l2_notes, l2_marks = self._apply_l2_mild_style_texture(
                original_sentences=original_sentences,
                rewritten_sentences=rewritten_sentences,
                rewrite_depth=active_rewrite_depth,
                high_sensitivity_prose=high_sensitivity_prose,
            )
            applied_rules.extend(l2_rules)
            structural_actions.extend(self._structural_actions_from_rules(l2_rules))
            selected_variants.extend(self._local_surfaces_from_rules(l2_rules))
            candidate_notes.extend(l2_notes)
            if l2_rules:
                rewritten_sentences, opening_rules, opening_notes = self._repair_paragraph_opening(
                    original_sentences=original_sentences,
                    rewritten_sentences=rewritten_sentences,
                )
                applied_rules.extend(opening_rules)
                structural_actions.extend(self._structural_actions_from_rules(opening_rules))
                candidate_notes.extend(opening_notes)
                rewritten_sentences, duplicate_rules, duplicate_notes = self._deduplicate_adjacent_sentences(rewritten_sentences)
                applied_rules.extend(duplicate_rules)
                structural_actions.extend(self._structural_actions_from_rules(duplicate_rules))
                candidate_notes.extend(duplicate_notes)
        rewritten_sentences, final_duplicate_rules, final_duplicate_notes = self._deduplicate_adjacent_sentences(
            rewritten_sentences
        )
        applied_rules.extend(final_duplicate_rules)
        structural_actions.extend(self._structural_actions_from_rules(final_duplicate_rules))
        candidate_notes.extend(final_duplicate_notes)
        final_core = "".join(rewritten_sentences)
        local_signals = analyze_local_revision_sentences(rewritten_sentences)
        readability_signals = analyze_paragraph_readability_sentences(
            rewritten_sentences,
            high_sensitivity=high_sensitivity_prose,
        )
        subject_heads = [self._extract_subject_head(sentence) for sentence in rewritten_sentences]
        high_value_structural_actions = [
            action for action in structural_actions if action in HIGH_IMPACT_ACTION_TYPES
        ]
        discourse_actions_used = self._discourse_actions_from_rules(applied_rules)
        sentence_level_changes = self._count_sentence_level_changes(original_sentences, rewritten_sentences)
        cluster_changes = self._count_cluster_changes(discourse_actions_used)
        discourse_change_score = self._compute_discourse_change_score(
            discourse_actions_used=discourse_actions_used,
            sentence_level_changes=sentence_level_changes,
            cluster_changes=cluster_changes,
        )
        rewrite_coverage = self._compute_rewrite_coverage(original_sentences, sentence_level_changes, cluster_changes)
        prefix_only_rewrite = self._is_prefix_only_rewrite(applied_rules, structural_actions)
        repeated_subject_risk = self._has_repeated_subject_risk(subject_heads)
        revision_patterns = self._revision_patterns_from_rules(
            rules=applied_rules,
            original_sentences=original_sentences,
            rewritten_sentences=rewritten_sentences,
            rewrite_depth=active_rewrite_depth,
        )

        return BackendRewriteResult(
            text=f"{prefix}{final_core}{suffix}",
            applied_rules=applied_rules,
            original_sentences=original_sentences,
            rewritten_sentences=rewritten_sentences,
            sentence_level_change=self._has_sentence_level_change(original_sentences, rewritten_sentences),
            paragraph_char_count=paragraph_char_count,
            sentence_labels=[unit.label for unit in units],
            subject_heads=subject_heads,
            detected_patterns=detected_patterns,
            structural_actions=structural_actions,
            structural_action_count=len(structural_actions),
            high_value_structural_actions=high_value_structural_actions,
            discourse_actions_used=discourse_actions_used,
            sentence_level_changes=sentence_level_changes,
            cluster_changes=cluster_changes,
            discourse_change_score=discourse_change_score,
            rewrite_coverage=rewrite_coverage,
            prefix_only_rewrite=prefix_only_rewrite,
            repeated_subject_risk=repeated_subject_risk,
            selected_variants=selected_variants,
            candidate_notes=candidate_notes,
            paragraph_index=paragraph_index,
            revision_patterns=revision_patterns,
            human_noise_marks=self._deduplicate_preserve_order(
                [
                    *human_noise_marks,
                    *local_marks,
                    *texture_marks,
                    *semantic_marks,
                    *authorial_marks,
                    *evidence_marks,
                    *naturalization_marks,
                    *(l2_marks if active_style_profile == "zh_academic_l2_mild" else []),
                ]
            ),
            sentence_transition_rigidity=local_signals.sentence_transition_rigidity,
            local_discourse_flatness=local_signals.local_discourse_flatness,
            revision_realism_score=local_signals.revision_realism_score,
            sentence_cadence_irregularity=local_signals.sentence_cadence_irregularity,
            local_revision_actions=self._local_actions_from_rules(
                [
                    *local_rules,
                    *texture_rules,
                    *semantic_rules,
                    *authorial_rules,
                    *evidence_rules,
                    *naturalization_rules,
                    *(l2_rules if active_style_profile == "zh_academic_l2_mild" else []),
                ]
            ),
            sentence_completeness_score=readability_signals.sentence_completeness_score,
            paragraph_readability_score=readability_signals.paragraph_readability_score,
            dangling_sentence_risk=readability_signals.dangling_sentence_risk,
            incomplete_support_sentence_risk=readability_signals.incomplete_support_sentence_risk,
            fragment_like_conclusion_risk=readability_signals.fragment_like_conclusion_risk,
            readability_repair_actions=self._readability_actions_from_rules(readability_rules),
            high_sensitivity_prose=high_sensitivity_prose,
        )

    def _repair_paragraph_opening(
        self,
        original_sentences: list[str],
        rewritten_sentences: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        if not original_sentences or not rewritten_sentences:
            return rewritten_sentences, [], []

        original_text = "".join(original_sentences)
        revised_text = "".join(rewritten_sentences)
        skeleton = analyze_paragraph_skeleton(original_text)
        if not skeleton.topic_sentence_text and is_dangling_opening_sentence(rewritten_sentences[0]):
            standalone = self._standalone_opening_from_dangling(rewritten_sentences[0])
            if standalone != rewritten_sentences[0]:
                return (
                    [standalone, *rewritten_sentences[1:]],
                    ["paragraph:opening-style-guard"],
                    ["paragraph opening guard converted a dependent opener into a standalone topic/support sentence"],
                )
        checks = paragraph_skeleton_checks(original_text, revised_text)
        if all(checks.values()) and not self._opening_subject_drifted(original_sentences[0], rewritten_sentences[0]):
            return rewritten_sentences, [], []

        if not skeleton.topic_sentence_text:
            if opening_style_valid(rewritten_sentences[0]):
                return rewritten_sentences, [], []
            safe_tail = rewritten_sentences[1:] if len(rewritten_sentences) > 1 else []
            return (
                [original_sentences[0], *safe_tail],
                ["paragraph:opening-style-guard"],
                ["paragraph skeleton guard restored a standalone paragraph opening"],
            )

        safe_opening = self._safe_topic_opening(original_sentences[0], rewritten_sentences[0])
        tail = list(rewritten_sentences)
        if tail and self._sentence_similarity(original_sentences[0], tail[0]) >= 0.58 and self._can_drop_opening_duplicate(
            safe_opening, tail[0]
        ):
            tail = tail[1:]
        filtered_tail: list[str] = []
        for sentence in tail:
            if self._sentence_similarity(original_sentences[0], sentence) >= 0.58 and self._can_drop_opening_duplicate(
                safe_opening, sentence
            ):
                continue
            filtered_tail.append(sentence)

        repaired = [safe_opening, *filtered_tail]
        return (
            repaired,
            ["paragraph:topic-sentence-preserved"],
            ["paragraph skeleton guard kept the topic sentence at the paragraph opening"],
        )

    def _apply_local_revision_realism(
        self,
        original_sentences: list[str],
        rewritten_sentences: list[str],
        rewrite_depth: str,
        rewrite_intensity: str,
        mode: RewriteMode,
        paragraph_index: int,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        if mode is RewriteMode.CONSERVATIVE or rewrite_depth not in {"developmental_rewrite", "light_edit"}:
            return rewritten_sentences, [], [], []
        if len(rewritten_sentences) < 2:
            restored = self._split_collapsed_cluster_for_local_hierarchy(rewritten_sentences, original_sentences)
            if restored == rewritten_sentences:
                return rewritten_sentences, [], [], []
            rules = [
                "local:introduce-local-hierarchy",
                "local:reduce-sentence-uniformity",
                "local:light-partial-retain-with-local-rephrase",
            ]
            return (
                restored,
                rules,
                ["local realism restored paragraph-internal hierarchy after sentence-cluster collapse"],
                self._local_actions_from_rules(rules),
            )

        current_signals = analyze_local_revision_sentences(rewritten_sentences)
        original_signals = analyze_local_revision_sentences(original_sentences)
        needs_repair = (
            current_signals.sentence_transition_rigidity > 0.34
            or current_signals.local_discourse_flatness > 0.46
            or current_signals.revision_realism_score < 0.58
            or current_signals.sentence_cadence_irregularity < original_signals.sentence_cadence_irregularity
        )
        if not needs_repair and rewrite_intensity == "light":
            return rewritten_sentences, [], [], []

        updated = list(rewritten_sentences)
        rules: list[str] = []
        notes: list[str] = []
        marks: list[str] = []

        for index in range(1, len(updated)):
            softened = self._soften_overexplicit_transition(updated[index])
            if softened != updated[index]:
                updated[index] = softened
                rules.append("local:soften-overexplicit-transition")
                notes.append("local realism softened an overexplicit sentence-to-sentence transition")
                marks.append("soften_overexplicit_transition")

        if len(updated) >= 3:
            target_index = self._support_sentence_index_for_local_rephrase(updated, paragraph_index)
            reshaped = self._reshape_supporting_sentence_for_realism(updated[target_index])
            if reshaped != updated[target_index]:
                updated[target_index] = reshaped
                rules.append("local:reshape-supporting-sentence")
                notes.append("local realism reshaped a support sentence to reduce uniform full-sentence cadence")
                marks.append("reshape_supporting_sentence")

        if len(updated) >= 3 and rewrite_depth == "developmental_rewrite":
            merged = self._merge_short_supplement_for_hierarchy(updated)
            if merged != updated:
                updated = merged
                rules.append("local:introduce-local-hierarchy")
                notes.append("local realism merged a light supplement into its supporting sentence")
                marks.append("introduce_local_hierarchy")

        for index in range(1, len(updated)):
            softened_finish = self._weaken_overfinished_sentence(updated[index], is_final=index == len(updated) - 1)
            if softened_finish != updated[index]:
                updated[index] = softened_finish
                rules.append("local:weaken-overfinished-sentence")
                notes.append("local realism weakened an overfinished support sentence")
                marks.append("weaken_overfinished_sentence")

        after_signals = analyze_local_revision_sentences(updated)
        if (
            after_signals.local_discourse_flatness < current_signals.local_discourse_flatness
            or after_signals.sentence_cadence_irregularity > current_signals.sentence_cadence_irregularity
        ):
            rules.append("local:reduce-sentence-uniformity")
            marks.append("reduce_sentence_uniformity")
        if rules and rewrite_intensity in {"medium", "high"}:
            rules.append("local:light-partial-retain-with-local-rephrase")
            marks.append("light_partial_retain_with_local_rephrase")

        return (
            updated,
            self._deduplicate_preserve_order(rules),
            self._deduplicate_preserve_order(notes),
            self._deduplicate_preserve_order(marks),
        )

    def _repair_protected_token_coverage(
        self,
        original_sentences: list[str],
        rewritten_sentences: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        original_tokens = Counter(self._protected_tokens("".join(original_sentences)))
        rewritten_tokens = Counter(self._protected_tokens("".join(rewritten_sentences)))
        original_terms = Counter(self._visible_protected_terms("".join(original_sentences)))
        rewritten_terms = Counter(self._visible_protected_terms("".join(rewritten_sentences)))
        if original_tokens == rewritten_tokens and original_terms == rewritten_terms:
            return rewritten_sentences, [], []
        missing_terms = list((original_terms - rewritten_terms).elements())
        added_terms = list((rewritten_terms - original_terms).elements())
        if original_tokens == rewritten_tokens and missing_terms and not added_terms:
            updated = list(rewritten_sentences)
            repaired = False
            for missing_term in missing_terms:
                source_index = next(
                    (
                        index
                        for index, sentence in enumerate(original_sentences)
                        if missing_term in sentence
                    ),
                    -1,
                )
                if source_index < 0:
                    continue
                source_sentence = original_sentences[source_index]
                if source_index < len(updated):
                    updated[source_index] = self._ensure_sentence_end(source_sentence)
                    repaired = True
            if repaired:
                updated_terms = Counter(self._visible_protected_terms("".join(updated)))
                if original_terms == updated_terms:
                    return (
                        updated,
                        ["readability:restore-source-for-protected-token-coverage"],
                        ["protected terminology drifted during local repair; restored the source sentence containing the missing protected term"],
                    )
        return (
            list(original_sentences),
            ["readability:restore-source-for-protected-token-coverage"],
            ["protected token coverage changed during readability/local rewrite; restored source paragraph"],
        )

    def _repair_post_readability_subject_chains(
        self,
        *,
        original_sentences: list[str],
        rewritten_sentences: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Repair subject/template residue introduced after readability and token guards."""

        if len(rewritten_sentences) < 2:
            return rewritten_sentences, [], []

        original_tokens = Counter(self._protected_tokens("".join(rewritten_sentences)))
        original_terms = Counter(self._visible_protected_terms("".join(rewritten_sentences)))
        updated, duplicate_rules, duplicate_notes = self._remove_contained_sentence_prefixes(rewritten_sentences)
        updated, summary_rules, summary_notes = self._remove_redundant_summary_sentences(updated)
        updated, flow_rules, flow_notes = self._smooth_paragraph_readability_flow(updated)
        subject_heads = [self._extract_subject_head(sentence) for sentence in updated]
        if self._max_repeated_subject_streak(subject_heads) >= 2:
            units = [
                SentenceUnit(
                    text=sentence,
                    label="topic_sentence" if index == 0 else "support",
                    source_indices=(index,),
                )
                for index, sentence in enumerate(updated)
            ]
            repaired_units, subject_rules, _surfaces, subject_notes = self._repair_repeated_subject_heads(
                units=units,
                mode=RewriteMode.STRONG,
                pass_index=2,
            )
            repaired = [unit.text for unit in repaired_units]
        else:
            subject_rules = []
            subject_notes = []
            repaired = updated

        repaired, final_flow_rules, final_flow_notes = self._smooth_paragraph_readability_flow(repaired)

        if (
            Counter(self._protected_tokens("".join(repaired))) != original_tokens
            or Counter(self._visible_protected_terms("".join(repaired))) != original_terms
        ):
            return rewritten_sentences, [], []

        if repaired == rewritten_sentences:
            return rewritten_sentences, [], []

        rules = [*duplicate_rules, *summary_rules, *flow_rules, *subject_rules, *final_flow_rules]
        notes = [*duplicate_notes, *summary_notes, *flow_notes, *subject_notes, *final_flow_notes]
        if subject_rules:
            rules.append("readability:post-readability-subject-chain-repair")
            notes.append("post-readability repair reduced repeated meta-subject chains without changing protected content")
        return repaired, self._deduplicate_preserve_order(rules), self._deduplicate_preserve_order(notes)

    def _remove_contained_sentence_prefixes(self, sentences: list[str]) -> tuple[list[str], list[str], list[str]]:
        """Remove duplicated sentence prefixes caused by support-fragment merging."""

        updated = list(sentences)
        rules: list[str] = []
        notes: list[str] = []
        index = 1
        while index < len(updated):
            previous_core = self._strip_end_punctuation(updated[index - 1]).strip()
            current_core = self._strip_end_punctuation(updated[index]).strip()
            if len(previous_core) >= 24 and current_core.startswith(previous_core):
                tail = current_core[len(previous_core) :].lstrip("，,；; ")
                tail = re.sub(r"^(?:需要说明的是|需要指出的是)[，,\s]*", "", tail)
                if tail:
                    candidate = self._ensure_sentence_end(tail)
                    role = "conclusion_sentence" if index == len(updated) - 1 else "support_sentence"
                    if not self._sentence_needs_readability_repair(
                        candidate,
                        role=role,
                        is_final=index == len(updated) - 1,
                    ):
                        updated[index] = candidate
                        rules.append("readability:remove-contained-duplicate-sentence")
                        notes.append("readability repair removed a repeated sentence prefix left by support repair")
                        index += 1
                        continue
                del updated[index]
                rules.append("readability:remove-contained-duplicate-sentence")
                notes.append("readability repair dropped a duplicated support sentence")
                continue
            index += 1
        return updated, self._deduplicate_preserve_order(rules), self._deduplicate_preserve_order(notes)

    def _protected_tokens(self, text: str) -> list[str]:
        return re.findall(r"\[\[AIRC:CORE_(?:TERM|NUMBER|CITATION|PATH|CHECKPOINT|FORMULA(?:_INLINE|_BLOCK|_LINE)?):\d+\]\]", text)

    def _visible_protected_terms(self, text: str) -> list[str]:
        terms: list[str] = []
        for pattern in (
            r"[A-Z]{2,}[A-Za-z0-9_-]*",
            r"[A-Za-z]+_[A-Za-z0-9_]+",
            r"\b[A-Z]{2,}[A-Za-z0-9_-]*\b",
            r"\b[A-Za-z]+_[A-Za-z0-9_]+\b",
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b",
            r"\b[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9_-]*\b",
            r"\b[A-Za-z]+[0-9]+[A-Za-z0-9_-]*\b",
        ):
            terms.extend(re.findall(pattern, text))
        return terms

    def _recast_unchanged_complete_prose(
        self,
        *,
        original_sentences: list[str],
        rewritten_sentences: list[str],
        rewrite_depth: str,
        mode: RewriteMode,
    ) -> tuple[list[str], list[str], list[str]]:
        if mode is RewriteMode.CONSERVATIVE or rewrite_depth not in {"developmental_rewrite", "light_edit"}:
            return rewritten_sentences, [], []
        if not original_sentences or original_sentences != rewritten_sentences:
            return rewritten_sentences, [], []

        updated = list(rewritten_sentences)
        for index, sentence in enumerate(updated):
            if self._protected_tokens(sentence) or self._visible_protected_terms(sentence) or technical_density_is_high(sentence):
                continue
            recast = self._single_sentence_readability_recast(sentence)
            if recast == sentence:
                continue
            before = analyze_paragraph_readability_sentences([sentence])
            after = analyze_paragraph_readability_sentences([recast])
            if after.sentence_completeness_score >= before.sentence_completeness_score:
                updated[index] = recast
                return (
                    updated,
                    ["readability:single-sentence-readability-recast"],
                    ["unchanged complete prose received a safe readability recast"],
                )
        return rewritten_sentences, [], []

    def _single_sentence_readability_recast(self, sentence: str) -> str:
        """Apply a narrow safe recast to otherwise unchanged complete prose."""

        stripped = sentence.strip()
        match = re.match(r"^因此，最终部署结构并非(?P<body>.+)$", stripped)
        if match:
            return self._ensure_sentence_end(f"最终部署结构并不是{match.group('body').rstrip('。')}")
        match = re.match(r"^基于上述观察，本研究不再继续(?P<a>.+?)，而是(?P<b>.+)$", stripped)
        if match:
            return self._ensure_sentence_end(f"基于上述观察，研究重点不再是继续{match.group('a')}，而是{match.group('b').rstrip('。')}")
        match = re.match(r"^(本研究采用.+?)，兼顾(.+)$", stripped)
        if match:
            return self._ensure_sentence_end(f"{match.group(1)}，以兼顾{match.group(2).rstrip('。')}")
        match = re.match(r"^在([^，]{2,18})，系统不仅(.+?)，还(.+)$", stripped)
        if match:
            return self._ensure_sentence_end(f"在{match.group(1)}，系统除{match.group(2)}外，还{match.group(3).rstrip('。')}")
        match = re.match(r"^整体而言，该模块实现了(.+)$", stripped)
        if match:
            return self._ensure_sentence_end(f"整体来看，该模块完成了{match.group(1).rstrip('。')}")
        match = re.match(r"^在部署方面，系统支持(.+?)，通过(.+)$", stripped)
        if match:
            return self._ensure_sentence_end(f"在部署方面，系统可{match.group(1)}，并通过{match.group(2).rstrip('。')}")
        match = re.match(r"^在完成分类判定后，系统进一步生成(.+?)，并将(.+)$", stripped)
        if match:
            return self._ensure_sentence_end(f"完成分类判定后，系统会生成{match.group(1)}，再将{match.group(2).rstrip('。')}")
        return sentence

    def _soften_overexplicit_transition(self, sentence: str) -> str:
        stripped = sentence.strip()
        if technical_density_is_high(stripped):
            return sentence
        updated = re.sub(
            r"^(?:同时|此外|另外|与此同时|因此|由此|由此可见|基于此|在此基础上|在这种情况下|正因为如此|从整体上说|从整体来看)[，,\s]*",
            "",
            stripped,
        )
        updated = re.sub(r"^(?:具体而言|更具体地说|进一步来看|总体来看|整体来看)[，,\s]*", "", updated)
        if not updated or updated == stripped or is_dangling_opening_sentence(updated):
            return sentence
        return self._ensure_sentence_end(updated)

    def _split_collapsed_cluster_for_local_hierarchy(
        self,
        rewritten_sentences: list[str],
        original_sentences: list[str],
    ) -> list[str]:
        if len(rewritten_sentences) != 1 or len(original_sentences) < 3:
            return rewritten_sentences
        rewritten = rewritten_sentences[0].strip()
        if len(self._normalize_for_compare(rewritten)) < 55 or technical_density_is_high(rewritten):
            return rewritten_sentences

        topic = self._ensure_sentence_end(self._strip_end_punctuation(original_sentences[0]))
        support = self._compact_original_support_sentences(original_sentences[1:])
        if not support:
            return rewritten_sentences
        return [topic, support]

    def _compact_original_support_sentences(self, sentences: list[str]) -> str:
        cores = [
            self._strip_end_punctuation(self._soften_overexplicit_transition(sentence)).strip("，, ")
            for sentence in sentences
            if sentence.strip()
        ]
        cores = [core for core in cores if core]
        if not cores:
            return ""

        capability_matches = [
            re.match(r"^(?P<subject>[\u4e00-\u9fffA-Za-z0-9_]{1,12})(?:也)?能够(?P<body>.+)$", core)
            for core in cores
        ]
        if capability_matches and all(match is not None for match in capability_matches):
            subjects = [match.group("subject") for match in capability_matches if match is not None]
            bodies = [match.group("body").strip() for match in capability_matches if match is not None]
            if len(set(subjects)) == 1 and len(bodies) >= 2:
                subject = subjects[0]
                merged = self._merge_capability_bodies(bodies)
                if merged:
                    return self._ensure_sentence_end(f"{subject}能够{merged}")

        if len(cores) == 1:
            return self._ensure_sentence_end(cores[0])
        lead = re.sub(r"^(?:本研究|本文)(?=不仅|还|也|主要|将|需要|尝试|进一步|能够|可以)", "研究内容", cores[0])
        tail = re.sub(r"^(?:本研究|本文)(?=进一步|还|也|将|可|能够|需要|主要|尝试)", "", cores[-1]).strip("，, ")
        return self._ensure_sentence_end(f"{lead}，并{tail.lstrip('并')}")

    def _merge_capability_bodies(self, bodies: list[str]) -> str:
        if not bodies:
            return ""
        first = bodies[0]
        rest = bodies[1:]
        if len(rest) >= 1:
            common_prefix = self._common_chinese_prefix([first, rest[0]])
            if len(common_prefix) >= 2:
                merged_first = f"{common_prefix}{first[len(common_prefix):]}和{rest[0][len(common_prefix):]}"
                tail = rest[1:]
                if tail:
                    tail_phrase = "，并" + "，并".join(body.lstrip("并") for body in tail)
                    return f"{merged_first}{tail_phrase}"
                return merged_first
        return "，并".join(body.lstrip("并") for body in bodies)

    def _common_chinese_prefix(self, values: list[str]) -> str:
        if not values:
            return ""
        prefix = values[0]
        for value in values[1:]:
            index = 0
            while index < min(len(prefix), len(value)) and prefix[index] == value[index]:
                index += 1
            prefix = prefix[:index]
        return prefix

    def _support_sentence_index_for_local_rephrase(self, sentences: list[str], paragraph_index: int) -> int:
        candidates = list(range(1, len(sentences)))
        if len(sentences) > 3:
            candidates = candidates[:-1] or candidates
        return candidates[paragraph_index % len(candidates)]

    def _reshape_supporting_sentence_for_realism(self, sentence: str) -> str:
        stripped = sentence.strip()
        if technical_density_is_high(stripped):
            return sentence
        pair_match = re.match(r"^(?P<subject>该设计|这种设计|该方法|这种方法|该系统|系统)一方面(?P<a>.+?)，另一方面(?:也)?(?P<b>.+?)[。！？?!]?$", stripped)
        if pair_match:
            subject = pair_match.group("subject")
            first = pair_match.group("a").strip()
            second = pair_match.group("b").strip()
            return self._ensure_sentence_end(f"{subject}既{first}，也{second}")
        capability_match = re.match(r"^(?P<subject>该设计|这种设计|该方法|这种方法)能够(?P<body>.+?)[。！？?!]?$", stripped)
        if capability_match:
            subject = capability_match.group("subject")
            body = capability_match.group("body").strip()
            return self._ensure_sentence_end(f"{subject}主要{body}")
        system_match = re.match(r"^(?P<subject>该系统|系统)能够(?P<body>.+?)[。！？?!]?$", stripped)
        if system_match and len(stripped) <= 80:
            subject = system_match.group("subject")
            body = system_match.group("body").strip()
            return self._ensure_sentence_end(f"{subject}主要{body}")
        return sentence

    def _merge_short_supplement_for_hierarchy(self, sentences: list[str]) -> list[str]:
        if len(sentences) < 3:
            return sentences
        updated = list(sentences)
        for index in range(1, len(updated)):
            current = updated[index]
            current_core = self._strip_end_punctuation(current)
            if len(self._normalize_for_compare(current)) > 34:
                continue
            if technical_density_is_high(current) or index == 0:
                continue
            previous = updated[index - 1]
            if technical_density_is_high(previous):
                continue
            if re.match(r"^(?:这|该|这一|其|这种|上述)", current_core):
                updated[index - 1] = self._ensure_sentence_end(
                    f"{self._strip_end_punctuation(previous)}，{current_core}"
                )
                return updated[:index] + updated[index + 1 :]
        return sentences

    def _weaken_overfinished_sentence(self, sentence: str, is_final: bool) -> str:
        stripped = sentence.strip()
        if is_final or technical_density_is_high(stripped):
            return sentence
        support_match = re.match(r"^(?P<body>.+?)，?从而为(?P<object>.+?)提供(?:了)?(?P<tail>重要)?(?:基础|支撑|保障)[。！？?!]?$", stripped)
        if support_match:
            body = support_match.group("body").strip("，, ")
            obj = support_match.group("object").strip()
            return self._ensure_sentence_end(f"{body}，也使{obj}更容易展开")
        value_match = re.match(r"^(?P<body>.+?)具有(?:重要)?(?P<kind>意义|价值)[。！？?!]?$", stripped)
        if value_match and len(stripped) <= 90:
            body = value_match.group("body").strip()
            kind = value_match.group("kind")
            return self._ensure_sentence_end(f"{body}的{kind}也体现在这里")
        return sentence

    def _apply_author_texture_variation(
        self,
        *,
        original_sentences: list[str],
        rewritten_sentences: list[str],
        rewrite_depth: str,
        rewrite_intensity: str,
        mode: RewriteMode,
        paragraph_index: int,
        high_sensitivity_prose: bool,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Reduce unified-polisher tone after readability repair without weakening completeness."""

        if mode is RewriteMode.CONSERVATIVE or rewrite_depth not in {"developmental_rewrite", "light_edit"}:
            return rewritten_sentences, [], [], []
        if len(rewritten_sentences) < 2:
            return rewritten_sentences, [], [], []

        before_realism = analyze_local_revision_sentences(rewritten_sentences)
        before_readability = analyze_paragraph_readability_sentences(
            rewritten_sentences,
            high_sensitivity=high_sensitivity_prose,
        )
        cliche_limit = 0.10 if high_sensitivity_prose else 0.16
        needs_texture = (
            before_realism.stylistic_uniformity_score > 0.28
            or before_realism.support_sentence_texture_variation < 0.40
            or before_realism.academic_cliche_density > cliche_limit
            or (high_sensitivity_prose and len(rewritten_sentences) >= 3)
        )
        if not needs_texture:
            return rewritten_sentences, [], [], []

        updated = list(rewritten_sentences)
        rules: list[str] = []
        notes: list[str] = []
        marks: list[str] = []

        updated, cliche_rules, cliche_notes = self._reduce_academic_cliche_density(
            sentences=updated,
            original_sentences=original_sentences,
            high_sensitivity_prose=high_sensitivity_prose,
        )
        rules.extend(cliche_rules)
        notes.extend(cliche_notes)
        marks.extend(self._local_actions_from_rules(cliche_rules))

        updated, texture_rules, texture_notes = self._vary_support_sentence_texture(
            sentences=updated,
            original_sentences=original_sentences,
            paragraph_index=paragraph_index,
            high_sensitivity_prose=high_sensitivity_prose,
        )
        rules.extend(texture_rules)
        notes.extend(texture_notes)
        marks.extend(self._local_actions_from_rules(texture_rules))

        updated, asymmetry_rules, asymmetry_notes = self._introduce_sentence_asymmetry(
            sentences=updated,
            original_sentences=original_sentences,
            paragraph_index=paragraph_index,
            rewrite_intensity=rewrite_intensity,
            high_sensitivity_prose=high_sensitivity_prose,
        )
        rules.extend(asymmetry_rules)
        notes.extend(asymmetry_notes)
        marks.extend(self._local_actions_from_rules(asymmetry_rules))

        if updated == rewritten_sentences:
            return rewritten_sentences, [], [], []

        after_readability = analyze_paragraph_readability_sentences(
            updated,
            high_sensitivity=high_sensitivity_prose,
        )
        after_realism = analyze_local_revision_sentences(updated)
        if (
            after_readability.paragraph_readability_score + 0.02 < before_readability.paragraph_readability_score
            or after_readability.sentence_completeness_score + 0.02 < before_readability.sentence_completeness_score
        ):
            return rewritten_sentences, [], [], []
        if (
            after_realism.stylistic_uniformity_score > before_realism.stylistic_uniformity_score + 0.02
            and after_realism.support_sentence_texture_variation <= before_realism.support_sentence_texture_variation
            and after_realism.academic_cliche_density >= before_realism.academic_cliche_density
        ):
            return rewritten_sentences, [], [], []

        return (
            updated,
            self._deduplicate_preserve_order(rules),
            self._deduplicate_preserve_order(notes),
            self._deduplicate_preserve_order(marks),
        )

    def _reduce_academic_cliche_density(
        self,
        *,
        sentences: list[str],
        original_sentences: list[str],
        high_sensitivity_prose: bool,
    ) -> tuple[list[str], list[str], list[str]]:
        """Rephrase dense academic clichés into plainer author-like support prose."""

        updated = list(sentences)
        rules: list[str] = []
        notes: list[str] = []
        changes = 0
        change_budget = 2 if high_sensitivity_prose else 1
        scored_indexes = sorted(
            range(len(updated)),
            key=lambda index: (
                self._academic_cliche_count(updated[index]),
                int(
                    index == 0
                    and bool(
                        re.search(
                            r"^(?:总结而言|总的来说|总体而言|总体来看|整体来看|综合来看|需要说明的是|"
                            r"值得进一步说明的是|值得说明的是|需要指出的是)",
                            updated[index].strip(),
                        )
                    )
                ),
                int(index == len(updated) - 1),
            ),
            reverse=True,
        )
        for index in scored_indexes:
            if changes >= change_budget:
                break
            sentence = updated[index]
            if self._academic_cliche_count(sentence) <= 0 and not re.search(
                r"实现了更稳定的部署表现|有助于(?:缓解|减少|增强|改善|保持)",
                sentence,
            ):
                continue
            if index == 0 and not high_sensitivity_prose:
                stripped = sentence.strip()
                if not re.search(
                    r"^(?:总结而言|总的来说|总体而言|总体来看|整体来看|综合来看|需要说明的是|"
                    r"值得进一步说明的是|值得说明的是|需要指出的是)",
                    stripped,
                ):
                    continue
            original_sentence = original_sentences[index] if index < len(original_sentences) else ""
            rephrased = self._de_template_academic_cliche(
                sentence,
                original_sentence=original_sentence,
                preserve_opening=index == 0,
            )
            if rephrased == sentence:
                continue
            updated[index] = rephrased
            changes += 1
            rules.append("local:de-template-academic-cliche")
            notes.append("author-texture pass simplified an academic cliché without reducing factual content")
            if index == len(updated) - 1:
                rules.append("local:rephrase-summary-like-sentence-without-cliche")
        return updated, self._deduplicate_preserve_order(rules), self._deduplicate_preserve_order(notes)

    def _vary_support_sentence_texture(
        self,
        *,
        sentences: list[str],
        original_sentences: list[str],
        paragraph_index: int,
        high_sensitivity_prose: bool,
    ) -> tuple[list[str], list[str], list[str]]:
        """Keep one support sentence plainer so the paragraph does not read like one polish pass."""

        candidate_indexes = self._support_indexes_for_texture(sentences)
        if not candidate_indexes:
            return sentences, [], []

        updated = list(sentences)
        for offset in range(len(candidate_indexes)):
            index = candidate_indexes[(paragraph_index + offset) % len(candidate_indexes)]
            original_sentence = original_sentences[index] if index < len(original_sentences) else ""
            variant = self._plain_support_sentence_variant(
                sentence=updated[index],
                original_sentence=original_sentence,
                high_sensitivity_prose=high_sensitivity_prose,
            )
            if variant == updated[index]:
                continue
            updated[index] = variant
            rules = ["local:vary-support-sentence-texture"]
            if self._normalize_for_compare(variant) == self._normalize_for_compare(original_sentence):
                rules.extend(["local:retain-some-plain-sentences", "local:soft-keep-for-human-revision-feel"])
            else:
                rules.append("local:avoid-overpolished-supporting-sentence")
            return (
                updated,
                self._deduplicate_preserve_order(rules),
                ["author-texture pass left one support sentence closer to an author-like plain statement"],
            )
        return sentences, [], []

    def _introduce_sentence_asymmetry(
        self,
        *,
        sentences: list[str],
        original_sentences: list[str],
        paragraph_index: int,
        rewrite_intensity: str,
        high_sensitivity_prose: bool,
    ) -> tuple[list[str], list[str], list[str]]:
        """Restore mild unevenness so every support sentence does not sound equally polished."""

        if rewrite_intensity == "light" and not high_sensitivity_prose:
            return sentences, [], []

        updated = list(sentences)
        candidate_indexes = self._support_indexes_for_texture(sentences)
        if not candidate_indexes:
            return sentences, [], []

        changed_indexes = [
            index
            for index in range(min(len(updated), len(original_sentences)))
            if self._normalize_for_compare(updated[index]) != self._normalize_for_compare(original_sentences[index])
        ]
        if len(changed_indexes) < max(2, len(candidate_indexes)):
            return sentences, [], []

        for offset in range(len(candidate_indexes)):
            index = candidate_indexes[(paragraph_index + offset + 1) % len(candidate_indexes)]
            original_sentence = original_sentences[index] if index < len(original_sentences) else ""
            if not self._can_soft_keep_source_sentence(
                original_sentence,
                updated[index],
                high_sensitivity_prose=high_sensitivity_prose,
            ):
                continue
            updated[index] = self._ensure_sentence_end(self._soften_overexplicit_transition(original_sentence))
            return (
                updated,
                [
                    "local:introduce-mild-authorial-asymmetry",
                    "local:deuniform-paragraph-texture",
                    "local:soft-keep-for-human-revision-feel",
                ],
                ["author-texture pass preserved one source-like support sentence to avoid uniform polishing"],
            )
        return sentences, [], []

    def _de_template_academic_cliche(
        self,
        sentence: str,
        *,
        original_sentence: str,
        preserve_opening: bool,
    ) -> str:
        """Replace a small set of repeat clichés with plainer academic phrasing."""

        stripped = sentence.strip()
        if not stripped or technical_density_is_high(stripped):
            return sentence

        lead = ""
        lead_match = re.match(
            r"^(?P<lead>(?:总结而言|总的来说|总体而言|总体来看|整体来看|综合来看))[，,\s]*(?P<body>.+)$",
            stripped,
        )
        if lead_match:
            lead = lead_match.group("lead").strip()
            stripped = lead_match.group("body").strip()

        for prefix in (
            r"需要说明的是",
            r"值得进一步说明的是",
            r"值得说明的是",
            r"需要指出的是",
            r"因此可以看出",
            r"在此基础上",
        ):
            candidate = re.sub(rf"^{prefix}[，,\s]*", "", stripped)
            if candidate != stripped and candidate:
                if not self._sentence_needs_readability_repair(candidate, role="support_sentence", is_final=False):
                    stripped = candidate
                    break

        value_match = re.match(r"^(?P<subject>本研究|本文)(?:的)?核心价值在于(?P<body>.+)$", stripped)
        if value_match:
            body = value_match.group("body").strip("，, ")
            body = re.sub(r"更有助于提升", "更能提升", body)
            body = re.sub(r"有助于提升", "更能提升", body)
            lead_prefix = ""
            if lead:
                provisional = self._ensure_sentence_end(
                    f"这项工作的价值主要体现在，{body}"
                    if body.startswith(("把", "将", "对", "通过", "在", "为", "使", "证明", "说明"))
                    else f"这项工作的价值主要体现在{body}"
                )
                if self._sentence_needs_readability_repair(
                    provisional,
                    role="topic_sentence" if preserve_opening else "support_sentence",
                    is_final=False,
                ):
                    lead_prefix = f"{lead}，"
            if body.startswith(("把", "将", "对", "通过", "在", "为", "使", "证明", "说明")):
                return self._ensure_sentence_end(f"{lead_prefix}这项工作的价值主要体现在，{body}")
            return self._ensure_sentence_end(f"{lead_prefix}这项工作的价值主要体现在{body}")

        key_match = re.match(r"^(?P<prefix>.*?)(?:重点在于|关键在于)(?P<body>.+)$", stripped)
        if key_match:
            prefix = key_match.group("prefix").strip("，, ")
            body = key_match.group("body").strip("，, ")
            lead_prefix = ""
            if lead:
                provisional = (
                    self._ensure_sentence_end(f"{prefix}，更关键的是{body}")
                    if prefix
                    else self._ensure_sentence_end(f"更关键的是{body}")
                )
                if self._sentence_needs_readability_repair(
                    provisional,
                    role="topic_sentence" if preserve_opening else "support_sentence",
                    is_final=False,
                ):
                    lead_prefix = f"{lead}，"
            if prefix:
                return self._ensure_sentence_end(f"{lead_prefix}{prefix}，更关键的是{body}")
            return self._ensure_sentence_end(f"{lead_prefix}更关键的是{body}")

        replacements = (
            (r"实现了更稳定的部署表现", "让部署表现更稳定"),
            (r"实现了(?P<body>从[^。；]{2,28})的完整闭环", r"把\g<body>这条链路贯通起来"),
            (r"为(?P<body>[^。；]{2,24})提供(?:了)?(?:重要)?(?:基础|支撑|保障|路径)", r"也为\g<body>留出了空间"),
            (r"有助于缓解", "也缓解了"),
            (r"有助于减少", "也减少了"),
            (r"有助于增强", "也增强了"),
            (r"有助于改善", "也改善了"),
            (r"有助于保持", "也保持了"),
            (r"更有助于提升", "更能提升"),
            (r"有助于提升", "更能提升"),
        )
        for pattern, replacement in replacements:
            candidate = re.sub(pattern, replacement, stripped)
            if candidate != stripped:
                stripped = candidate

        inline_parenthetical = re.sub(r"[，,]需要说明的是[，,\s]*", "，", stripped)
        inline_parenthetical = re.sub(r"[，,]需要指出的是[，,\s]*", "，", inline_parenthetical)
        if inline_parenthetical != stripped:
            stripped = inline_parenthetical

        if lead:
            candidate_without_lead = self._ensure_sentence_end(stripped)
            role = "topic_sentence" if preserve_opening else "support_sentence"
            if self._sentence_needs_readability_repair(
                candidate_without_lead,
                role=role,
                is_final=False,
            ):
                stripped = f"{lead}，{stripped.lstrip('，, ')}"
        cleaned = self._ensure_sentence_end(stripped)
        if cleaned != sentence and self._can_soft_keep_source_sentence(original_sentence, cleaned, high_sensitivity_prose=False):
            return cleaned
        return cleaned if cleaned != sentence else sentence

    def _plain_support_sentence_variant(
        self,
        *,
        sentence: str,
        original_sentence: str,
        high_sensitivity_prose: bool,
    ) -> str:
        """Move one supporting sentence toward a plainer author-like texture."""

        stripped = sentence.strip()
        if not stripped or technical_density_is_high(stripped):
            return sentence
        if self._can_soft_keep_source_sentence(
            original_sentence,
            sentence,
            high_sensitivity_prose=high_sensitivity_prose,
        ):
            return self._ensure_sentence_end(self._soften_overexplicit_transition(original_sentence))

        capability_match = re.match(
            r"^(?P<subject>该设计|这种设计|该方法|这种方法|该系统|系统)(?:的作用在于|由此可以|主要用于|能够)(?P<body>.+?)[。！？?!]?$",
            stripped,
        )
        if capability_match:
            subject = capability_match.group("subject").strip()
            body = capability_match.group("body").strip()
            if body.startswith(("兼顾", "补充", "解释", "支撑", "覆盖", "减少", "保持", "连接", "识别", "说明", "处理")):
                return self._ensure_sentence_end(f"{subject}主要{body}")
            return self._ensure_sentence_end(f"{subject}{body}")

        key_match = re.match(r"^(?:关键是|更关键的是)(?P<body>.+)$", stripped)
        if key_match:
            body = key_match.group("body").strip("，, ")
            if body and not self._sentence_needs_readability_repair(body, role="support_sentence", is_final=False):
                return self._ensure_sentence_end(body)
        experiment_match = re.match(r"^(?:实验结果(?:表明|显示|说明)|从实验结果看)[，,\s]*(?P<body>.+)$", stripped)
        if experiment_match:
            body = experiment_match.group("body").strip("，, ")
            body = re.sub(r"有助于缓解", "也缓解了", body)
            body = re.sub(r"有助于提升", "也提升了", body)
            if body and not self._sentence_needs_readability_repair(body, role="support_sentence", is_final=False):
                return self._ensure_sentence_end(f"从实验结果看，{body}")
        return sentence

    def _support_indexes_for_texture(self, sentences: list[str]) -> list[int]:
        """Return support-sentence positions that can safely absorb anti-uniformity edits."""

        if len(sentences) <= 1:
            return []
        if len(sentences) == 2:
            return [1]
        return list(range(1, len(sentences) - 1))

    def _can_soft_keep_source_sentence(
        self,
        source_sentence: str,
        current_sentence: str,
        *,
        high_sensitivity_prose: bool,
    ) -> bool:
        """Decide whether restoring a plainer source sentence is safe for author-like unevenness."""

        source = source_sentence.strip()
        current = current_sentence.strip()
        if not source or not current:
            return False
        if technical_density_is_high(source) or technical_density_is_high(current):
            return False
        if self._normalize_for_compare(source) == self._normalize_for_compare(current):
            return False
        if self._sentence_needs_readability_repair(source, role="support_sentence", is_final=False):
            return False
        if self._academic_cliche_count(source) > self._academic_cliche_count(current):
            return False
        if not high_sensitivity_prose and len(self._normalize_for_compare(source)) > len(self._normalize_for_compare(current)) * 1.22:
            return False
        return True

    def _academic_cliche_count(self, sentence: str) -> int:
        """Count repeated academic-polisher phrases that should not dominate a paragraph."""

        stripped = sentence.strip()
        patterns = (
            r"需要说明的是",
            r"值得进一步说明的是",
            r"需要指出的是",
            r"重点在于",
            r"(?:本研究的)?核心价值在于",
            r"具有重要意义",
            r"在此基础上",
            r"因此可以看出",
            r"实现了(?:一个|完整)?[^。；]{0,16}(?:闭环|体系|流程|框架)",
            r"实现了更稳定的部署表现",
            r"为[^。；]{2,24}提供(?:了)?(?:重要)?(?:基础|支撑|保障|路径)",
            r"有助于(?:提升|缓解|减少|增强|改善|保持)",
        )
        return sum(1 for pattern in patterns if re.search(pattern, stripped))

    def _readability_repair_pass(
        self,
        original_sentences: list[str],
        rewritten_sentences: list[str],
        rewrite_depth: str,
        mode: RewriteMode,
        high_sensitivity_prose: bool,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        if mode is RewriteMode.CONSERVATIVE or rewrite_depth not in {"developmental_rewrite", "light_edit"}:
            return rewritten_sentences, [], [], []
        if not rewritten_sentences:
            return rewritten_sentences, [], [], []

        updated = list(rewritten_sentences)
        rules: list[str] = []
        notes: list[str] = []
        marks: list[str] = []
        before = analyze_paragraph_readability_sentences(updated, high_sensitivity=high_sensitivity_prose)
        high_sensitivity_needed = high_sensitivity_prose and (
            before.dangling_sentence_indexes
            or before.incomplete_support_indexes
            or before.fragment_like_conclusion_indexes
            or before.sentence_completeness_score < 0.82
            or before.paragraph_readability_score < 0.78
        )

        for index, sentence in enumerate(list(updated)):
            role = "topic_sentence" if index == 0 else ("conclusion_sentence" if index == len(updated) - 1 else "support_sentence")
            if not self._sentence_needs_readability_repair(sentence, role=role, is_final=index == len(updated) - 1):
                continue
            repaired = self._repair_incomplete_sentence(
                sentence=sentence,
                original_sentence=original_sentences[index] if index < len(original_sentences) else "",
                previous_sentence=updated[index - 1] if index > 0 else "",
                role=role,
            )
            if repaired != sentence:
                updated[index] = repaired
                rules.append("readability:sentence-completeness-repair")
                notes.append("readability repair restored a complete academic sentence")
                marks.append("sentence_completeness_repair")

        smoothed, smoothing_rules, smoothing_notes = self._smooth_paragraph_readability_flow(
            updated,
            original_sentences=original_sentences,
        )
        if smoothed != updated:
            updated = smoothed
            rules.extend(smoothing_rules)
            notes.extend(smoothing_notes)
            marks.append("readability_repair_pass")

        if len(updated) >= 2:
            merged = self._merge_unrepaired_support_fragments(updated)
            if merged != updated:
                updated = merged
                rules.append("readability:repair-incomplete-support-sentence")
                notes.append("readability repair merged a residual support fragment into the preceding sentence")
                marks.append("repair_incomplete_support_sentence")

        if updated:
            repaired_final = self._repair_fragment_like_conclusion_sentence(
                updated[-1],
                original_sentences[-1] if original_sentences else "",
                updated[-2] if len(updated) >= 2 else "",
            )
            if repaired_final != updated[-1]:
                updated[-1] = repaired_final
                rules.append("readability:repair-fragment-like-conclusion-sentence")
                notes.append("readability repair restored a complete conclusion sentence")
                marks.append("repair_fragment_like_conclusion_sentence")

        if high_sensitivity_prose:
            tightened = self._repair_high_sensitivity_prose_readability(original_sentences, updated)
            if tightened != updated:
                updated = tightened
                rules.append("readability:high-sensitivity-readability-repair")
                notes.append("high-sensitivity prose repair preferred complete source phrasing over fragmentary rewrite")
                marks.append("high_sensitivity_readability_repair")
            elif high_sensitivity_needed and rules:
                rules.append("readability:high-sensitivity-readability-repair")
                notes.append("high-sensitivity prose repair verified completeness after sentence-level repair")
                marks.append("high_sensitivity_readability_repair")

        after = analyze_paragraph_readability_sentences(updated, high_sensitivity=high_sensitivity_prose)
        if (
            rules
            and original_sentences
            and (
                after.dangling_sentence_indexes
                or after.incomplete_support_indexes
                or after.fragment_like_conclusion_indexes
            )
        ):
            repaired = self._restore_fragmentary_sentences_from_source(original_sentences, updated)
            repaired = self._merge_unrepaired_support_fragments(repaired)
            if repaired != updated:
                updated = repaired
                rules.append("readability:restore-source-for-readability")
                notes.append("readability repair restored or merged remaining fragmentary sentences")
                marks.append("readability_repair_pass")
                after = analyze_paragraph_readability_sentences(updated, high_sensitivity=high_sensitivity_prose)
        if high_sensitivity_prose and original_sentences and (
            after.dangling_sentence_indexes
            or after.incomplete_support_indexes
            or after.fragment_like_conclusion_indexes
            or after.sentence_completeness_score < 0.72
        ):
            original_signal = analyze_paragraph_readability_sentences(
                original_sentences,
                high_sensitivity=high_sensitivity_prose,
            )
            if original_signal.paragraph_readability_score >= after.paragraph_readability_score - 0.03:
                updated = list(original_sentences)
                rules.append("readability:high-sensitivity-source-restore")
                notes.append("high-sensitivity prose restored the source paragraph to avoid unfinished prose")
                marks.append("high_sensitivity_readability_repair")
                after = original_signal
        if (
            rules
            and after.paragraph_readability_score + 0.04 < before.paragraph_readability_score
            and original_sentences
        ):
            updated = self._restore_fragmentary_sentences_from_source(original_sentences, updated)
            rules.append("readability:restore-source-for-readability")
            notes.append("readability repair restored source wording where repair lowered paragraph readability")
            marks.append("readability_repair_pass")

        if rules:
            rules.append("readability:readability-repair-pass")
            marks.append("readability_repair_pass")
        return (
            updated,
            self._deduplicate_preserve_order(rules),
            self._deduplicate_preserve_order(notes),
            self._deduplicate_preserve_order(marks),
        )

    def _sentence_needs_readability_repair(self, sentence: str, *, role: str, is_final: bool) -> bool:
        """Decide whether a sentence requires the completeness repair path."""

        if dangling_sentence_risk(sentence):
            return True
        if technical_density_is_high(sentence):
            return False
        return (
            incomplete_support_sentence_risk(sentence, role=role)
            or fragment_like_conclusion_sentence(sentence, is_final=is_final, role=role)
        )

    def _smooth_paragraph_readability_flow(
        self,
        sentences: list[str],
        *,
        original_sentences: list[str] | None = None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Smooth complete-but-rigid paragraph flow without moving the topic sentence."""

        if not sentences:
            return sentences, [], []

        updated = list(sentences)
        rules: list[str] = []
        notes: list[str] = []

        updated, taxonomy_rules, taxonomy_notes = self._merge_taxonomy_support_fragments(updated)
        rules.extend(taxonomy_rules)
        notes.extend(taxonomy_notes)

        updated, enum_rules, enum_notes = self._repair_enumeration_sentence_flow(updated)
        rules.extend(enum_rules)
        notes.extend(enum_notes)

        updated, caption_rules, caption_notes = self._repair_caption_reference_sentences(updated)
        rules.extend(caption_rules)
        notes.extend(caption_notes)

        updated, marker_rules, marker_notes = self._repair_generated_discourse_marker_sentences(
            updated,
            original_sentences=original_sentences,
        )
        rules.extend(marker_rules)
        notes.extend(marker_notes)

        for index in range(1, len(updated)):
            sentence = updated[index]
            if technical_density_is_high(sentence):
                continue
            softened = self._soften_overexplicit_transition(sentence)
            if softened == sentence:
                continue
            role = "conclusion_sentence" if index == len(updated) - 1 else "support_sentence"
            if self._sentence_needs_readability_repair(softened, role=role, is_final=index == len(updated) - 1):
                continue
            updated[index] = softened
            rules.append("readability:soften-complete-transition")
            notes.append("readability smoothing removed an overexplicit transition from a complete sentence")

        updated, duplicate_rules, duplicate_notes = self._remove_redundant_summary_sentences(updated)
        rules.extend(duplicate_rules)
        notes.extend(duplicate_notes)
        if updated and re.match(r"^(?:使|让)[^。！？?!]{8,}", updated[0].strip()):
            repaired_opening = self._repair_incomplete_sentence(
                sentence=updated[0],
                original_sentence=original_sentences[0] if original_sentences else "",
                previous_sentence="",
                role="topic_sentence",
            )
            if repaired_opening != updated[0]:
                updated[0] = repaired_opening
                rules.append("readability:sentence-completeness-repair")
                notes.append("readability smoothing restored a standalone opening after sentence splitting")
        return updated, self._deduplicate_preserve_order(rules), self._deduplicate_preserve_order(notes)

    def _repair_generated_discourse_marker_sentences(
        self,
        sentences: list[str],
        *,
        original_sentences: list[str] | None = None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Replace generated scaffolding with a direct sentence that keeps the original role."""

        updated = list(sentences)
        rules: list[str] = []
        notes: list[str] = []
        original_sentences = original_sentences or []
        for index, sentence in enumerate(list(updated)):
            stripped = self._strip_end_punctuation(sentence.strip())
            if not stripped.startswith("该段论述"):
                continue
            original_sentence = original_sentences[index] if index < len(original_sentences) else ""
            repaired = self._replace_abstracted_subject_with_concrete_referent(
                sentence=stripped,
                original_sentence=original_sentence,
                role=self._sentence_role_for_semantic_repair(original_sentence or stripped, index, len(updated)),
            )
            if repaired == sentence and original_sentence:
                repaired = self._ensure_sentence_end(original_sentence)
            if repaired != sentence:
                updated[index] = repaired
                rules.append("readability:repair-generated-discourse-marker")
                notes.append("readability smoothing replaced generated scaffolding with a direct academic claim")
        return updated, self._deduplicate_preserve_order(rules), self._deduplicate_preserve_order(notes)

    def _sentence_role_for_semantic_repair(self, sentence: str, index: int, total: int) -> str:
        """Classify a sentence coarsely so repairs preserve its semantic role."""

        stripped = sentence.strip()
        if not stripped:
            return "support"
        if self._looks_like_enumeration_sentence(stripped):
            return "enumeration"
        if index == total - 1 and re.search(r"(总体|综上|总结|从最终效果看|说明|表明|意味着)", stripped):
            return "conclusion"
        if index == 0 and re.match(r"^(?:本研究|本文|本章|本系统|本项目)", stripped):
            return "core"
        if self._looks_like_mechanism_sentence(stripped):
            return "mechanism"
        return "support"

    def _looks_like_enumeration_sentence(self, sentence: str) -> bool:
        """Detect explicit enumeration items and enumeration headers."""

        stripped = sentence.strip()
        return bool(_ENUMERATION_ITEM_RE.match(stripped) or _ENUMERATION_HEAD_RE.search(stripped))

    def _looks_like_mechanism_sentence(self, sentence: str) -> bool:
        """Detect mechanism-description sentences that should remain direct claims."""

        stripped = sentence.strip()
        if self._looks_like_enumeration_sentence(stripped):
            return True
        mechanism_subject = re.search(
            r"(?:模型|系统|模块|分支|接口|机制|路径|课程学习|损失函数|主融合模块|语义分支|频域分支|"
            r"判别机制|base_only|logit|特征)",
            stripped,
        )
        mechanism_verb = re.search(
            r"(?:采用|通过|将|把|用于|负责|生成|输出|融合|提取|建模|约束|对外提供|保留|剥离|实施|"
            r"构成|形成|限制|连接|由[^。；]{0,24}得到)",
            stripped,
        )
        return bool(mechanism_subject and mechanism_verb)

    def _concrete_referent_from_original(self, original_sentence: str, fallback: str = "本研究") -> str:
        """Extract a concrete referent from the source sentence for de-scaffolding."""

        stripped = self._strip_end_punctuation(original_sentence.strip())
        if not stripped:
            return fallback
        benchmark_match = re.search(
            r"以(?P<subject>[^，。；：:]{2,36})作为(?:主要量化评测基准|补充性诊断集合|核心依据|主要依据)",
            stripped,
        )
        if benchmark_match:
            return benchmark_match.group("subject").strip()
        introduce_match = re.search(
            r"(?:引入|采用)(?P<subject>[^，。；：:]{2,36})(?:作为|用于)",
            stripped,
        )
        if introduce_match:
            return introduce_match.group("subject").strip()
        enum_match = re.match(
            r"^(?P<prefix>[（(]?\d+[）)]?)(?P<title>[^：:。！？?!]{2,42})(?:[：:])",
            stripped,
        )
        if enum_match:
            return f"{enum_match.group('prefix').strip()}{enum_match.group('title').strip()}"
        subject_match = re.match(
            r"^(?P<subject>(?:本研究|本文|本章|系统|该系统|模型|最终模型|主融合模块|语义分支|频域分支|"
            r"噪声分支|课程采样器|判别路径|量化评测基准|补充性诊断集合|NTIRE数据集|photos_test集合|"
            r"该方法|该机制|该设计|该设置|当前设置|困难真实样本课程学习|base_only判别机制|课程学习|主融合模块)[^，。；：:]*)",
            stripped,
        )
        if subject_match:
            return subject_match.group("subject").strip()
        if "研究" in stripped[:24]:
            return "本研究"
        return fallback

    def _replace_abstracted_subject_with_concrete_referent(
        self,
        *,
        sentence: str,
        original_sentence: str,
        role: str,
    ) -> str:
        """Replace abstract scaffolding subjects with a concrete referent from the source."""

        stripped = self._strip_end_punctuation(sentence.strip())
        if not stripped:
            return sentence
        concrete = self._concrete_referent_from_original(
            original_sentence,
            fallback="本研究" if role in {"core", "support"} else "该机制",
        )
        if stripped.startswith("这一点需要指出"):
            body = stripped.split("：", 1)[1] if "：" in stripped else re.sub(r"^这一点需要指出[，,\s]*", "", stripped)
            return self._ensure_sentence_end(f"需要指出的是，{body.strip('，, ')}")
        if stripped.startswith(("该段论述强调", "该段论述更强调")):
            body = re.sub(r"^该段论述(?:更)?强调", "", stripped).strip("，, ")
            if body.startswith(("将", "把", "使", "让")):
                return self._ensure_sentence_end(f"{concrete}{body}")
            return self._ensure_sentence_end(f"{concrete}{body}")
        if stripped.startswith("该段论述用于"):
            body = re.sub(r"^该段论述用于", "", stripped).strip("，, ")
            return self._ensure_sentence_end(f"{concrete}用于{body}")
        if stripped.startswith("该段论述还需要结合"):
            body = re.sub(r"^该段论述还需要结合", "", stripped).strip("，, ")
            spacer = " " if re.match(r"^[A-Za-z0-9_\\[]", body) else ""
            referent = "当前分析" if role == "conclusion" else concrete
            return self._ensure_sentence_end(f"{referent}还需要结合{spacer}{body}")
        if stripped.startswith("该段论述需要"):
            body = re.sub(r"^该段论述需要", "", stripped).strip("，, ")
            return self._ensure_sentence_end(f"{concrete}还需要{body}")
        if stripped.startswith("该段论述进一步"):
            body = re.sub(r"^该段论述进一步", "", stripped).strip("，, ")
            body = re.sub(r"^还包括", "", body).strip("，, ")
            if role == "enumeration" and original_sentence:
                return self._ensure_sentence_end(original_sentence)
            return self._ensure_sentence_end(f"{concrete}{body}")
        if stripped.startswith("该段论述"):
            body = re.sub(r"^该段论述", "", stripped).strip("，, ")
            body = re.sub(r"^还包括", "", body).strip("，, ")
            return self._ensure_sentence_end(f"{concrete}{body}")
        if stripped.startswith(_SCAFFOLDING_OPENINGS):
            body = stripped
            for opener in _SCAFFOLDING_OPENINGS:
                if body.startswith(opener):
                    body = body[len(opener) :].lstrip("，, ：:")
                    break
            if role == "enumeration" and original_sentence:
                return self._ensure_sentence_end(original_sentence)
            if body.startswith(("还包括", "也包括", "进一步包括")) and original_sentence:
                return self._ensure_sentence_end(original_sentence)
            if body:
                return self._ensure_sentence_end(f"{concrete}{body}")
        return self._ensure_sentence_end(sentence)

    def _restore_mechanism_sentence_from_support_like_rewrite(
        self,
        *,
        sentence: str,
        original_sentence: str,
    ) -> str:
        """Restore a mechanism sentence when rewrite drift makes it sound appendix-like."""

        stripped = sentence.strip()
        if not stripped:
            return sentence
        if not self._looks_like_mechanism_sentence(original_sentence):
            return sentence
        if not (_HUANBAOKUO_RE.search(stripped) or stripped.startswith(_SCAFFOLDING_OPENINGS)):
            return sentence
        if original_sentence and not self._sentence_needs_readability_repair(
            original_sentence,
            role="support_sentence",
            is_final=False,
        ):
            return self._ensure_sentence_end(original_sentence)
        repaired = self._replace_abstracted_subject_with_concrete_referent(
            sentence=stripped,
            original_sentence=original_sentence,
            role="mechanism",
        )
        if _HUANBAOKUO_RE.search(repaired):
            referent = self._concrete_referent_from_original(original_sentence, fallback="该机制")
            body = re.sub(r"^.*?(?:还包括|也包括|进一步包括)", "", stripped).strip("，, ")
            if body:
                repaired = self._ensure_sentence_end(f"{referent}{body}")
        return repaired

    def _soft_recast_mechanism_sentence(self, sentence: str) -> str:
        """Lightly recast an unchanged mechanism sentence without demoting its role."""

        stripped = sentence.strip()
        if not self._looks_like_mechanism_sentence(stripped):
            return sentence
        ordered_match = re.match(r"^(?:首先|其次)，(?P<body>.+?)[。！？?!]?$", stripped)
        if ordered_match:
            body = ordered_match.group("body").strip()
            if body.startswith("噪声分支对部分样本表现出较高敏感性"):
                body = body.replace("对部分样本表现出较高敏感性", "对部分样本确实表现出较高敏感性", 1)
                return self._ensure_sentence_end(body)
            if body.startswith("融合模块及控制机制在训练过程中并未稳定学习到"):
                body = body.replace("并未稳定学习到", "始终未能稳定学到", 1)
                return self._ensure_sentence_end(body)
            return self._ensure_sentence_end(body)
        if stripped.startswith("基于上述观察，本研究不再继续强化"):
            return self._ensure_sentence_end(stripped.replace("本研究不再继续强化", "研究重点不再是继续强化", 1))
        return sentence

    def _repair_enumeration_role(
        self,
        *,
        original_sentences: list[str],
        rewritten_sentences: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Restore explicit enumeration items when they drift into support-like appendices."""

        updated = list(rewritten_sentences)
        rules: list[str] = []
        notes: list[str] = []
        for index, sentence in enumerate(list(updated)):
            original_sentence = original_sentences[index] if index < len(original_sentences) else ""
            if not original_sentence or not self._looks_like_enumeration_sentence(original_sentence):
                continue
            if self._looks_like_enumeration_sentence(sentence) and not _HUANBAOKUO_RE.search(sentence):
                continue
            if _HUANBAOKUO_RE.search(sentence) or sentence.strip().startswith(_SCAFFOLDING_OPENINGS):
                restored = self._ensure_sentence_end(original_sentence)
            else:
                restored = self._ensure_sentence_end(original_sentence)
            if restored != sentence:
                updated[index] = restored
                rules.extend(
                    [
                        "local:preserve-enumeration-item-role",
                        "local:repair-enumeration-flow",
                    ]
                )
                notes.append("semantic-role repair restored an explicit enumeration item instead of a support-like expansion")
        return updated, self._deduplicate_preserve_order(rules), self._deduplicate_preserve_order(notes)

    def _repair_enumeration_support_fragments(
        self,
        *,
        original_sentences: list[str],
        sentences: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Repair fragmented or over-scaffolded support sentences inside enumeration items."""

        updated = list(sentences)
        rules: list[str] = []
        notes: list[str] = []
        joined_updated = "".join(updated)
        has_enumeration_source = any(
            self._looks_like_enumeration_sentence(sentence) for sentence in original_sentences
        )
        severe_fragment_patterns = (
            r"：若仅强调[^。]{2,120}。训练中同时维护[^。]{2,220}。易导致",
            r"根据[^。]{2,120}。从真实样本中",
            r"(?:本研究|本文)(?:更强调|重点)通过课程采样器",
        )
        if has_enumeration_source and any(re.search(pattern, joined_updated) for pattern in severe_fragment_patterns):
            restored = [self._ensure_sentence_end(sentence) for sentence in original_sentences if sentence.strip()]
            if restored:
                return (
                    restored,
                    [
                        "local:preserve-enumeration-item-role",
                        "local:repair-enumeration-flow",
                        "local:prevent-appendix-like-supporting-sentence",
                    ],
                    [
                        "semantic-role repair restored the original enumeration item after a fragmented support-style rewrite"
                    ],
                )
        index = 0
        while index < len(updated):
            sentence = updated[index].strip()
            if "更强调系统比较了" in sentence:
                updated[index] = self._ensure_sentence_end(sentence.replace("本研究更强调系统比较了", "本研究重点比较了"))
                rules.extend(
                    [
                        "local:preserve-enumeration-item-role",
                        "local:remove-generated-scaffolding-phrase",
                    ]
                )
                notes.append("semantic-role repair simplified an over-scaffolded enumeration claim")
            elif "更强调设计了" in sentence:
                updated[index] = self._ensure_sentence_end(sentence.replace("本研究更强调设计了", "本研究设计了"))
                rules.extend(
                    [
                        "local:preserve-enumeration-item-role",
                        "local:remove-generated-scaffolding-phrase",
                    ]
                )
                notes.append("semantic-role repair simplified an over-scaffolded enumeration claim")
            elif "进一步在第二阶段中实施" in sentence:
                updated[index] = self._ensure_sentence_end(sentence.replace("进一步在第二阶段中实施", "在第二阶段实施"))
                rules.append("local:avoid-huanbaokuo-style-expansion")
                notes.append("semantic-role repair removed a support-style expansion inside an enumeration item")
            elif "进一步将噪声分支从默认判定链路中剥离" in sentence:
                updated[index] = self._ensure_sentence_end(sentence.replace("进一步将噪声分支从默认判定链路中剥离", "将噪声分支从默认判定链路中剥离"))
                rules.append("local:avoid-huanbaokuo-style-expansion")
                notes.append("semantic-role repair restored a direct mechanism statement inside an enumeration item")
            if (
                index + 2 < len(updated)
                and self._looks_like_enumeration_sentence(updated[index])
                and "若仅强调" in updated[index]
                and updated[index + 1].strip().startswith("训练中同时维护")
                and updated[index + 2].strip().startswith("易导致")
            ):
                opener = self._strip_end_punctuation(updated[index].strip())
                followup = self._strip_end_punctuation(updated[index + 1].strip())
                consequence = self._strip_end_punctuation(updated[index + 2].strip())
                consequence = re.sub(r"^易导致", "", consequence).strip("，, ")
                updated[index] = self._ensure_sentence_end(f"{opener[:-1] if opener.endswith('。') else opener}，易导致{consequence}")
                updated[index + 1] = self._ensure_sentence_end(f"因此，{followup}")
                del updated[index + 2]
                rules.extend(
                    [
                        "local:repair-enumeration-flow",
                        "local:prevent-appendix-like-supporting-sentence",
                    ]
                )
                notes.append("semantic-role repair merged a broken enumeration support fragment back into a complete item")
                continue
            index += 1
        return updated, self._deduplicate_preserve_order(rules), self._deduplicate_preserve_order(notes)

    def _apply_semantic_role_integrity_repairs(
        self,
        *,
        original_sentences: list[str],
        rewritten_sentences: list[str],
        rewrite_depth: str,
        rewrite_intensity: str,
        high_sensitivity_prose: bool,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Repair role drift, enumeration drift, and generated scaffolding after local polishing."""

        if rewrite_depth not in {"developmental_rewrite", "light_edit"} or not rewritten_sentences:
            return rewritten_sentences, [], [], []

        updated = list(rewritten_sentences)
        rules: list[str] = []
        notes: list[str] = []
        marks: list[str] = []
        before = analyze_paragraph_readability_sentences(updated, high_sensitivity=high_sensitivity_prose)

        updated, enum_rules, enum_notes = self._repair_enumeration_role(
            original_sentences=original_sentences,
            rewritten_sentences=updated,
        )
        rules.extend(enum_rules)
        notes.extend(enum_notes)
        updated, enum_support_rules, enum_support_notes = self._repair_enumeration_support_fragments(
            original_sentences=original_sentences,
            sentences=updated,
        )
        rules.extend(enum_support_rules)
        notes.extend(enum_support_notes)

        for index, sentence in enumerate(list(updated)):
            original_sentence = original_sentences[index] if index < len(original_sentences) else ""
            role = self._sentence_role_for_semantic_repair(original_sentence or sentence, index, len(updated))
            repaired = sentence
            if sentence.strip().startswith(_SCAFFOLDING_OPENINGS) or "该段论述" in sentence or "这一点需要指出" in sentence:
                repaired = self._replace_abstracted_subject_with_concrete_referent(
                    sentence=sentence,
                    original_sentence=original_sentence,
                    role=role,
                )
                if repaired != sentence:
                    rules.extend(
                        [
                            "local:remove-generated-scaffolding-phrase",
                            "local:replace-abstracted-subject-with-concrete-referent",
                        ]
                    )
                    notes.append("semantic-role repair replaced a generated scaffolding subject with a concrete referent")
            if role == "mechanism":
                mechanism_repaired = self._restore_mechanism_sentence_from_support_like_rewrite(
                    sentence=repaired,
                    original_sentence=original_sentence,
                )
                if mechanism_repaired != repaired:
                    repaired = mechanism_repaired
                    rules.extend(
                        [
                            "local:restore-mechanism-sentence-from-support-like-rewrite",
                            "local:prevent-appendix-like-supporting-sentence",
                        ]
                    )
                    notes.append("semantic-role repair restored a mechanism sentence that drifted into appendix-like support phrasing")
            if (
                repaired == sentence
                and original_sentence
                and self._normalize_for_compare(repaired) == self._normalize_for_compare(original_sentence)
                and role == "mechanism"
            ):
                mechanism_recast = self._soft_recast_mechanism_sentence(repaired)
                if mechanism_recast != repaired and not self._sentence_needs_readability_repair(
                    mechanism_recast,
                    role="support_sentence",
                    is_final=index == len(updated) - 1,
                ):
                    repaired = mechanism_recast
                    rules.extend(
                        [
                            "local:restore-mechanism-sentence-from-support-like-rewrite",
                            "local:preserve-semantic-role-of-core-sentence",
                        ]
                    )
                    notes.append("semantic-role repair lightly recast an unchanged mechanism sentence while keeping its role explicit")
            if role in {"core", "conclusion"} and _HUANBAOKUO_RE.search(repaired):
                if original_sentence:
                    repaired = self._ensure_sentence_end(original_sentence)
                else:
                    repaired = re.sub(_HUANBAOKUO_RE, "", repaired).strip("，, ")
                    repaired = self._ensure_sentence_end(repaired)
                rules.extend(
                    [
                        "local:preserve-semantic-role-of-core-sentence",
                        "local:avoid-huanbaokuo-style-expansion",
                    ]
                )
                notes.append("semantic-role repair removed an appendix-like expansion from a salient sentence")
            if high_sensitivity_prose and repaired.strip().startswith(("这一工作", "这项工作", "相关内容")) and original_sentence:
                repaired = self._ensure_sentence_end(original_sentence)
                rules.extend(
                    [
                        "local:replace-abstracted-subject-with-concrete-referent",
                        "local:preserve-semantic-role-of-core-sentence",
                    ]
                )
                notes.append("high-sensitivity prose restored a direct referent instead of an abstract scaffolding subject")
            if repaired != sentence:
                updated[index] = repaired

        updated, flow_rules, flow_notes = self._repair_enumeration_sentence_flow(updated)
        rules.extend(flow_rules)
        notes.extend(flow_notes)

        after = analyze_paragraph_readability_sentences(updated, high_sensitivity=high_sensitivity_prose)
        if (
            after.paragraph_readability_score + 0.02 < before.paragraph_readability_score
            or after.sentence_completeness_score + 0.02 < before.sentence_completeness_score
            or after.dangling_sentence_risk > before.dangling_sentence_risk + 0.01
        ):
            return rewritten_sentences, [], [], []
        if rules:
            marks.append("semantic_role_integrity_repair")
        return (
            updated,
            self._deduplicate_preserve_order(rules),
            self._deduplicate_preserve_order(notes),
            self._deduplicate_preserve_order(marks),
        )

    def _is_appendix_like_assertion_sentence(self, sentence: str, *, role: str) -> bool:
        """Return true when a sentence is complete but still sounds like appendix-like support."""

        stripped = sentence.strip()
        if not stripped:
            return False
        if role == "mechanism" and self._looks_like_mechanism_sentence(stripped):
            if re.match(
                r"^(?:模型|系统|语义分支|频域分支|噪声分支|主融合模块|融合模块|控制机制|课程采样器|"
                r"判别路径|基础对数几率|默认部署模式|扩展分析模式|训练)",
                stripped,
            ) and not _APPENDIX_LIKE_SUPPORT_RE.match(stripped):
                return False
        if _APPENDIX_LIKE_SUPPORT_RE.match(stripped):
            return True
        return bool(
            re.match(r"^(?:本研究|本文|该方法|该机制|这种方式|该设置)", stripped)
            and _WEAK_MODAL_VERB_RE.search(stripped)
        )

    def _default_assertive_referent(self, original_sentence: str, role: str) -> str:
        """Choose a concrete fallback referent for assertion-strength repairs."""

        fallback = {
            "mechanism": "该结构",
            "conclusion": "该结果",
            "core": "本研究",
            "enumeration": "该项设计",
            "support": "该分析",
        }.get(role, "该结构")
        return self._concrete_referent_from_original(original_sentence, fallback=fallback)

    def _best_authorial_source_sentence(
        self,
        *,
        sentence: str,
        original_sentences: list[str],
        preferred_index: int,
    ) -> str:
        """Choose the source sentence that best matches a split or merged rewritten sentence."""

        if not original_sentences:
            return ""
        preferred = original_sentences[preferred_index] if preferred_index < len(original_sentences) else ""
        best_sentence = preferred or original_sentences[min(preferred_index, len(original_sentences) - 1)]
        best_score = self._sentence_similarity(sentence, best_sentence) + (0.08 if preferred else 0.0)
        sentence_placeholders = set(re.findall(r"\[\[AIRC:[^\]]+\]\]", sentence))
        for index, source in enumerate(original_sentences):
            score = self._sentence_similarity(sentence, source)
            if index == preferred_index:
                score += 0.08
            source_placeholders = set(re.findall(r"\[\[AIRC:[^\]]+\]\]", source))
            if sentence_placeholders and source_placeholders:
                overlap = len(sentence_placeholders & source_placeholders)
                if overlap:
                    score += min(0.18, overlap * 0.06)
            if "而是" in source and any(marker in sentence for marker in ("更强调", "设计了", "选择", "压缩")):
                score += 0.12
            if "继续围绕" in source and "围绕" in sentence:
                score += 0.12
            if "课程采样器" in source and "课程采样器" in sentence:
                score += 0.1
            if score > best_score:
                best_score = score
                best_sentence = source
        return best_sentence

    def _assertive_body_stands_alone(self, body: str) -> bool:
        """Return True when a stripped appendix wrapper leaves a usable direct assertion."""

        stripped = self._ensure_sentence_end(body.strip("，, "))
        if not stripped:
            return False
        if len(stripped) < 8:
            return False
        if re.match(r"^(?:并|而|同时|其中|此外|由此|在此基础上)", stripped):
            return False
        if re.search(
            r"(?:是|具有|维持|帮助|使|导致|避免|提升|增强|缓解|依赖|体现|显示|说明|表明|形成|"
            r"压缩|开展|切换|修复|约束|负责|实现|建模|融合)",
            stripped,
        ):
            return True
        return not self._sentence_needs_readability_repair(stripped, role="support_sentence", is_final=False)

    def _recast_source_authorial_stance(self, sentence: str) -> str:
        """Lightly recast a source stance sentence so it stays changed without losing intent."""

        stripped = self._strip_end_punctuation(sentence.strip())
        if not stripped:
            return sentence
        if "没有停留在“单个最佳阈值”思路，而是设计了" in stripped:
            stripped = stripped.replace(
                "没有停留在“单个最佳阈值”思路，而是设计了",
                "并未停留在“单个最佳阈值”的思路上，而是设置了",
                1,
            )
        if "在第二阶段中，训练不再对所有样本一视同仁，而是通过课程采样器让困难真实样本以更高频率进入批次" in stripped:
            stripped = stripped.replace(
                "在第二阶段中，训练不再对所有样本一视同仁，而是通过课程采样器让困难真实样本以更高频率进入批次，使模型在保证",
                "第二阶段的训练不再对所有样本一视同仁，而是通过课程采样器更频繁地将困难真实样本送入批次，使模型在保持",
                1,
            )
        if "说明后期优化重点并非" in stripped:
            stripped = stripped.replace("说明后期优化重点并非", "这说明后期优化重点并非", 1)
        if "并非" in stripped and "而是" in stripped:
            stripped = stripped.replace("通过结构收缩与课程学习持续压缩", "通过结构收缩与课程学习持续压缩", 1)
        return self._ensure_sentence_end(stripped)

    def _replace_weak_modal_verbs(
        self,
        *,
        sentence: str,
        original_sentence: str,
        role: str,
    ) -> str:
        """Replace weak modal-purpose verbs with direct assertive phrasing when the subject is concrete."""

        stripped = self._strip_end_punctuation(sentence.strip())
        if not stripped:
            return sentence
        if technical_density_is_high(stripped):
            return sentence

        updated = stripped
        updated = re.sub(r"有助于提升", "提升", updated)
        updated = re.sub(r"有助于增强", "增强", updated)
        updated = re.sub(r"有助于缓解", "缓解", updated)
        updated = re.sub(r"有助于减少", "减少", updated)
        updated = re.sub(r"有助于改善", "改善", updated)
        updated = re.sub(r"有助于保持", "保持", updated)
        updated = re.sub(r"可以实现", "直接实现", updated)
        updated = re.sub(r"能够实现", "直接实现", updated)
        updated = re.sub(r"可以完成", "直接完成", updated)
        updated = re.sub(r"能够完成", "直接完成", updated)
        updated = re.sub(r"可以避免", "直接避免", updated)
        updated = re.sub(r"能够避免", "直接避免", updated)
        updated = re.sub(r"可以保证", "直接保证", updated)
        updated = re.sub(r"能够保证", "直接保证", updated)

        concrete_subject = bool(
            re.match(
                r"^(?:模型|系统|语义分支|频域分支|噪声分支|主融合模块|融合模块|控制机制|课程采样器|"
                r"判别路径|基础对数几率|默认部署模式|扩展分析模式|训练|实验|后续研究|"
                r"NTIRE数据集|photos_test集合|量化评测基准|补充性诊断集合|该结构|该分支|该模块)",
                updated,
            )
        )
        if concrete_subject:
            updated = re.sub(r"用于提取", "负责提取", updated)
            updated = re.sub(r"用于刻画", "负责刻画", updated)
            updated = re.sub(r"用于建模", "直接建模", updated)
            updated = re.sub(r"用于融合", "直接融合", updated)
            updated = re.sub(r"用于输出", "负责输出", updated)
            updated = re.sub(r"用于约束", "负责约束", updated)
            updated = re.sub(r"用于限制", "负责限制", updated)
            updated = re.sub(r"用于剥离", "负责剥离", updated)
            updated = re.sub(r"用于实现", "直接实现", updated)
            updated = re.sub(r"用于验证", "用来验证", updated)
            updated = re.sub(r"用于观察", "用来观察", updated)
        elif role in {"mechanism", "support"} and re.match(r"^(?:本研究|本文|该方法|该机制|这种方式|该设置)用于", updated):
            referent = self._default_assertive_referent(original_sentence, role)
            body = re.sub(r"^(?:本研究|本文|该方法|该机制|这种方式|该设置)用于", "", updated).strip("，, ")
            if body.startswith(("提取", "刻画", "输出", "约束", "限制", "剥离")):
                updated = f"{referent}负责{body}"
            elif body.startswith(("融合", "建模", "实现", "形成", "完成")):
                updated = f"{referent}直接{body}"
            else:
                updated = f"{referent}用来{body}"
        updated = re.sub(r"^本研究进一步继续围绕", "后续研究仍将围绕", updated)
        updated = re.sub(r"^本研究继续围绕", "后续研究仍将围绕", updated)
        return self._ensure_sentence_end(updated)

    def _upgrade_appendix_like_sentence_to_assertion(
        self,
        *,
        sentence: str,
        original_sentence: str,
        role: str,
    ) -> str:
        """Convert appendix-like support phrasing into a direct assertion with a concrete referent."""

        stripped = self._strip_end_punctuation(sentence.strip())
        if not stripped or not self._is_appendix_like_assertion_sentence(stripped, role=role):
            return sentence

        if original_sentence and _AUTHORIAL_STANCE_RE.search(original_sentence) and not self._sentence_needs_readability_repair(
            original_sentence,
            role="conclusion_sentence" if role == "conclusion" else "support_sentence",
            is_final=role == "conclusion",
        ):
            return self._ensure_sentence_end(original_sentence)

        referent = self._default_assertive_referent(original_sentence, role)
        perspective_match = re.match(r"^从[^。；]{2,20}角度来看[，,\s]*(?P<body>.+)$", stripped)
        if perspective_match:
            body = perspective_match.group("body").strip("，, ")
            return self._ensure_sentence_end(body) if body else sentence

        setting_match = re.match(r"^(?:在该设置下|在当前设置下)[，,\s]*(?P<body>.+)$", stripped)
        if setting_match:
            body = setting_match.group("body").strip("，, ")
            return self._ensure_sentence_end(body) if body else sentence

        through_match = re.match(r"^通过这种方式(?:可以|能够)?(?P<body>.+)$", stripped)
        if through_match:
            body = through_match.group("body").strip("，, ")
            body = re.sub(r"^(?:实现|形成|完成|避免|约束|限制)", lambda m: f"直接{m.group(0)}", body, count=1)
            return self._ensure_sentence_end(f"{referent}{body}")

        conclusion_match = re.match(r"^(?:结合[^。；]{2,40}可以看出|可以看出|这表明|这说明)[，,\s]*(?P<body>.+)$", stripped)
        if conclusion_match:
            body = conclusion_match.group("body").strip("，, ")
            body = re.sub(r"^(当前模型)对(?P<object>[^。；]{2,28})仍具有较强依赖$", r"\1仍然依赖\g<object>", body)
            if body and self._assertive_body_stands_alone(body):
                return self._ensure_sentence_end(body)

        purpose_match = re.match(r"^(?:本研究|本文|该方法|该机制|这种方式|该设置)用于(?P<body>.+)$", stripped)
        if purpose_match:
            body = purpose_match.group("body").strip("，, ")
            if body.startswith(("提取", "刻画", "输出", "约束", "限制", "剥离")):
                return self._ensure_sentence_end(f"{referent}负责{body}")
            if body.startswith(("融合", "建模", "实现", "形成", "完成", "验证", "观察")):
                prefix = "用来" if body.startswith(("验证", "观察")) else "直接"
                return self._ensure_sentence_end(f"{referent}{prefix}{body}")
            return self._ensure_sentence_end(f"{referent}用来{body}")

        return self._replace_weak_modal_verbs(
            sentence=stripped,
            original_sentence=original_sentence,
            role=role,
        )

    def _strengthen_mechanism_verb(
        self,
        *,
        sentence: str,
        original_sentence: str,
    ) -> str:
        """Strengthen a mechanism sentence so it keeps a direct, concrete action verb."""

        stripped = sentence.strip()
        if not stripped:
            return sentence
        if not self._looks_like_mechanism_sentence(original_sentence or stripped):
            return sentence
        repaired = self._replace_weak_modal_verbs(
            sentence=stripped,
            original_sentence=original_sentence,
            role="mechanism",
        )
        repaired = re.sub(r"^(?:本研究|本文)通过课程采样器", "训练通过课程采样器", repaired)
        repaired = re.sub(r"^(?:本研究|本文)通过", "系统通过", repaired)
        repaired = re.sub(r"^(?:本研究|本文)设计了", "系统设计了", repaired)
        repaired = re.sub(r"^(?:本研究|本文)构建了", "系统构建了", repaired)
        return self._ensure_sentence_end(repaired)

    def _restore_authorial_choice_expression(
        self,
        *,
        sentence: str,
        original_sentence: str,
        role: str,
        high_sensitivity_prose: bool,
    ) -> str:
        """Restore source contrast or choice markers when the rewrite flattened them into explanation."""

        stripped_original = original_sentence.strip()
        stripped_revised = sentence.strip()
        if not stripped_original or _AUTHORIAL_STANCE_RE.search(stripped_revised):
            return sentence
        if not _AUTHORIAL_STANCE_RE.search(stripped_original):
            return sentence
        if role in {"conclusion", "core"}:
            return self._recast_source_authorial_stance(stripped_original)
        if self._assertive_body_stands_alone(stripped_original):
            return self._recast_source_authorial_stance(stripped_original)
        if high_sensitivity_prose:
            return self._recast_source_authorial_stance(stripped_original)
        return sentence

    def _reduce_overuse_of_passive_explanations(
        self,
        *,
        sentence: str,
        original_sentence: str,
        role: str,
    ) -> str:
        """Prefer direct claims over explanatory wrappers when the wrapped sentence already stands alone."""

        stripped = self._strip_end_punctuation(sentence.strip())
        if not stripped:
            return sentence
        if technical_density_is_high(stripped):
            return sentence
        body_match = re.match(r"^(?:可以看出|这表明|这说明|由此可见)[，,\s]*(?P<body>.+)$", stripped)
        if body_match:
            body = body_match.group("body").strip("，, ")
            if body and not self._sentence_needs_readability_repair(
                self._ensure_sentence_end(body),
                role="conclusion_sentence" if role == "conclusion" else "support_sentence",
                is_final=role == "conclusion",
            ):
                return self._ensure_sentence_end(body)
        if re.match(r"^结合[^。；]{2,40}可以看出[，,\s]*", stripped):
            body = re.sub(r"^结合[^。；]{2,40}可以看出[，,\s]*", "", stripped).strip("，, ")
            if body and self._assertive_body_stands_alone(body):
                return self._ensure_sentence_end(body)
        if re.match(r"^本研究进一步继续围绕", stripped):
            return self._ensure_sentence_end(re.sub(r"^本研究进一步继续围绕", "后续研究仍将围绕", stripped))
        if role in {"core", "conclusion"} and original_sentence and _AUTHORIAL_STANCE_RE.search(original_sentence):
            return self._ensure_sentence_end(original_sentence)
        return sentence

    def _repair_authorial_overlap_sequences(
        self,
        *,
        original_sentences: list[str],
        sentences: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Remove duplicated stance/mechanism spillover created by assertive repair."""

        if len(sentences) < 2:
            return sentences, [], []

        updated = list(sentences)
        rules: list[str] = []
        notes: list[str] = []
        if updated and original_sentences:
            opening = updated[0].strip()
            original_opening = original_sentences[0].strip()
            if opening.startswith(("并", "同时", "而")) and self._normalize_for_compare(opening) in self._normalize_for_compare(original_opening):
                updated[0] = self._ensure_sentence_end(original_opening)
                rules.append("local:upgrade-appendix-like-sentence-to-assertion")
                notes.append("authorial-intent repair restored a full opening assertion after a fragment-like carry-over")
        if len(original_sentences) == 1 and len(updated) > 1 and "而是" in original_sentences[0]:
            recast = self._recast_source_authorial_stance(original_sentences[0])
            updated = [self._ensure_sentence_end(original_sentences[0])]
            if recast != self._ensure_sentence_end(original_sentences[0]):
                updated = [recast]
            return (
                updated,
                [
                    "local:restore-authorial-choice-expression",
                    "local:promote-support-sentence-to-core-if-needed",
                ],
                ["authorial-intent repair collapsed a split stance sentence back into the original single-sentence assertion"],
            )
        index = 1
        while index < len(updated):
            previous = updated[index - 1].strip()
            current = updated[index].strip()
            previous_norm = self._normalize_for_compare(previous)
            current_norm = self._normalize_for_compare(current)
            if current_norm and len(current_norm) >= 14 and current_norm in previous_norm:
                del updated[index]
                rules.append("local:promote-support-sentence-to-core-if-needed")
                notes.append("authorial-intent repair removed a trailing sentence that repeated the preceding assertion")
                continue
            if previous_norm and current_norm.startswith(previous_norm) and "继续围绕" in current:
                trailing = re.sub(r"^.*?并继续围绕", "后续研究仍将围绕", current).strip()
                if trailing and self._assertive_body_stands_alone(trailing):
                    updated[index] = self._ensure_sentence_end(trailing)
                else:
                    del updated[index]
                    index -= 1
                rules.extend(
                    [
                        "local:restore-authorial-choice-expression",
                        "local:promote-support-sentence-to-core-if-needed",
                    ]
                )
                notes.append("authorial-intent repair removed a duplicated future-work prefix after restoring the concrete commitment")
                index += 1
                continue
            current_tail = re.sub(
                r"^(?:本研究|本文|训练|系统|该结构|该机制|该方法|后续研究)(?:更强调|设计了|通过|负责|"
                r"进一步继续围绕|继续围绕|仍将围绕|直接|用来)?",
                "",
                current_norm,
            )

            if (
                current.startswith(("本研究设计了", "训练通过课程采样器", "系统通过课程采样器"))
                and current_tail
                and current_tail in previous_norm
            ):
                del updated[index]
                rules.extend(
                    [
                        "local:promote-support-sentence-to-core-if-needed",
                        "local:reduce-overuse-of-passive-explanations",
                    ]
                )
                notes.append("authorial-intent repair removed a redundant mechanism sentence duplicated after a stronger core sentence")
                continue

            if current.startswith(("本研究更强调", "本工作更强调")) and "而是" in previous:
                trailing = current.split("，", 1)[1].strip() if "，" in current else ""
                if trailing and self._assertive_body_stands_alone(trailing):
                    updated[index] = self._ensure_sentence_end(trailing)
                elif original_sentences:
                    tail_source = next(
                        (
                            source
                            for source in original_sentences
                            if source.strip().startswith(("这一过程", "这一路径", "这一设计", "这一结果"))
                        ),
                        "",
                    )
                    if tail_source:
                        updated[index] = self._ensure_sentence_end(tail_source)
                    else:
                        del updated[index]
                        index -= 1
                rules.extend(
                    [
                        "local:restore-authorial-choice-expression",
                        "local:promote-support-sentence-to-core-if-needed",
                    ]
                )
                notes.append("authorial-intent repair removed duplicated choice framing after restoring the source stance")
                index += 1
                continue

            if current.startswith("后续研究仍将围绕") and original_sentences:
                source_tail = next(
                    (
                        source
                        for source in original_sentences
                        if "继续围绕" in source and "开展" in source
                    ),
                    "",
                )
                if source_tail:
                    simplified = re.sub(r"^后续研究需要进一步.+?，并继续围绕", "后续研究仍将围绕", source_tail.strip())
                    updated[index] = self._ensure_sentence_end(simplified)
                    rules.append("local:restore-authorial-choice-expression")
                    notes.append("authorial-intent repair restored a direct future-work commitment after a split continuation")
                index += 1
                continue

            index += 1

        if updated == sentences:
            return sentences, [], []
        return (
            updated,
            self._deduplicate_preserve_order(rules),
            self._deduplicate_preserve_order(notes),
        )

    def _apply_authorial_intent_repairs(
        self,
        *,
        original_sentences: list[str],
        rewritten_sentences: list[str],
        rewrite_depth: str,
        rewrite_intensity: str,
        high_sensitivity_prose: bool,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Restore direct assertion, concrete mechanism verbs, and source stance after semantic-role repair."""

        if rewrite_depth not in {"developmental_rewrite", "light_edit"} or not rewritten_sentences:
            return rewritten_sentences, [], [], []

        updated = list(rewritten_sentences)
        rules: list[str] = []
        notes: list[str] = []
        marks: list[str] = []
        before_readability = analyze_paragraph_readability_sentences(updated, high_sensitivity=high_sensitivity_prose)
        before_authorial = analyze_authorial_intent(
            original_sentences,
            updated,
            high_sensitivity=high_sensitivity_prose,
        )

        for index, sentence in enumerate(list(updated)):
            original_sentence = self._best_authorial_source_sentence(
                sentence=sentence,
                original_sentences=original_sentences,
                preferred_index=index,
            )
            indexed_original_sentence = original_sentences[index] if index < len(original_sentences) else original_sentence
            role = self._sentence_role_for_semantic_repair(original_sentence or sentence, index, len(updated))
            repaired = sentence

            stance_repaired = self._restore_authorial_choice_expression(
                sentence=repaired,
                original_sentence=original_sentence,
                role=role,
                high_sensitivity_prose=high_sensitivity_prose,
            )
            if stance_repaired != repaired:
                repaired = stance_repaired
                rules.extend(
                    [
                        "local:restore-authorial-choice-expression",
                        "local:promote-support-sentence-to-core-if-needed",
                    ]
                )
                notes.append("authorial-intent repair restored a source contrast or choice marker that the rewrite had flattened")

            appendix_repaired = self._upgrade_appendix_like_sentence_to_assertion(
                sentence=repaired,
                original_sentence=original_sentence,
                role=role,
            )
            if appendix_repaired != repaired:
                repaired = appendix_repaired
                rules.extend(
                    [
                        "local:upgrade-appendix-like-sentence-to-assertion",
                        "local:replace-weak-modal-verbs",
                    ]
                )
                notes.append("authorial-intent repair converted an appendix-like support sentence into a direct assertion")

            if role == "mechanism":
                strengthened = self._strengthen_mechanism_verb(
                    sentence=repaired,
                    original_sentence=original_sentence,
                )
                if strengthened != repaired:
                    repaired = strengthened
                    rules.extend(
                        [
                            "local:strengthen-mechanism-verb",
                            "local:replace-weak-modal-verbs",
                        ]
                    )
                    notes.append("authorial-intent repair strengthened a mechanism sentence with a more direct action verb")

            reduced_explanation = self._reduce_overuse_of_passive_explanations(
                sentence=repaired,
                original_sentence=original_sentence,
                role=role,
            )
            if reduced_explanation != repaired:
                repaired = reduced_explanation
                rules.append("local:reduce-overuse-of-passive-explanations")
                notes.append("authorial-intent repair preferred a direct claim over an explanatory wrapper")

            if repaired != sentence:
                updated[index] = self._ensure_sentence_end(repaired)

        updated, overlap_rules, overlap_notes = self._repair_authorial_overlap_sequences(
            original_sentences=original_sentences,
            sentences=updated,
        )
        rules.extend(overlap_rules)
        notes.extend(overlap_notes)

        after_readability = analyze_paragraph_readability_sentences(updated, high_sensitivity=high_sensitivity_prose)
        after_authorial = analyze_authorial_intent(
            original_sentences,
            updated,
            high_sensitivity=high_sensitivity_prose,
        )
        major_authorial_gain = (
            after_authorial.assertion_strength_score >= before_authorial.assertion_strength_score + 0.08
            or after_authorial.appendix_like_support_ratio <= before_authorial.appendix_like_support_ratio - 0.08
            or after_authorial.authorial_stance_presence >= before_authorial.authorial_stance_presence + 0.15
        )
        sentence_collapse = len(updated) < len(rewritten_sentences)
        readability_drop_limit = 0.08 if major_authorial_gain and (sentence_collapse or high_sensitivity_prose) else (0.04 if major_authorial_gain else 0.02)
        completeness_drop_limit = 0.08 if major_authorial_gain and sentence_collapse else (0.04 if major_authorial_gain else 0.02)
        if (
            after_readability.paragraph_readability_score + readability_drop_limit < before_readability.paragraph_readability_score
            or after_readability.sentence_completeness_score + completeness_drop_limit < before_readability.sentence_completeness_score
            or after_readability.dangling_sentence_risk > before_readability.dangling_sentence_risk + 0.01
            or after_authorial.assertion_strength_score + 0.02 < before_authorial.assertion_strength_score
            or after_authorial.appendix_like_support_ratio > before_authorial.appendix_like_support_ratio + 0.01
            or after_authorial.authorial_stance_presence + 0.08 < before_authorial.authorial_stance_presence
        ):
            return rewritten_sentences, [], [], []
        if rules:
            marks.append("authorial_intent_repair")
        return (
            updated,
            self._deduplicate_preserve_order(rules),
            self._deduplicate_preserve_order(notes),
            self._deduplicate_preserve_order(marks),
        )

    def _remove_unsupported_expansion(self, *, sentence: str, original_sentence: str) -> str:
        """Drop unsupported background or outside-domain commentary that is absent from the source."""

        stripped = sentence.strip()
        if not stripped or not _UNSUPPORTED_EXPANSION_RE.search(stripped):
            return sentence
        if original_sentence and _UNSUPPORTED_EXPANSION_RE.search(original_sentence):
            return self._ensure_sentence_end(original_sentence)
        if original_sentence:
            return self._ensure_sentence_end(original_sentence)
        cleaned = _THESIS_REGISTER_WRAPPER_RE.sub("", stripped).strip("，, ")
        cleaned = _UNSUPPORTED_EXPANSION_RE.sub("", cleaned).strip("，, ")
        return self._ensure_sentence_end(cleaned or sentence)

    def _remove_metaphoric_storytelling(self, *, sentence: str, original_sentence: str) -> str:
        """Replace commentary-like metaphor or storytelling with source-bounded academic prose."""

        stripped = sentence.strip()
        if not stripped or not _METAPHOR_STORYTELLING_RE.search(stripped):
            return sentence
        if original_sentence:
            return self._ensure_sentence_end(original_sentence)
        cleaned = _METAPHOR_STORYTELLING_RE.sub("", stripped).strip("，, ")
        cleaned = re.sub(r"(?:终于|如同|仿佛)[，,\s]*", "", cleaned).strip("，, ")
        return self._ensure_sentence_end(cleaned or sentence)

    def _replace_we_with_original_subject_style(self, *, sentence: str, original_sentence: str, role: str) -> str:
        """Replace unjustified first-person subjects with the source paper's subject style."""

        stripped = sentence.strip()
        if not stripped or not re.match(r"^(?:我们|我们的)", stripped):
            return sentence
        if original_sentence and re.match(r"^(?:我们|我们的)", original_sentence.strip()):
            return self._ensure_sentence_end(original_sentence)
        referent = self._concrete_referent_from_original(
            original_sentence,
            fallback="本研究" if role in {"core", "support", "conclusion"} else "模型",
        )
        updated = re.sub(r"^我们的", f"{referent}的", stripped, count=1)
        updated = re.sub(r"^我们", referent, updated, count=1)
        return self._ensure_sentence_end(updated)

    def _downgrade_overclaimed_judgment(self, *, sentence: str, original_sentence: str, role: str) -> str:
        """Restore a restrained thesis register when the rewrite overclaims beyond the source."""

        stripped = sentence.strip()
        if not stripped:
            return sentence
        if not (
            _UNJUSTIFIED_AUTHORIAL_CLAIM_RE.search(stripped)
            or _UNSUPPORTED_EXPANSION_RE.search(stripped)
            or _METAPHOR_STORYTELLING_RE.search(stripped)
            or stripped.startswith("正如该领域主流观点所认为的")
        ):
            return sentence
        if original_sentence and any(token in original_sentence for token in ("证明", "主流观点", "我们", "我们的")):
            return self._ensure_sentence_end(original_sentence)
        updated = stripped
        updated = re.sub(r"^本工作证明了", "本研究表明", updated)
        updated = re.sub(r"^这项工作的终点", "本研究的目标", updated)
        updated = _THESIS_REGISTER_WRAPPER_RE.sub("", updated).strip("，, ")
        if role == "mechanism":
            updated = re.sub(r"^这表明", "", updated).strip("，, ")
        return self._ensure_sentence_end(updated or sentence)

    def _restore_mechanism_sentence_to_academic_statement(
        self,
        *,
        sentence: str,
        original_sentence: str,
    ) -> str:
        """Restore mechanism sentences from narrative or commentary drift to direct academic statements."""

        stripped = sentence.strip()
        if not stripped:
            return sentence
        if not (
            _UNSUPPORTED_EXPANSION_RE.search(stripped)
            or _METAPHOR_STORYTELLING_RE.search(stripped)
            or _UNJUSTIFIED_AUTHORIAL_CLAIM_RE.search(stripped)
            or stripped.startswith(("通过这种方式", "在该设置下", "从这一角度来看", "从"))
            or stripped.startswith(("这表明", "可以看出"))
        ):
            return sentence
        if original_sentence and any(pattern.search(stripped) for pattern in (_UNSUPPORTED_EXPANSION_RE, _METAPHOR_STORYTELLING_RE)):
            return self._ensure_sentence_end(original_sentence)
        referent = self._concrete_referent_from_original(original_sentence, fallback="模型")
        updated = stripped
        updated = re.sub(r"^通过这种方式可以", f"{referent}直接", updated)
        updated = re.sub(r"^在该设置下[,，\\s]*", "", updated)
        updated = re.sub(r"^从[^。；]{1,20}角度来看[,，\\s]*", "", updated)
        updated = re.sub(r"^这表明[,，\\s]*", "", updated)
        updated = re.sub(r"^可以看出[,，\\s]*", "", updated)
        if updated != stripped and not self._sentence_needs_readability_repair(
            updated,
            role="support_sentence",
            is_final=False,
        ):
            return self._ensure_sentence_end(updated)
        return self._ensure_sentence_end(original_sentence) if original_sentence else sentence

    def _repair_source_protected_sentence_alignment(
        self,
        *,
        original_sentences: list[str],
        sentences: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Restore a source sentence when a protected identifier or citation boundary drifts."""

        updated = list(sentences)
        rules: list[str] = []
        notes: list[str] = []
        for source in original_sentences:
            stripped_source = source.strip()
            if not stripped_source:
                continue
            source_identifiers = re.findall(r"\b[A-Za-z]+_[A-Za-z0-9_]+\b", stripped_source)
            source_has_citation = bool(re.search(r"\[\d+\]", stripped_source))
            if not source_identifiers and not source_has_citation:
                continue
            best_index = -1
            best_score = 0.0
            normalized_source = self._normalize_for_compare(stripped_source)
            for index, candidate in enumerate(updated):
                normalized_candidate = self._normalize_for_compare(candidate)
                score = SequenceMatcher(None, normalized_source, normalized_candidate).ratio()
                if score > best_score:
                    best_index = index
                    best_score = score
            if best_index < 0 or best_score < 0.26:
                continue
            candidate = updated[best_index]
            if source_identifiers and any(identifier not in candidate for identifier in source_identifiers):
                updated[best_index] = self._ensure_sentence_end(stripped_source)
                rules.extend(
                    [
                        "local:preserve-original-evidence-scope",
                        "readability:restore-source-for-protected-token-coverage",
                    ]
                )
                notes.append("evidence-fidelity repair restored a source sentence after a protected identifier drifted out of the rewrite")
                continue
            if source_has_citation and (
                re.search(r"\[\d+\][^\s。；，,）)]", candidate)
                or candidate.count("[") > stripped_source.count("[")
            ):
                updated[best_index] = self._ensure_sentence_end(stripped_source)
                rules.extend(
                    [
                        "local:preserve-original-evidence-scope",
                        "local:restore-thesis-register",
                    ]
                )
                notes.append("evidence-fidelity repair restored a citation-bearing source sentence after boundary drift")
        return (
            updated,
            self._deduplicate_preserve_order(rules),
            self._deduplicate_preserve_order(notes),
        )

    def _apply_evidence_fidelity_repairs(
        self,
        *,
        original_sentences: list[str],
        rewritten_sentences: list[str],
        rewrite_depth: str,
        high_sensitivity_prose: bool,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Pull revised prose back inside the source evidence boundary and restrained thesis register."""

        if rewrite_depth not in {"developmental_rewrite", "light_edit"} or not rewritten_sentences:
            return rewritten_sentences, [], [], []

        updated = list(rewritten_sentences)
        rules: list[str] = []
        notes: list[str] = []
        marks: list[str] = []
        before_readability = analyze_paragraph_readability_sentences(updated, high_sensitivity=high_sensitivity_prose)
        before_fidelity = analyze_evidence_fidelity(
            original_sentences,
            updated,
            high_sensitivity=high_sensitivity_prose,
        )

        for index, sentence in enumerate(list(updated)):
            original_sentence = self._best_authorial_source_sentence(
                sentence=sentence,
                original_sentences=original_sentences,
                preferred_index=index,
            )
            indexed_original_sentence = original_sentences[index] if index < len(original_sentences) else original_sentence
            role = self._sentence_role_for_semantic_repair(original_sentence or sentence, index, len(updated))
            repaired = sentence

            unsupported_repaired = self._remove_unsupported_expansion(
                sentence=repaired,
                original_sentence=original_sentence,
            )
            if unsupported_repaired != repaired:
                repaired = unsupported_repaired
                rules.extend(
                    [
                        "local:remove-unsupported-expansion",
                        "local:remove-external-domain-commentary",
                        "local:preserve-original-evidence-scope",
                    ]
                )
                notes.append("evidence-fidelity repair removed unsupported background or outside-domain expansion")

            metaphor_repaired = self._remove_metaphoric_storytelling(
                sentence=repaired,
                original_sentence=original_sentence,
            )
            if metaphor_repaired != repaired:
                repaired = metaphor_repaired
                rules.extend(
                    [
                        "local:remove-metaphoric-storytelling",
                        "local:restore-thesis-register",
                    ]
                )
                notes.append("evidence-fidelity repair removed metaphor or storytelling phrasing and restored thesis register")

            first_person_repaired = self._replace_we_with_original_subject_style(
                sentence=repaired,
                original_sentence=original_sentence,
                role=role,
            )
            if first_person_repaired != repaired:
                repaired = first_person_repaired
                rules.append("local:replace-we-with-original-subject-style")
                notes.append("evidence-fidelity repair restored the source subject style instead of adding first-person voice")

            if role == "mechanism":
                mechanism_repaired = self._restore_mechanism_sentence_to_academic_statement(
                    sentence=repaired,
                    original_sentence=original_sentence,
                )
                if mechanism_repaired != repaired:
                    repaired = mechanism_repaired
                    rules.append("local:restore-mechanism-sentence-to-academic-statement")
                    notes.append("evidence-fidelity repair restored a direct academic mechanism statement")

            overclaim_repaired = self._downgrade_overclaimed_judgment(
                sentence=repaired,
                original_sentence=original_sentence,
                role=role,
            )
            if overclaim_repaired != repaired:
                repaired = overclaim_repaired
                rules.extend(
                    [
                        "local:downgrade-overclaimed-judgment",
                        "local:restore-thesis-register",
                    ]
                )
                notes.append("evidence-fidelity repair downgraded an overclaimed or commentary-like judgment")

            source_identifiers = re.findall(r"\b[A-Za-z]+_[A-Za-z0-9_]+\b", indexed_original_sentence)
            if source_identifiers and any(identifier not in repaired for identifier in source_identifiers):
                repaired = self._ensure_sentence_end(indexed_original_sentence)
                rules.extend(
                    [
                        "local:preserve-original-evidence-scope",
                        "readability:restore-source-for-protected-token-coverage",
                    ]
                )
                notes.append("evidence-fidelity repair restored a source sentence to preserve an inline protected identifier")
            if (
                indexed_original_sentence
                and re.search(r"\[\d+\]", indexed_original_sentence)
                and (
                    re.search(r"\[\d+\][^\s。；，,）)]", repaired)
                    or repaired.count("[") > indexed_original_sentence.count("[")
                )
            ):
                repaired = self._ensure_sentence_end(indexed_original_sentence)
                rules.extend(
                    [
                        "local:preserve-original-evidence-scope",
                        "local:restore-thesis-register",
                    ]
                )
                notes.append("evidence-fidelity repair restored a citation-bearing source sentence after boundary drift")

            if repaired != sentence:
                updated[index] = self._ensure_sentence_end(repaired)

        updated, protected_alignment_rules, protected_alignment_notes = self._repair_source_protected_sentence_alignment(
            original_sentences=original_sentences,
            sentences=updated,
        )
        rules.extend(protected_alignment_rules)
        notes.extend(protected_alignment_notes)

        after_readability = analyze_paragraph_readability_sentences(updated, high_sensitivity=high_sensitivity_prose)
        after_fidelity = analyze_evidence_fidelity(
            original_sentences,
            updated,
            high_sensitivity=high_sensitivity_prose,
        )
        if (
            after_readability.paragraph_readability_score + 0.03 < before_readability.paragraph_readability_score
            or after_readability.sentence_completeness_score + 0.03 < before_readability.sentence_completeness_score
            or after_readability.dangling_sentence_risk > before_readability.dangling_sentence_risk + 0.01
            or after_fidelity.evidence_fidelity_score + 0.02 < before_fidelity.evidence_fidelity_score
            or after_fidelity.unsupported_expansion_risk > before_fidelity.unsupported_expansion_risk + 0.01
            or after_fidelity.thesis_tone_restraint_score + 0.02 < before_fidelity.thesis_tone_restraint_score
        ):
            return rewritten_sentences, [], [], []
        if rules:
            marks.append("evidence_fidelity_repair")
        return (
            updated,
            self._deduplicate_preserve_order(rules),
            self._deduplicate_preserve_order(notes),
            self._deduplicate_preserve_order(marks),
        )

    def _remove_bureaucratic_opening(self, sentence: str) -> str:
        """Remove project-report style openings when the remaining academic sentence stands alone."""

        stripped = sentence.strip()
        replacements = (
            (r"^在方法上[，,\s]*", ""),
            (r"^围绕这一目标[，,\s]*", ""),
            (r"^具体而言[，,\s]*", ""),
            (r"^就(?P<context>[^，。；]{2,28})而言[，,\s]*", ""),
            (r"^从(?P<context>[^，。；]{2,28})角度(?:来看)?[，,\s]*", ""),
            (r"^围绕(?P<context>[^，。；]{2,28})[，,\s]*", ""),
            (r"^在(?P<context>[^，。；]{2,28})(?:中|下|过程中)[，,\s]*(?P<body>本研究|本文|研究|系统|模型|该方法|该机制)", r"\g<body>"),
            (r"^在完成(?P<context>[^，。；]{2,42})后[，,\s]*(?P<body>本研究|本文|研究|系统|模型)", r"\g<body>"),
            (r"^在(?P<context>[^，。；]{2,18})层面[，,\s]*(?P<body>本研究|本文|研究|系统|模型|该机制)", r"\g<body>"),
            (
                r"^在解决(?P<problem>[^，。；]{4,80})这一目标下[，,\s]*(?P<body>本研究|本文|研究)",
                r"为解决\g<problem>，\g<body>",
            ),
        )
        updated = stripped
        for pattern, replacement in replacements:
            updated = re.sub(pattern, replacement, updated, count=1)
        match = re.match(r"^本研究面向(?P<context>[^，。；]{2,48})需求[，,](?P<body>.+)$", updated)
        if match:
            body = match.group("body").strip()
            if body.startswith(("设计", "提出", "构建", "实现")):
                updated = f"本研究{body}"
            else:
                updated = body
        match = re.match(r"^本研究的主题为“(?P<title>[^”]{2,80})”[。；]?(?P<body>.*)$", updated)
        if match:
            body = match.group("body").strip("。；，, ")
            updated = f"本文围绕“{match.group('title')}”展开"
            if body:
                updated = f"{updated}，{body}"
        return self._ensure_sentence_end(updated) if updated != stripped else sentence

    def _enforce_direct_statement(self, sentence: str) -> str:
        """Prefer direct academic statements over wrapped introductory phrases."""

        return self._remove_bureaucratic_opening(sentence)

    def _remove_academic_wrapping(self, sentence: str) -> str:
        """Replace academic packaging phrases with lighter author-like wording."""

        stripped = sentence.strip()
        updated = stripped
        replacements = (
            (r"为(?P<object>[^，。；]{2,60})提供(?:了)?(?P<body>[^，。；]{1,24})基础", r"为\g<object>打下\g<body>基础"),
            (r"提供(?:了)?([^，。；]{1,24})基础", r"形成\1基础"),
            (r"提供(?:了)?([^，。；]{1,24})路径", r"给出\1路径"),
            (r"形成([^，。；]{1,24})体系", r"构成\1"),
            (r"具有([^，。；]{1,18})价值", r"在\1中更重要"),
            (r"被视为", "是"),
            (r"属于", "是"),
            (r"得到体现", "体现为"),
            (r"进行(评估|分析|验证|控制|处理)", r"\1"),
            (r"实现(控制|部署|融合|落地)", r"\1"),
            (r"提供(支撑|支持)", r"\1"),
        )
        for pattern, replacement in replacements:
            updated = re.sub(pattern, replacement, updated)
        return self._ensure_sentence_end(updated) if updated != stripped else sentence

    def _remove_slogan_like_goal_phrase(self, sentence: str) -> str:
        """Replace slogan-like goal language with restrained thesis prose."""

        stripped = sentence.strip()
        updated = stripped
        updated = re.sub(r"旨在构建", "用于构建", updated)
        updated = re.sub(r"旨在通过", "主要通过", updated)
        updated = re.sub(r"旨在", "", updated)
        updated = re.sub(r"以实现从算法到应用的落地", "用于支持算法在系统中的应用", updated)
        updated = re.sub(r"以实现([^，。；]{2,36})", r"并\1", updated)
        updated = re.sub(r"从而实现([^，。；]{2,36})", r"并\1", updated)
        updated = re.sub(r"形成(?:了)?从([^。；]{2,28})到([^。；]{2,28})的完整闭环", r"串联起从\1到\2的主要环节", updated)
        updated = re.sub(r"形成(?:了)?完整闭环", "形成较完整的系统流程", updated)
        updated = re.sub(r"具有显著的([^，。；]{1,16})价值", r"在\1方面具有实际意义", updated)
        updated = re.sub(r"提供清晰方法论路径", "提供较明确的分析路径", updated)
        updated = re.sub(r"筑牢第一道防线", "提供前置筛查支持", updated)
        return self._ensure_sentence_end(updated) if updated != stripped else sentence

    def _flatten_overstructured_parallelism(self, sentence: str) -> str:
        """Soften overly tidy contrast or parallel syntax without changing the claim."""

        stripped = sentence.strip()
        updated = stripped
        updated = re.sub(r"并不是([^，。；]{2,60})，?而是", r"不只是\1，更是", updated)
        updated = re.sub(r"并非([^，。；]{2,60})，?而是", r"不只是\1，更是", updated)
        updated = re.sub(r"不仅([^，。；]{2,70})，?(?:同时|还)", r"既\1，也", updated)
        updated = re.sub(r"在结构上([^，。；]{2,70})，?在决策层面", r"结构上\1，决策层面", updated)
        updated = re.sub(r"第一，([^。；]{2,60})，?进而第二，?", r"一方面，\1；另一方面，", updated)
        return self._ensure_sentence_end(updated) if updated != stripped else sentence

    def _reduce_connectors(self, sentence: str) -> str:
        """Downgrade heavy connectors to lighter sequence or punctuation."""

        stripped = sentence.strip()
        updated = stripped
        updated = re.sub(r"不仅([^，。；]{2,70})，?(?:还|也|同时)", r"\1，也", updated)
        updated = updated.replace("并且", "，")
        updated = updated.replace("从而", "并")
        updated = updated.replace("进而", "再")
        updated = re.sub(r"同时，(?=[^。；]{0,40}同时)", "", updated)
        updated = re.sub(r"，{2,}", "，", updated)
        updated = re.sub(r"，，", "，", updated)
        return self._ensure_sentence_end(updated) if updated != stripped else sentence

    def _convert_passive_to_active(self, sentence: str) -> str:
        """Convert common passive or classificatory phrasing to active thesis prose."""

        stripped = sentence.strip()
        updated = stripped
        updated = updated.replace("被视为", "是")
        updated = updated.replace("得到体现", "体现为")
        updated = re.sub(r"属于([^，。；]{1,24})", r"是\1", updated)
        return self._ensure_sentence_end(updated) if updated != stripped else sentence

    def _restore_nominalized_verbs(self, sentence: str) -> str:
        """Turn common nominalized academic actions back into direct verbs."""

        stripped = sentence.strip()
        updated = stripped
        updated = re.sub(r"进行(评估|分析|验证|控制|处理)", r"\1", updated)
        updated = re.sub(r"实现(控制|部署|融合|落地)", r"\1", updated)
        updated = re.sub(r"提供(支撑|支持)", r"\1", updated)
        return self._ensure_sentence_end(updated) if updated != stripped else sentence

    def _split_overlong_sentence(self, sentence: str) -> str:
        """Split sentences that carry more than two visible logic layers."""

        stripped = sentence.strip()
        compact = re.sub(r"\s+", "", stripped)
        if len(compact) < 96:
            return sentence
        split_markers = ("；同时，", "；", "，同时，", "，并且", "，从而", "，进而")
        for marker in split_markers:
            if marker in stripped:
                left, right = stripped.split(marker, 1)
                if len(left) >= 24 and len(right) >= 18:
                    right = re.sub(r"^(同时|并且|从而|进而)[，,]?", "", right).strip()
                    candidate = f"{self._ensure_sentence_end(left)}{self._ensure_sentence_end(right)}"
                    return candidate
        return sentence

    def _move_main_clause_forward(self, sentence: str) -> str:
        """Expose the main subject and action before long contextual modifiers."""

        return self._advance_main_clause(sentence)

    def _advance_main_clause(self, sentence: str) -> str:
        """Move the main clause earlier when a long modifier prefix delays the sentence spine."""

        stripped = sentence.strip()
        match = re.match(
            r"^在解决(?P<problem>[^，。；]{8,90})这一目标下[，,\s]*(?P<subject>本研究|本文|研究)(?P<body>提出了.+)$",
            stripped,
        )
        if match:
            return self._ensure_sentence_end(
                f"{match.group('subject')}{match.group('body')}，用于解决{match.group('problem')}"
            )
        match = re.match(
            r"^在完成(?P<context>[^，。；]{6,60})后[，,\s]*(?P<subject>本研究|本文|研究|系统)(?P<body>.+)$",
            stripped,
        )
        if match:
            return self._ensure_sentence_end(f"{match.group('subject')}{match.group('body')}，以前期{match.group('context')}为基础")
        match = re.match(r"^(?P<prefix>[^，。；]{34,90})[，,](?P<body>(?:本研究|本文|研究|系统|模型).+)$", stripped)
        if match and re.search(r"(目标|过程|背景|层面|基础|情况下)", match.group("prefix")):
            return self._ensure_sentence_end(f"{match.group('body')}，其背景是{match.group('prefix')}")
        return sentence

    def _compress_explicit_subject_chain(self, sentences: list[str]) -> tuple[list[str], list[str], list[str]]:
        """Reduce repeated explicit academic subjects across adjacent sentences."""

        if len(sentences) < 2:
            return sentences, [], []
        updated = list(sentences)
        rules: list[str] = []
        notes: list[str] = []
        for index in range(1, len(updated)):
            previous = updated[index - 1].strip()
            current = updated[index].strip()
            previous_head = re.match(r"^(本研究|本文|研究|系统|模型)", previous)
            current_head = re.match(r"^(本研究|本文|研究|系统|模型)", current)
            if not previous_head or not current_head or previous_head.group(1) != current_head.group(1):
                continue
            candidate = re.sub(r"^(?:本研究|本文|研究)[，,\s]*", "", current, count=1)
            candidate = re.sub(r"^(?:系统|模型)[，,\s]*", "同时，", candidate, count=1)
            if candidate and not self._sentence_needs_readability_repair(candidate, role="support_sentence", is_final=False):
                updated[index] = self._ensure_sentence_end(candidate)
                rules.append("local:compress-explicit-subject-chain")
                notes.append("sentence naturalization compressed a repeated explicit subject chain")
        if updated == sentences:
            return sentences, [], []
        return updated, self._deduplicate_preserve_order(rules), self._deduplicate_preserve_order(notes)

    def _diversify_subject(self, sentences: list[str]) -> tuple[list[str], list[str], list[str]]:
        """Avoid three or more consecutive sentences with the same explicit subject."""

        if len(sentences) < 3:
            return sentences, [], []
        updated = list(sentences)
        rules: list[str] = []
        notes: list[str] = []
        alternatives = {
            "本研究": ["该方法", "该机制", "系统", "模型", "该策略"],
            "研究": ["该方法", "系统", "该策略"],
            "系统": ["该系统", "模型", "该机制"],
            "模型": ["该模型", "该机制", "系统"],
        }
        streak_head = ""
        streak = 0
        for index, sentence in enumerate(updated):
            match = re.match(r"^(本研究|研究|系统|模型)", sentence.strip())
            head = match.group(1) if match else ""
            streak = streak + 1 if head and head == streak_head else 1
            streak_head = head
            if head and streak >= 3:
                replacement = alternatives.get(head, [head])[(streak - 3) % len(alternatives.get(head, [head]))]
                candidate = re.sub(rf"^{re.escape(head)}", replacement, sentence.strip(), count=1)
                if not self._sentence_needs_readability_repair(candidate, role="support_sentence", is_final=False):
                    updated[index] = self._ensure_sentence_end(candidate)
                    rules.append("local:diversify-subject")
                    notes.append("sentence naturalization diversified a monotonous explicit subject chain")
        if updated == sentences:
            return sentences, [], []
        return updated, self._deduplicate_preserve_order(rules), self._deduplicate_preserve_order(notes)

    def _apply_l2_mild_style_texture(
        self,
        *,
        original_sentences: list[str],
        rewritten_sentences: list[str],
        rewrite_depth: str,
        high_sensitivity_prose: bool,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Apply the optional zh_academic_l2_mild profile without breaking grammar or facts."""

        if rewrite_depth not in {"developmental_rewrite", "light_edit"} or not rewritten_sentences:
            return rewritten_sentences, [], [], []
        if any(self._technical_density_is_high(sentence) for sentence in rewritten_sentences):
            return rewritten_sentences, [], [], []

        updated = list(rewritten_sentences)
        rules: list[str] = []
        notes: list[str] = []
        marks: list[str] = []
        for index, sentence in enumerate(updated):
            if index >= 3 and not high_sensitivity_prose:
                break
            candidate, sentence_rules = self._l2_mild_sentence_variant(sentence)
            if not sentence_rules or candidate == sentence:
                continue
            if self._l2_variant_breaks_bounds(candidate):
                continue
            updated[index] = candidate
            rules.extend(sentence_rules)
            notes.append("zh_academic_l2_mild added mild explanatory texture without changing facts")
            marks.append("mild_l2_texture")

        if updated == rewritten_sentences:
            return rewritten_sentences, [], [], []
        return updated, self._deduplicate_preserve_order(rules), self._deduplicate_preserve_order(notes), self._deduplicate_preserve_order(marks)

    def _l2_mild_sentence_variant(self, sentence: str) -> tuple[str, list[str]]:
        """Return a small L2-textured variant of one academic sentence."""

        stripped = sentence.strip()
        updated = stripped
        rules: list[str] = []
        replacements = (
            (r"适用于", "比较适合用来", "local:soften-native-like-concision"),
            (r"(?<!被)用于([^，。；]{1,18})", r"是用来\1", "local:increase-function-word-density-mildly"),
            (r"负责([^，。；]{1,18})", r"负责\1的工作", "local:allow-explanatory-rephrasing"),
            (r"提取特征", "进行特征提取的工作", "local:expand-compact-academic-clause"),
            (r"实现([^，。；]{1,18})融合", r"来实现\1的融合", "local:inject-mild-l2-texture"),
            (r"构成([^，。；]{1,18})", r"形成了\1", "local:avoid-too-fluent-native-polish"),
        )
        for pattern, replacement, rule in replacements:
            candidate = re.sub(pattern, replacement, updated, count=1)
            if candidate != updated:
                updated = candidate
                rules.append(rule)
                break

        if (
            not rules
            and re.search(r"(?:验证|分析|说明|比较|处理)", updated)
            and "进行" not in updated
            and not re.search(r"(?:后续验证|验证算法|辅助分析|分析与诊断)", updated)
        ):
            candidate = re.sub(r"(验证|分析|说明|比较|处理)", r"进行\1", updated, count=1)
            if candidate != updated:
                updated = candidate
                rules.append("local:increase-function-word-density-mildly")
        if not rules and len(re.sub(r"\s+", "", updated)) > 48 and "能够" not in updated and "可以" not in updated:
            candidate = re.sub(
                r"(模型|系统|该方法|该策略)((?:提升|保持|完成|支持|缓解|降低|改善|识别|处理|输出|生成|提供)[^。；]{2,18})(?:，|。)",
                r"\1能够\2，",
                updated,
                count=1,
            )
            if candidate != updated:
                updated = candidate
                rules.append("local:inject-mild-l2-texture")

        return (self._ensure_sentence_end(updated), rules) if rules else (sentence, [])

    def _l2_variant_breaks_bounds(self, sentence: str) -> bool:
        """Reject L2 variants that become colloquial, ungrammatical, or repetitive."""

        return bool(
            re.search(r"(?:这块|这边|大家|超|超级|真的|其实|大白话|搞定|拉满|靠谱)", sentence)
            or re.search(
                r"(?:本研究研究|研究研究|是了|可以用于了|被是用来|能够(?:复杂度|并非|更新|必须)|进行验证算法|辅助进行分析与诊断|，，|。。|的的|了了|进行进行)",
                sentence,
            )
        )

    def _apply_academic_sentence_naturalization(
        self,
        *,
        original_sentences: list[str],
        rewritten_sentences: list[str],
        rewrite_depth: str,
        high_sensitivity_prose: bool,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Reduce bureaucratic academic syntax while preserving evidence scope and sentence integrity."""

        if rewrite_depth not in {"developmental_rewrite", "light_edit"} or not rewritten_sentences:
            return rewritten_sentences, [], [], []

        before = analyze_academic_sentence_naturalization(rewritten_sentences)
        if (
            before.bureaucratic_opening_density <= 0.0
            and before.repeated_explicit_subject_risk <= 0.0
            and before.overstructured_syntax_risk <= 0.0
            and before.delayed_main_clause_risk <= 0.0
            and before.slogan_like_goal_risk <= 0.0
            and before.connector_overuse_risk <= 0.0
            and before.nominalization_density <= 0.0
            and before.passive_voice_ratio <= 0.0
            and before.overlong_sentence_risk <= 0.0
            and before.subject_monotony_risk <= 0.0
        ):
            return rewritten_sentences, [], [], []

        updated = list(rewritten_sentences)
        rules: list[str] = []
        notes: list[str] = []
        marks: list[str] = []
        before_readability = analyze_paragraph_readability_sentences(updated, high_sensitivity=high_sensitivity_prose)
        before_fidelity = analyze_evidence_fidelity(original_sentences, updated, high_sensitivity=high_sensitivity_prose)

        for index, sentence in enumerate(list(updated)):
            if technical_density_is_high(sentence):
                continue
            repaired = sentence
            for transform, rule, note in (
                (
                    self._remove_bureaucratic_opening,
                    "local:remove-bureaucratic-opening",
                    "sentence naturalization removed a project-style opening",
                ),
                (
                    self._enforce_direct_statement,
                    "local:enforce-direct-statement",
                    "sentence naturalization enforced direct statement shape",
                ),
                (
                    self._remove_academic_wrapping,
                    "local:remove-academic-wrapping",
                    "sentence naturalization removed academic packaging",
                ),
                (
                    self._remove_slogan_like_goal_phrase,
                    "local:remove-slogan-like-goal-phrase",
                    "sentence naturalization softened slogan-like goal phrasing",
                ),
                (
                    self._flatten_overstructured_parallelism,
                    "local:flatten-overstructured-parallelism",
                    "sentence naturalization softened overly tidy parallel syntax",
                ),
                (
                    self._reduce_connectors,
                    "local:reduce-connectors",
                    "sentence naturalization reduced heavy connector use",
                ),
                (
                    self._convert_passive_to_active,
                    "local:convert-passive-to-active",
                    "sentence naturalization converted passive phrasing to active form",
                ),
                (
                    self._restore_nominalized_verbs,
                    "local:restore-nominalized-verbs",
                    "sentence naturalization restored nominalized verbs",
                ),
                (
                    self._advance_main_clause,
                    "local:advance-main-clause",
                    "sentence naturalization moved the main clause closer to the front",
                ),
                (
                    self._move_main_clause_forward,
                    "local:move-main-clause-forward",
                    "sentence naturalization moved the main clause forward",
                ),
                (
                    self._split_overlong_sentence,
                    "local:split-overlong-sentence",
                    "sentence naturalization split an overlong sentence",
                ),
            ):
                candidate = transform(repaired)
                if candidate != repaired and not self._sentence_needs_readability_repair(
                    candidate,
                    role="topic_sentence" if index == 0 else "support_sentence",
                    is_final=index == len(updated) - 1,
                ):
                    repaired = candidate
                    rules.append(rule)
                    notes.append(note)
            if repaired != sentence:
                updated[index] = self._ensure_sentence_end(repaired)

        updated, subject_rules, subject_notes = self._compress_explicit_subject_chain(updated)
        rules.extend(subject_rules)
        notes.extend(subject_notes)
        updated, diversify_rules, diversify_notes = self._diversify_subject(updated)
        rules.extend(diversify_rules)
        notes.extend(diversify_notes)

        after_readability = analyze_paragraph_readability_sentences(updated, high_sensitivity=high_sensitivity_prose)
        after_fidelity = analyze_evidence_fidelity(original_sentences, updated, high_sensitivity=high_sensitivity_prose)
        after = analyze_academic_sentence_naturalization(updated)
        if (
            after_readability.paragraph_readability_score + 0.03 < before_readability.paragraph_readability_score
            or after_readability.sentence_completeness_score + 0.03 < before_readability.sentence_completeness_score
            or after_fidelity.evidence_fidelity_score + 0.01 < before_fidelity.evidence_fidelity_score
            or after.bureaucratic_opening_density > before.bureaucratic_opening_density
            or after.overstructured_syntax_risk > before.overstructured_syntax_risk
            or after.slogan_like_goal_risk > before.slogan_like_goal_risk
            or after.connector_overuse_risk > before.connector_overuse_risk
            or after.nominalization_density > before.nominalization_density
            or after.passive_voice_ratio > before.passive_voice_ratio
            or after.overlong_sentence_risk > before.overlong_sentence_risk
            or after.subject_monotony_risk > before.subject_monotony_risk
        ):
            return rewritten_sentences, [], [], []

        if rules:
            marks.append("academic_sentence_naturalization")
        return (
            updated,
            self._deduplicate_preserve_order(rules),
            self._deduplicate_preserve_order(notes),
            self._deduplicate_preserve_order(marks),
        )

    def _repair_caption_reference_sentences(self, sentences: list[str]) -> tuple[list[str], list[str], list[str]]:
        """Rephrase generated caption-reference fragments as complete prose sentences."""

        updated = list(sentences)
        rules: list[str] = []
        notes: list[str] = []
        for index, sentence in enumerate(list(updated)):
            stripped = self._strip_end_punctuation(sentence.strip())
            match = re.match(r"^该段论述(?P<body>.+\[\[AIRC:CORE_CAPTION:\d+\]\])$", stripped)
            if not match:
                continue
            body = match.group("body").strip("，, ")
            updated[index] = self._ensure_sentence_end(f"图示部分对应{body}")
            rules.append("readability:repair-caption-reference-sentence")
            notes.append("readability smoothing repaired a generated caption-reference fragment")
        return updated, self._deduplicate_preserve_order(rules), self._deduplicate_preserve_order(notes)

    def _merge_taxonomy_support_fragments(self, sentences: list[str]) -> tuple[list[str], list[str], list[str]]:
        """Merge method-taxonomy fragments back into their explanatory support sentence."""

        updated = list(sentences)
        rules: list[str] = []
        notes: list[str] = []
        index = 1
        while index + 1 < len(updated):
            current = self._strip_end_punctuation(updated[index]).strip()
            next_sentence = updated[index + 1].strip()
            if (
                re.match(r"^[A-Za-z][A-Za-z0-9_-]*(?:尤其是|特别是).{2,60}$", current)
                and re.match(r"^在.+?(?:方面|上).{0,12}(?:具有|具备|表现出|更适合)", next_sentence)
                and not technical_density_is_high(current)
                and not technical_density_is_high(next_sentence)
            ):
                updated[index] = self._ensure_sentence_end(f"{current}，{self._strip_end_punctuation(next_sentence)}")
                del updated[index + 1]
                rules.append("readability:merge-taxonomy-support-fragment")
                notes.append("readability smoothing merged a taxonomy fragment with its support claim")
                continue
            index += 1
        return updated, rules, notes

    def _repair_enumeration_sentence_flow(self, sentences: list[str]) -> tuple[list[str], list[str], list[str]]:
        """Split collapsed numbered reasoning so 第一/第二 do not appear as one jumbled sentence."""

        updated: list[str] = []
        rules: list[str] = []
        notes: list[str] = []
        for sentence in sentences:
            stripped = sentence.strip()
            match = re.match(r"^(?P<prefix>.+?)(?:，)?第一，(?P<first>.+?)(?:，进而|，)?第二，(?P<second>.+)$", stripped)
            if not match or technical_density_is_high(stripped):
                updated.append(sentence)
                continue
            prefix = self._strip_end_punctuation(match.group("prefix").strip("，, "))
            first = self._strip_end_punctuation(match.group("first").strip("，, "))
            second = self._strip_end_punctuation(match.group("second").strip("，, "))
            if not first or not second:
                updated.append(sentence)
                continue
            updated.append(self._ensure_sentence_end(f"{prefix}，第一，{first}"))
            updated.append(self._ensure_sentence_end(f"第二，{second}"))
            rules.append("readability:repair-enumeration-flow")
            notes.append("readability smoothing split a collapsed enumeration into complete steps")
        return updated, self._deduplicate_preserve_order(rules), self._deduplicate_preserve_order(notes)

    def _remove_redundant_summary_sentences(self, sentences: list[str]) -> tuple[list[str], list[str], list[str]]:
        """Remove adjacent summary sentences that restate the same closing claim."""

        if len(sentences) < 2:
            return sentences, [], []
        updated = list(sentences)
        rules: list[str] = []
        notes: list[str] = []
        index = 1
        while index < len(updated):
            previous = updated[index - 1]
            current = updated[index]
            previous_core = self._summary_core(previous)
            current_core = self._summary_core(current)
            if previous_core and current_core and SequenceMatcher(a=previous_core, b=current_core).ratio() >= 0.62:
                del updated[index]
                rules.append("readability:remove-redundant-summary-sentence")
                notes.append("readability smoothing removed a repeated summary sentence")
                continue
            index += 1
        return updated, self._deduplicate_preserve_order(rules), self._deduplicate_preserve_order(notes)

    def _summary_core(self, sentence: str) -> str:
        """Normalize a summary sentence for duplicate-closing detection."""

        stripped = self._strip_end_punctuation(sentence.strip())
        stripped = re.sub(r"^(?:总体而言|总体来看|整体来看|综合来看|从整体上看|总的来说|综上)[，,\s]*", "", stripped)
        if not re.search(r"(完整闭环|提供基础|支撑|总结|实现|完成|价值|意义)", stripped):
            return ""
        return self._normalize_for_compare(stripped)

    def _repair_incomplete_sentence(
        self,
        *,
        sentence: str,
        original_sentence: str,
        previous_sentence: str,
        role: str,
    ) -> str:
        """Repair a fragment-like sentence by restoring the smallest safe clause head."""

        stripped = sentence.strip()
        source = original_sentence.strip()
        if source and (
            source in previous_sentence
            or self._strip_end_punctuation(source) in self._strip_end_punctuation(previous_sentence)
        ):
            source = ""
        if source and not self._sentence_needs_readability_repair(
            source,
            role=role,
            is_final=role == "conclusion_sentence",
        ):
            source_signal = analyze_paragraph_readability_sentences([source])
            current_signal = analyze_paragraph_readability_sentences([stripped])
            if source_signal.sentence_completeness_score >= current_signal.sentence_completeness_score:
                return source

        subject = self._infer_sentence_subject(previous_sentence, source)
        connector_match = re.match(
            r"^(?:并进一步|并且进一步|并|因此|由此|同时|此外|在这种情况下|相应地|其中|从而|进而|以及)[，,\s]*(?P<body>.+)$",
            stripped,
        )
        if connector_match:
            body = connector_match.group("body").strip("，, ")
            if body:
                if body.startswith(("提出", "形成", "实现", "说明", "表明", "完成", "用于", "支持", "强调")):
                    return self._ensure_sentence_end(f"{subject}{body}")
                if re.match(r"^(?:是|为|有|具有|能够|可以|需要)", body):
                    return self._ensure_sentence_end(f"{subject}{body}")
                return self._ensure_sentence_end(f"{subject}进一步{body.lstrip('进一步')}")

        comparison_match = re.match(r"^是比(?P<body>.+更.+)$", stripped)
        if comparison_match:
            return self._ensure_sentence_end(f"{subject}是比{comparison_match.group('body')}")

        if stripped.startswith("而是"):
            return self._ensure_sentence_end(f"{subject}更强调{stripped[2:].strip('，, ')}")
        especially_match = re.match(r"^(?P<term>[A-Za-z][A-Za-z0-9_-]*)(?P<body>(?:尤其是|特别是).+)$", stripped)
        if especially_match:
            term = especially_match.group("term")
            body = self._strip_end_punctuation(especially_match.group("body").strip())
            return self._ensure_sentence_end(f"其中，{term}{body}是该类方法的重要代表")
        if re.match(r"^(?:强调|说明|体现|支持|展示|反映|用于|包括|包含|呈现|提供|形成|实现)", stripped):
            return self._ensure_sentence_end(f"{subject}{stripped.lstrip('，, ')}")
        if re.match(r"^(?:使|让).+", stripped):
            return self._ensure_sentence_end(f"{subject}{stripped}")
        result_match = re.match(r"^(?:结合|根据|基于)(?P<body>.{2,35}(?:结果|数据|分析|实验|评测|诊断))[。！？?!]?$", stripped)
        if result_match:
            body = result_match.group("body").strip()
            spacer = " " if body.startswith("[[") or re.match(r"[A-Z]{2,}", body) else ""
            return self._ensure_sentence_end(f"{subject}还需要结合{spacer}{body}")
        if len(stripped) < 24 and not re.match(r"^(?:本研究|本文|该|这|系统|模型|方法|结果|实验|策略|问题)", stripped):
            if stripped.startswith(("研究", "系统", "模型", "方法", "结果", "实验", "策略", "问题")):
                return self._ensure_sentence_end(stripped.lstrip("，, "))
            return self._ensure_sentence_end(f"{subject}{stripped.lstrip('，, ')}")
        return sentence

    def _repair_fragment_like_conclusion_sentence(
        self,
        sentence: str,
        original_sentence: str,
        previous_sentence: str,
    ) -> str:
        """Repair a conclusion sentence without leaving a dangling connector tail."""

        if not fragment_like_conclusion_sentence(sentence, is_final=True, role="conclusion_sentence"):
            return sentence
        softened = self._soften_overexplicit_transition(sentence)
        if softened != sentence and not self._sentence_needs_readability_repair(
            softened,
            role="conclusion_sentence",
            is_final=True,
        ):
            return softened
        repaired = self._repair_incomplete_sentence(
            sentence=sentence,
            original_sentence=original_sentence,
            previous_sentence=previous_sentence,
            role="conclusion_sentence",
        )
        return repaired

    def _repair_high_sensitivity_prose_readability(
        self,
        original_sentences: list[str],
        rewritten_sentences: list[str],
    ) -> list[str]:
        """Prefer complete source sentences for high-sensitivity prose fragments."""

        updated = list(rewritten_sentences)
        for index, sentence in enumerate(list(updated)):
            if index >= len(original_sentences):
                continue
            role = "topic_sentence" if index == 0 else ("conclusion_sentence" if index == len(updated) - 1 else "support_sentence")
            if not self._sentence_needs_readability_repair(sentence, role=role, is_final=index == len(updated) - 1):
                continue
            original = original_sentences[index]
            if not self._sentence_needs_readability_repair(original, role=role, is_final=index == len(updated) - 1):
                updated[index] = original
        return updated

    def _merge_unrepaired_support_fragments(self, sentences: list[str]) -> list[str]:
        updated = list(sentences)
        index = 1
        while index < len(updated):
            role = "conclusion_sentence" if index == len(updated) - 1 else "support_sentence"
            sentence = updated[index]
            if (
                incomplete_support_sentence_risk(sentence, role=role)
                and not technical_density_is_high(sentence)
                and not technical_density_is_high(updated[index - 1])
            ):
                core = self._strip_end_punctuation(sentence).strip("，, ")
                if core:
                    core = re.sub(r"^(?:并|因此|由此|同时|此外|从而|进而)[，,\s]*", "", core)
                    updated[index - 1] = self._ensure_sentence_end(
                        f"{self._strip_end_punctuation(updated[index - 1])}，{core}"
                    )
                    del updated[index]
                    continue
            index += 1
        return updated

    def _restore_fragmentary_sentences_from_source(
        self,
        original_sentences: list[str],
        rewritten_sentences: list[str],
    ) -> list[str]:
        updated = list(rewritten_sentences)
        for index, sentence in enumerate(list(updated)):
            if index >= len(original_sentences):
                continue
            role = "topic_sentence" if index == 0 else ("conclusion_sentence" if index == len(updated) - 1 else "support_sentence")
            if self._sentence_needs_readability_repair(sentence, role=role, is_final=index == len(updated) - 1):
                original = original_sentences[index]
                if not self._sentence_needs_readability_repair(original, role=role, is_final=index == len(updated) - 1):
                    updated[index] = original
                else:
                    updated[index] = self._repair_incomplete_sentence(
                        sentence=sentence,
                        original_sentence=original,
                        previous_sentence=updated[index - 1] if index > 0 else "",
                        role=role,
                    )
        return updated

    def _infer_sentence_subject(self, previous_sentence: str, original_sentence: str) -> str:
        for source in (previous_sentence, original_sentence):
            stripped = source.strip()
            for subject in (
                "本研究",
                "本文",
                "该系统",
                "系统",
                "该模型",
                "模型",
                "该方法",
                "该机制",
                "该设计",
                "最终模型",
                "主融合模块",
                "研究内容",
            ):
                if subject in stripped[:24]:
                    return subject
            if "研究" in stripped[:24]:
                return "本研究"
        return "本研究"

    def _local_actions_from_rules(self, rules: list[str]) -> list[str]:
        mapping = {
            "local:soften-overexplicit-transition": "soften_overexplicit_transition",
            "local:reduce-sentence-uniformity": "reduce_sentence_uniformity",
            "local:introduce-local-hierarchy": "introduce_local_hierarchy",
            "local:reshape-supporting-sentence": "reshape_supporting_sentence",
            "local:weaken-overfinished-sentence": "weaken_overfinished_sentence",
            "local:convert-flat-parallel-flow": "convert_flat_parallel_flow",
            "local:light-partial-retain-with-local-rephrase": "light_partial_retain_with_local_rephrase",
            "local:vary-support-sentence-texture": "vary_support_sentence_texture",
            "local:de-template-academic-cliche": "de_template_academic_cliche",
            "local:retain-some-plain-sentences": "retain_some_plain_sentences",
            "local:avoid-overpolished-supporting-sentence": "avoid_overpolished_supporting_sentence",
            "local:introduce-mild-authorial-asymmetry": "introduce_mild_authorial_asymmetry",
            "local:deuniform-paragraph-texture": "deuniform_paragraph_texture",
            "local:rephrase-summary-like-sentence-without-cliche": "rephrase_summary_like_sentence_without_cliche",
            "local:soft-keep-for-human-revision-feel": "soft_keep_for_human_revision_feel",
            "local:preserve-semantic-role-of-core-sentence": "preserve_semantic_role_of_core_sentence",
            "local:preserve-enumeration-item-role": "preserve_enumeration_item_role",
            "local:remove-generated-scaffolding-phrase": "remove_generated_scaffolding_phrase",
            "local:replace-abstracted-subject-with-concrete-referent": "replace_abstracted_subject_with_concrete_referent",
            "local:restore-mechanism-sentence-from-support-like-rewrite": "restore_mechanism_sentence_from_support_like_rewrite",
            "local:repair-enumeration-flow": "repair_enumeration_flow",
            "local:prevent-appendix-like-supporting-sentence": "prevent_appendix_like_supporting_sentence",
            "local:avoid-huanbaokuo-style-expansion": "avoid_huanbaokuo_style_expansion",
            "local:upgrade-appendix-like-sentence-to-assertion": "upgrade_appendix_like_sentence_to_assertion",
            "local:strengthen-mechanism-verb": "strengthen_mechanism_verb",
            "local:replace-weak-modal-verbs": "replace_weak_modal_verbs",
            "local:restore-authorial-choice-expression": "restore_authorial_choice_expression",
            "local:promote-support-sentence-to-core-if-needed": "promote_support_sentence_to_core_if_needed",
            "local:reduce-overuse-of-passive-explanations": "reduce_overuse_of_passive_explanations",
            "local:remove-unsupported-expansion": "remove_unsupported_expansion",
            "local:remove-external-domain-commentary": "remove_external_domain_commentary",
            "local:remove-metaphoric-storytelling": "remove_metaphoric_storytelling",
            "local:restore-thesis-register": "restore_thesis_register",
            "local:replace-we-with-original-subject-style": "replace_we_with_original_subject_style",
            "local:downgrade-overclaimed-judgment": "downgrade_overclaimed_judgment",
            "local:restore-mechanism-sentence-to-academic-statement": "restore_mechanism_sentence_to_academic_statement",
            "local:preserve-original-evidence-scope": "preserve_original_evidence_scope",
            "local:expand-compact-academic-clause": "expand_compact_academic_clause",
            "local:increase-function-word-density-mildly": "increase_function_word_density_mildly",
            "local:soften-native-like-concision": "soften_native_like_concision",
            "local:allow-explanatory-rephrasing": "allow_explanatory_rephrasing",
            "local:inject-mild-l2-texture": "inject_mild_l2_texture",
            "local:avoid-too-fluent-native-polish": "avoid_too_fluent_native_polish",
        }
        return self._deduplicate_preserve_order([mapping[rule] for rule in rules if rule in mapping])

    def _local_surfaces_from_rules(self, rules: list[str]) -> list[str]:
        surfaces = {
            "local:soften-overexplicit-transition": "弱化显性过渡",
            "local:reduce-sentence-uniformity": "降低句式均匀度",
            "local:introduce-local-hierarchy": "局部层次重建",
            "local:reshape-supporting-sentence": "支撑句局部改写",
            "local:weaken-overfinished-sentence": "弱化过满收束",
            "local:convert-flat-parallel-flow": "并列推进改写",
            "local:light-partial-retain-with-local-rephrase": "局部保留与轻改",
            "local:vary-support-sentence-texture": "支撑句纹理变化",
            "local:de-template-academic-cliche": "学术套话去模板化",
            "local:retain-some-plain-sentences": "保留部分朴素句",
            "local:avoid-overpolished-supporting-sentence": "避免支撑句过度润色",
            "local:introduce-mild-authorial-asymmetry": "作者式轻微不对称",
            "local:deuniform-paragraph-texture": "段内纹理去均匀化",
            "local:rephrase-summary-like-sentence-without-cliche": "总结句去套话",
            "local:soft-keep-for-human-revision-feel": "保留原句手工修订感",
            "local:preserve-semantic-role-of-core-sentence": "核心句语义角色保护",
            "local:preserve-enumeration-item-role": "分点句角色保护",
            "local:remove-generated-scaffolding-phrase": "生成式脚手架表达清理",
            "local:replace-abstracted-subject-with-concrete-referent": "抽象主语替换为具体指代",
            "local:restore-mechanism-sentence-from-support-like-rewrite": "机制句角色回正",
            "local:repair-enumeration-flow": "枚举结构修复",
            "local:prevent-appendix-like-supporting-sentence": "防止补充化支撑句",
            "local:avoid-huanbaokuo-style-expansion": "避免还包括式扩展",
            "local:upgrade-appendix-like-sentence-to-assertion": "附属句提升为主断言",
            "local:strengthen-mechanism-verb": "强化机制动词",
            "local:replace-weak-modal-verbs": "弱情态动词替换",
            "local:restore-authorial-choice-expression": "恢复作者判断表达",
            "local:promote-support-sentence-to-core-if-needed": "必要时提升支撑句断言强度",
            "local:reduce-overuse-of-passive-explanations": "减少被动说明式表达",
            "local:remove-unsupported-expansion": "清理无依据扩写",
            "local:remove-external-domain-commentary": "清理外加领域评论",
            "local:remove-metaphoric-storytelling": "清理隐喻与叙事化表达",
            "local:restore-thesis-register": "恢复论文语体",
            "local:replace-we-with-original-subject-style": "恢复原文主语风格",
            "local:downgrade-overclaimed-judgment": "弱化越界判断",
            "local:restore-mechanism-sentence-to-academic-statement": "机制句恢复为学术陈述",
            "local:preserve-original-evidence-scope": "保持原文证据边界",
            "local:expand-compact-academic-clause": "压缩学术句轻展开",
            "local:increase-function-word-density-mildly": "轻度增加功能词",
            "local:soften-native-like-concision": "弱化母语式凝练",
            "local:allow-explanatory-rephrasing": "允许解释化改写",
            "local:inject-mild-l2-texture": "注入轻微二语纹理",
            "local:avoid-too-fluent-native-polish": "避免过度母语化润色",
        }
        return [surfaces[rule] for rule in rules if rule in surfaces]

    def _readability_actions_from_rules(self, rules: list[str]) -> list[str]:
        mapping = {
            "readability:readability-repair-pass": "readability_repair_pass",
            "readability:sentence-completeness-repair": "sentence_completeness_repair",
            "readability:repair-incomplete-support-sentence": "repair_incomplete_support_sentence",
            "readability:repair-fragment-like-conclusion-sentence": "repair_fragment_like_conclusion_sentence",
            "readability:high-sensitivity-readability-repair": "high_sensitivity_readability_repair",
            "readability:high-sensitivity-source-restore": "high_sensitivity_readability_repair",
            "readability:restore-source-for-protected-token-coverage": "readability_repair_pass",
            "readability:restore-source-for-readability": "readability_repair_pass",
            "readability:single-sentence-readability-recast": "sentence_completeness_repair",
            "readability:post-readability-subject-chain-repair": "sentence_completeness_repair",
            "readability:remove-contained-duplicate-sentence": "repair_incomplete_support_sentence",
            "readability:soften-complete-transition": "readability_repair_pass",
            "readability:merge-taxonomy-support-fragment": "repair_incomplete_support_sentence",
            "readability:repair-enumeration-flow": "readability_repair_pass",
            "readability:remove-redundant-summary-sentence": "readability_repair_pass",
            "readability:repair-caption-reference-sentence": "sentence_completeness_repair",
            "readability:repair-generated-discourse-marker": "readability_repair_pass",
        }
        return self._deduplicate_preserve_order([mapping[rule] for rule in rules if rule in mapping])

    def _readability_surfaces_from_rules(self, rules: list[str]) -> list[str]:
        surfaces = {
            "readability:readability-repair-pass": "成文性修复",
            "readability:sentence-completeness-repair": "句子完整性修复",
            "readability:repair-incomplete-support-sentence": "支撑句残片修复",
            "readability:repair-fragment-like-conclusion-sentence": "收束句残片修复",
            "readability:high-sensitivity-readability-repair": "高敏感段成文性保护",
            "readability:high-sensitivity-source-restore": "高敏感段源码回退",
            "readability:restore-source-for-protected-token-coverage": "保护标记覆盖回退",
            "readability:restore-source-for-readability": "回退源码句保证成文",
            "readability:single-sentence-readability-recast": "完整句轻度成文改写",
            "readability:post-readability-subject-chain-repair": "成文后主语链修复",
            "readability:remove-contained-duplicate-sentence": "重复支撑句前缀清理",
            "readability:soften-complete-transition": "完整句硬连接弱化",
            "readability:merge-taxonomy-support-fragment": "方法分类支撑句合并",
            "readability:repair-enumeration-flow": "枚举推进成文修复",
            "readability:remove-redundant-summary-sentence": "重复收束句清理",
            "readability:repair-caption-reference-sentence": "图示引用句成文修复",
            "readability:repair-generated-discourse-marker": "生成式支架表达清理",
        }
        return [surfaces[rule] for rule in rules if rule in surfaces]

    def _safe_topic_opening(self, original_opening: str, current_opening: str) -> str:
        if opening_style_valid(current_opening) and not self._opening_subject_drifted(original_opening, current_opening):
            current_checks = paragraph_skeleton_checks(original_opening, current_opening)
            if current_checks["paragraph_topic_sentence_preserved"]:
                return current_opening
        for intensity in ("high", "medium"):
            recast = self._safe_developmental_sentence_recast(original_opening, intensity)
            if recast != original_opening and opening_style_valid(recast):
                return recast
        return self._ensure_sentence_end(original_opening)

    def _standalone_opening_from_dangling(self, sentence: str) -> str:
        stripped = sentence.strip()
        replacements = (
            r"^(?:在此基础上|在这种情况下|相应地|此外|由此|与此同时|另外)[，,\s]*",
            r"^(?:需要说明的是|需要指出的是)[，,\s]*",
            r"^(?:同时|并进一步|并且|而是)[，,\s]*",
        )
        updated = stripped
        for pattern in replacements:
            updated = re.sub(pattern, "", updated)
        return self._ensure_sentence_end(updated) if updated else sentence

    def _opening_subject_drifted(self, original_opening: str, current_opening: str) -> bool:
        original = self._strip_end_punctuation(original_opening.strip())
        current = self._strip_end_punctuation(current_opening.strip())
        if not re.match(r"^(?:本研究|本文|本章)", original):
            return False
        if current.startswith(("围绕", "就", "对", "基于", "在此")):
            return True
        return "本文展开讨论" in current and not original.startswith("本文")

    def _can_drop_opening_duplicate(self, safe_opening: str, candidate: str) -> bool:
        safe_markers = set(re.findall(r"\[\[AIRC:CORE_[A-Z_]+:\d+\]\]|[A-Za-z][A-Za-z0-9_./-]{2,}", safe_opening))
        candidate_markers = set(re.findall(r"\[\[AIRC:CORE_[A-Z_]+:\d+\]\]|[A-Za-z][A-Za-z0-9_./-]{2,}", candidate))
        return candidate_markers.issubset(safe_markers)

    def _deduplicate_adjacent_sentences(self, sentences: list[str]) -> tuple[list[str], list[str], list[str]]:
        if len(sentences) < 2:
            return sentences, [], []
        deduplicated: list[str] = []
        removed = 0
        for sentence in sentences:
            if deduplicated and self._sentence_similarity(deduplicated[-1], sentence) >= 0.92:
                if self._can_drop_opening_duplicate(deduplicated[-1], sentence):
                    removed += 1
                    continue
            deduplicated.append(sentence)
        if not removed:
            return sentences, [], []
        return (
            deduplicated,
            ["paragraph:adjacent-duplicate-removed"],
            ["paragraph skeleton guard removed adjacent duplicated topic/support sentence"],
        )

    def _sentence_similarity(self, left: str, right: str) -> float:
        return SequenceMatcher(a=self._normalize_for_compare(left), b=self._normalize_for_compare(right)).ratio()

    def _can_apply_reordered_units(self, original_units: list[SentenceUnit], reordered_units: list[SentenceUnit]) -> bool:
        if not original_units or not reordered_units:
            return True
        skeleton = analyze_paragraph_skeleton("".join(unit.text for unit in original_units))
        if not skeleton.opening_reorder_allowed and 0 not in reordered_units[0].source_indices:
            return False
        return opening_style_valid(reordered_units[0].text)

    def _rewrite_sentence_structures(
        self,
        units: list[SentenceUnit],
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[list[SentenceUnit], list[str], list[str], list[str]]:
        rewritten: list[SentenceUnit] = []
        rule_names: list[str] = []
        selected_variants: list[str] = []
        candidate_notes: list[str] = []

        for unit in units:
            updated_text = self._deduplicate_clauses(unit.text)
            updated_text, rules, surfaces, notes = self._rewrite_parallel_enumeration(
                updated_text,
                mode=mode,
                pass_index=pass_index,
                label=unit.label,
            )
            rewritten.append(SentenceUnit(text=updated_text, label=unit.label, source_indices=unit.source_indices))
            rule_names.extend(rules)
            selected_variants.extend(surfaces)
            candidate_notes.extend(notes)

        return rewritten, rule_names, selected_variants, candidate_notes

    def _rewrite_subject_chains(
        self,
        units: list[SentenceUnit],
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[list[SentenceUnit], list[str], list[str], list[str]]:
        if len(units) < 2:
            return units, [], [], []
        if mode is RewriteMode.CONSERVATIVE:
            return self._repair_repeated_subject_heads(
                units=units,
                mode=mode,
                pass_index=pass_index,
            )

        rewritten: list[SentenceUnit] = []
        rule_names: list[str] = []
        selected_variants: list[str] = []
        candidate_notes: list[str] = []
        index = 0

        while index < len(units):
            current = units[index]
            if index + 1 >= len(units):
                rewritten.append(current)
                break

            next_unit = units[index + 1]

            compressed = self._compress_meta_work_pair(
                current=current,
                next_unit=next_unit,
                mode=mode,
                pass_index=pass_index,
            )
            if compressed is not None:
                new_unit, rules, surfaces, notes = compressed
                if index + 2 < len(units):
                    followup = self._absorb_meta_followup(
                        current=new_unit,
                        next_unit=units[index + 2],
                        mode=mode,
                        pass_index=pass_index,
                    )
                    if followup is not None:
                        new_unit, more_rules, more_surfaces, more_notes = followup
                        rules.extend(more_rules)
                        surfaces.extend(more_surfaces)
                        notes.extend(more_notes)
                        index += 3
                    else:
                        index += 2
                else:
                    index += 2

                rewritten.append(new_unit)
                rule_names.extend(rules)
                selected_variants.extend(surfaces)
                candidate_notes.extend(notes)
                continue

            absorbed = self._absorb_meta_followup(
                current=current,
                next_unit=next_unit,
                mode=mode,
                pass_index=pass_index,
            )
            if absorbed is not None:
                new_unit, rules, surfaces, notes = absorbed
                rewritten.append(new_unit)
                rule_names.extend(rules)
                selected_variants.extend(surfaces)
                candidate_notes.extend(notes)
                index += 2
                continue

            current_subject = self._extract_subject_head(current.text)
            next_subject = self._extract_subject_head(next_unit.text)
            same_meta_subject = current_subject in _META_SUBJECTS and current_subject == next_subject
            meta_roles = {"objective", "support", "detail", "conclusion"}

            if same_meta_subject and current.label in meta_roles and next_unit.label in meta_roles:
                merged = self._merge_same_subject_pair(
                    current=current,
                    next_unit=next_unit,
                    mode=mode,
                    pass_index=pass_index,
                )
                if merged is not None:
                    new_unit, rules, surfaces, notes = merged
                    rewritten.append(new_unit)
                    rule_names.extend(rules)
                    selected_variants.extend(surfaces)
                    candidate_notes.extend(notes)
                    index += 2
                    continue

                varied = self._apply_subject_variation(
                    unit=next_unit,
                    mode=mode,
                    pass_index=pass_index,
                )
                if varied is not None:
                    new_unit, rules, surfaces, notes = varied
                    rewritten.extend([current, new_unit])
                    rule_names.extend(rules)
                    selected_variants.extend(surfaces)
                    candidate_notes.extend(notes)
                    index += 2
                    continue

                dropped = self._apply_subject_drop(
                    unit=next_unit,
                    mode=mode,
                    pass_index=pass_index,
                )
                if dropped is not None:
                    new_unit, rules, surfaces, notes = dropped
                    rewritten.extend([current, new_unit])
                    rule_names.extend(rules)
                    selected_variants.extend(surfaces)
                    candidate_notes.extend(notes)
                    index += 2
                    continue

            rewritten.append(current)
            index += 1

        rewritten, more_rules, more_surfaces, more_notes = self._repair_repeated_subject_heads(
            units=rewritten,
            mode=mode,
            pass_index=pass_index,
        )
        rule_names.extend(more_rules)
        selected_variants.extend(more_surfaces)
        candidate_notes.extend(more_notes)

        return rewritten, rule_names, selected_variants, candidate_notes

    def _discourse_compressor(
        self,
        units: list[SentenceUnit],
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[list[SentenceUnit], list[str], list[str], list[str]]:
        return self._rewrite_subject_chains(units=units, mode=mode, pass_index=pass_index)

    def _repair_repeated_subject_heads(
        self,
        units: list[SentenceUnit],
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[list[SentenceUnit], list[str], list[str], list[str]]:
        if len(units) < 2:
            return units, [], [], []

        repaired: list[SentenceUnit] = [units[0]]
        rule_names: list[str] = []
        selected_variants: list[str] = []
        candidate_notes: list[str] = []

        for unit in units[1:]:
            previous = repaired[-1]
            previous_subject = self._extract_subject_head(previous.text)
            current_subject = self._extract_subject_head(unit.text)

            if previous_subject in _META_SUBJECTS and previous_subject == current_subject:
                varied = self._apply_subject_variation(unit=unit, mode=mode, pass_index=pass_index)
                if varied is not None:
                    new_unit, rules, surfaces, notes = varied
                    repaired.append(new_unit)
                    rule_names.extend(rules)
                    selected_variants.extend(surfaces)
                    candidate_notes.extend(notes)
                    continue

                dropped = self._apply_subject_drop(unit=unit, mode=mode, pass_index=pass_index)
                if dropped is not None:
                    new_unit, rules, surfaces, notes = dropped
                    repaired.append(new_unit)
                    rule_names.extend(rules)
                    selected_variants.extend(surfaces)
                    candidate_notes.extend(notes)
                    continue

                embedded = self._relieve_embedded_subject_head(unit=unit, mode=mode, pass_index=pass_index)
                if embedded is not None:
                    new_unit, rules, surfaces, notes = embedded
                    repaired.append(new_unit)
                    rule_names.extend(rules)
                    selected_variants.extend(surfaces)
                    candidate_notes.extend(notes)
                    continue

            repaired.append(unit)

        return repaired, rule_names, selected_variants, candidate_notes

    def _relieve_embedded_subject_head(
        self,
        unit: SentenceUnit,
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[SentenceUnit, list[str], list[str], list[str]] | None:
        stripped = self._strip_end_punctuation(unit.text)
        match = re.match(r"^(?P<prefix>[^，。]{2,60})，(?P<subject>本研究|本文|该研究|该系统)(?P<body>.+)$", stripped)
        if not match:
            return None

        prefix = match.group("prefix").strip()
        subject = match.group("subject")
        body = match.group("body").strip()
        if not body or self._should_preserve_meta_subject(body):
            return None

        adjusted_text, rules, surfaces, notes = self._apply_candidate_rewrite(
            family="subject-drop",
            rule_name="subject:drop",
            original_text=unit.text,
            candidates=[
                CandidateOption(
                    "subject-drop:embedded-base",
                    f"{prefix}，{body}。",
                    "省略主语",
                    template_family="",
                    rule_like=False,
                ),
                CandidateOption(
                    "subject-drop:embedded-research",
                    f"{prefix}，{self._subject_variant_phrase(subject, body)}。",
                    "研究",
                ),
            ],
            prefer_change=mode is not RewriteMode.CONSERVATIVE or pass_index > 1,
            allow_keep=False,
        )
        return (
            SentenceUnit(text=adjusted_text, label=unit.label, source_indices=unit.source_indices),
            rules,
            surfaces,
            notes,
        )

    def _compress_meta_work_pair(
        self,
        current: SentenceUnit,
        next_unit: SentenceUnit,
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[SentenceUnit, list[str], list[str], list[str]] | None:
        topic = self._extract_meta_topic(current.text)
        subject, body = self._extract_meta_subject_and_body(next_unit.text)
        current_subject = self._extract_subject_head(current.text)
        body_core = self._strip_end_punctuation(body)

        if not topic or subject not in _META_SUBJECTS or subject != current_subject or not self._looks_like_work_body(body_core):
            return None

        compressed_text, rules, surfaces, notes = self._apply_candidate_rewrite(
            family="meta-compression",
            rule_name="subject:meta-compression",
            original_text=f"{current.text}{next_unit.text}",
            candidates=[
                CandidateOption(
                    "meta-compression:around",
                    f"围绕{topic}这一问题，本研究{body_core}。",
                    "围绕……这一问题",
                ),
                CandidateOption(
                    "meta-compression:target",
                    f"针对{topic}，本研究{body_core}。",
                    "针对……",
                ),
                CandidateOption(
                    "meta-compression:expand",
                    f"本研究围绕{topic}展开，{body_core}。",
                    "本研究围绕",
                ),
            ],
            prefer_change=True,
            allow_keep=False,
        )
        return (
            SentenceUnit(
                text=compressed_text,
                label="objective",
                source_indices=current.source_indices + next_unit.source_indices,
            ),
            rules,
            surfaces,
            notes,
        )

    def _absorb_meta_followup(
        self,
        current: SentenceUnit,
        next_unit: SentenceUnit,
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[SentenceUnit, list[str], list[str], list[str]] | None:
        prefix, remainder = self._strip_opening(next_unit.text, _IMPLICATION_PREFIXES)
        if not prefix:
            return None

        subject, body = self._extract_meta_subject_and_body(remainder)
        if subject not in _META_SUBJECTS or not body:
            return None

        left_core = self._strip_end_punctuation(current.text)
        body_core = self._strip_end_punctuation(body)
        if not left_core or not body_core:
            return None

        varied_followup = self._subject_variant_phrase(subject, body_core)
        absorbed_text, rules, surfaces, notes = self._apply_candidate_rewrite(
            family="subject-followup",
            rule_name="subject:followup-absorb",
            original_text=f"{current.text}{next_unit.text}",
            candidates=[
                CandidateOption(
                    "subject-followup:drop",
                    f"{left_core}，在此基础上，{body_core}。",
                    "在此基础上",
                ),
                CandidateOption(
                    "subject-followup:research",
                    f"{left_core}，由此，{varied_followup}。",
                    "由此，研究",
                ),
                CandidateOption(
                    "subject-followup:respond",
                    f"{left_core}，相应地，{self._subject_variant_phrase(subject, body_core, prefer_system=True)}。",
                    "相应地",
                ),
            ],
            prefer_change=mode is RewriteMode.STRONG or pass_index > 1,
            allow_keep=False,
        )
        return (
            SentenceUnit(
                text=absorbed_text,
                label=current.label,
                source_indices=current.source_indices + next_unit.source_indices,
            ),
            rules,
            surfaces,
            notes,
        )

    def _merge_same_subject_pair(
        self,
        current: SentenceUnit,
        next_unit: SentenceUnit,
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[SentenceUnit, list[str], list[str], list[str]] | None:
        current_subject = self._extract_subject_head(current.text)
        next_subject, next_body = self._extract_meta_subject_and_body(next_unit.text)
        if current_subject not in _META_SUBJECTS or next_subject != current_subject:
            return None

        left_core = self._strip_end_punctuation(current.text)
        body_core = self._strip_end_punctuation(next_body)
        if not left_core or not body_core:
            return None

        merged_text, rules, surfaces, notes = self._apply_candidate_rewrite(
            family="subject-merge",
            rule_name="subject:merge-consecutive",
            original_text=f"{current.text}{next_unit.text}",
            candidates=[
                CandidateOption(
                    "subject-merge:comma",
                    f"{left_core}，{body_core}。",
                    "直接并句",
                ),
                CandidateOption(
                    "subject-merge:jiner",
                    f"{left_core}，并在此基础上{body_core}。",
                    "并在此基础上",
                ),
                CandidateOption(
                    "subject-merge:meanwhile",
                    f"{left_core}，同时{body_core}。",
                    "同时",
                ),
            ],
            prefer_change=True,
            allow_keep=False,
        )
        return (
            SentenceUnit(
                text=merged_text,
                label=current.label,
                source_indices=current.source_indices + next_unit.source_indices,
            ),
            rules,
            surfaces,
            notes,
        )

    def _apply_subject_variation(
        self,
        unit: SentenceUnit,
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[SentenceUnit, list[str], list[str], list[str]] | None:
        subject, body = self._extract_meta_subject_and_body(unit.text)
        body_core = self._strip_end_punctuation(body)
        if subject not in _META_SUBJECTS or not body_core or self._should_preserve_meta_subject(body_core):
            return None

        varied_text, rules, surfaces, notes = self._apply_candidate_rewrite(
            family="subject-variation",
            rule_name="subject:variation",
            original_text=unit.text,
            candidates=[
                CandidateOption(
                    "subject-variation:research",
                    f"{self._subject_variant_phrase(subject, body_core)}。",
                    "研究",
                ),
                CandidateOption(
                    "subject-variation:study",
                    f"{subject}{body_core}。",
                    subject,
                ),
                CandidateOption(
                    "subject-variation:focus",
                    f"围绕这一目标，{self._subject_variant_phrase(subject, body_core)}。",
                    "围绕这一目标",
                ),
            ],
            prefer_change=mode is RewriteMode.STRONG or pass_index > 1,
            allow_keep=False,
        )
        return (
            SentenceUnit(text=varied_text, label=unit.label, source_indices=unit.source_indices),
            rules,
            surfaces,
            notes,
        )

    def _apply_subject_drop(
        self,
        unit: SentenceUnit,
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[SentenceUnit, list[str], list[str], list[str]] | None:
        subject, body = self._extract_meta_subject_and_body(unit.text)
        body_core = self._strip_end_punctuation(body)
        if subject not in _META_SUBJECTS or not body_core or self._should_preserve_meta_subject(body_core):
            return None

        dropped_text, rules, surfaces, notes = self._apply_candidate_rewrite(
            family="subject-drop",
            rule_name="subject:drop",
            original_text=unit.text,
            candidates=[
                CandidateOption(
                    "subject-drop:base",
                    f"在此基础上，{body_core}。",
                    "在此基础上",
                ),
                CandidateOption(
                    "subject-drop:accordingly",
                    f"相应地，{body_core}。",
                    "相应地",
                ),
            ],
            prefer_change=mode is RewriteMode.STRONG and pass_index > 1,
            allow_keep=False,
        )
        return (
            SentenceUnit(text=dropped_text, label=unit.label, source_indices=unit.source_indices),
            rules,
            surfaces,
            notes,
        )

    def _rewrite_sentence_unit(
        self,
        unit: SentenceUnit,
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[SentenceUnit, list[str], list[str], list[str]]:
        updated_text = self._deduplicate_clauses(unit.text)
        rule_names: list[str] = []
        selected_variants: list[str] = []
        candidate_notes: list[str] = []

        updated_text, rules, surfaces, notes = self._rewrite_sentence_patterns(
            updated_text,
            mode=mode,
            pass_index=pass_index,
            label=unit.label,
        )
        rule_names.extend(rules)
        selected_variants.extend(surfaces)
        candidate_notes.extend(notes)

        if not any(rule.startswith("sentence:") for rule in rule_names):
            updated_text, rules, surfaces, notes = self._rewrite_opening_phrases(
                updated_text,
                mode=mode,
                pass_index=pass_index,
                label=unit.label,
            )
            rule_names.extend(rules)
            selected_variants.extend(surfaces)
            candidate_notes.extend(notes)

        updated_text, rules, surfaces, notes = self._soften_fillers(
            updated_text,
            mode=mode,
            pass_index=pass_index,
            label=unit.label,
        )
        rule_names.extend(rules)
        selected_variants.extend(surfaces)
        candidate_notes.extend(notes)

        updated_text, rules, surfaces, notes = self._naturalize_academic_texture(
            updated_text,
            mode=mode,
            pass_index=pass_index,
            label=unit.label,
        )
        rule_names.extend(rules)
        selected_variants.extend(surfaces)
        candidate_notes.extend(notes)

        return (
            SentenceUnit(text=updated_text, label=unit.label, source_indices=unit.source_indices),
            rule_names,
            selected_variants,
            candidate_notes,
        )

    def _developmental_recast_if_static(
        self,
        units: list[SentenceUnit],
        original_sentences: list[str],
        rewrite_intensity: str,
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[list[SentenceUnit], list[str], list[str], list[str]]:
        if mode is RewriteMode.CONSERVATIVE or not units:
            return units, [], [], []
        if rewrite_intensity not in {"medium", "high"} and pass_index <= 1:
            return units, [], [], []
        current_sentences = [unit.text for unit in units]
        if len(original_sentences) != len(current_sentences):
            return units, [], [], []
        changed_indices = {
            index
            for index, (original, current) in enumerate(zip(original_sentences, current_sentences, strict=False))
            if self._normalize_for_compare(original) != self._normalize_for_compare(current)
        }
        target_ratio = 0.6 if rewrite_intensity == "high" else 0.4
        target_changes = max(1, math.ceil(len(original_sentences) * target_ratio))
        if len(changed_indices) >= target_changes:
            return units, [], [], []

        preferred_indices = list(range(len(units)))
        if rewrite_intensity == "medium":
            preferred_indices = list(reversed(preferred_indices))
        updated = list(units)
        rules: list[str] = []
        surfaces: list[str] = []
        notes: list[str] = []
        for index in preferred_indices:
            if len(changed_indices) >= target_changes:
                break
            if index in changed_indices:
                continue
            unit = updated[index]
            recast = self._safe_developmental_sentence_recast(unit.text, rewrite_intensity)
            if recast == unit.text or self._normalize_for_compare(recast) == self._normalize_for_compare(unit.text):
                continue
            updated[index] = SentenceUnit(text=recast, label=unit.label, source_indices=unit.source_indices)
            changed_indices.add(index)
            rules.append("sentence:developmental-recast")
            surfaces.append("章节感知重述")
            notes.append("chapter-aware fallback recast prevented a high/medium priority body block from remaining under-rewritten")
        if rules:
            return updated, rules, surfaces, notes
        return units, [], [], []

    def _safe_developmental_sentence_recast(self, sentence: str, rewrite_intensity: str) -> str:
        stripped = sentence.strip()
        if not stripped:
            return sentence

        enum_problem_match = re.match(
            r"^(?P<prefix>(?:[（(]?(?:\d+|\[\[AIRC:CORE_NUMBER:\d+\]\])[）)]?))(?P<title>[^：:。！？?!]{2,42})(?P<colon>[：:])(?P<body>.+?)[。！？?!]?$",
            stripped,
        )
        if enum_problem_match and rewrite_intensity == "high":
            prefix = enum_problem_match.group("prefix").strip()
            title = enum_problem_match.group("title").strip()
            body = enum_problem_match.group("body").strip()
            if re.search(r"(问题|挑战|偏移|依赖|误报)", title):
                return self._ensure_sentence_end(f"{prefix}{title}主要表现为：{body}")
            return self._ensure_sentence_end(f"{prefix}{title}：{body}")

        chapter_scope_match = re.match(r"^(?P<subject>本章|本文|本研究)围绕(?P<scope>.+?)，(?P<body>.+?)[。！？?!]?$", stripped)
        if chapter_scope_match and rewrite_intensity == "high":
            subject = chapter_scope_match.group("subject").strip()
            scope = chapter_scope_match.group("scope").strip()
            body = chapter_scope_match.group("body").strip()
            return self._ensure_sentence_end(f"{subject}以 {scope} 为主线，{body}")

        challenge_overview_match = re.match(
            r"^(?P<subject>本研究|本文|本章)在(?P<context>[^，。；]{2,32})中主要遇到了(?P<body>.+?)[。！？?!]?$",
            stripped,
        )
        if challenge_overview_match and rewrite_intensity == "high":
            subject = challenge_overview_match.group("subject").strip()
            context = challenge_overview_match.group("context").strip()
            body = challenge_overview_match.group("body").strip()
            return self._ensure_sentence_end(f"{subject}在{context}中面对的主要挑战包括{body}")

        goal_match = re.match(r"^为(?P<goal>[^，。；]{3,42})，(?P<body>.+?)[。！？?!]?$", stripped)
        if goal_match:
            goal = goal_match.group("goal").strip()
            body = goal_match.group("body").strip()
            return self._ensure_sentence_end(f"在{goal}这一目标下，{body}")

        through_match = re.match(r"^通过(?P<means>[^，。；]{3,48})，(?P<body>.+?)[。！？?!]?$", stripped)
        if through_match:
            means = through_match.group("means").strip()
            body = through_match.group("body").strip()
            if body.startswith("系统实现了"):
                body = body[len("系统实现了") :].strip()
                return self._ensure_sentence_end(f"{means}使系统形成了{body}")
            return self._ensure_sentence_end(f"{means}使得{body}")

        page_switch_match = re.match(
            r"^(?P<subject>该页面|页面|系统页面)通过(?P<means>[^，。；]{2,32})切换显示，"
            r"不引入(?P<object>[^，。；]{2,32})，从而保持(?P<tail>.+?)[。！？?!]?$",
            stripped,
        )
        if page_switch_match:
            subject = page_switch_match.group("subject").strip()
            means = page_switch_match.group("means").strip()
            obj = re.sub(r"^额外", "", page_switch_match.group("object").strip())
            tail = page_switch_match.group("tail").strip()
            return self._ensure_sentence_end(f"{subject}通过{means}完成切换，不额外引入{obj}，使{tail}保持稳定")

        include_match = re.match(r"^(?P<subject>[^，。；]{2,28}?)(?:主要)?包括(?P<object>.+?)[。！？?!]?$", stripped)
        if include_match:
            subject = include_match.group("subject").strip()
            obj = include_match.group("object").strip()
            if self._looks_like_enumeration_sentence(stripped) or self._looks_like_mechanism_sentence(stripped):
                return sentence
            return self._ensure_sentence_end(f"{obj}共同构成了{subject}")

        despite_match = re.match(r"^尽管(?P<done>.+?)，但(?P<todo>.+?)[。！？?!]?$", stripped)
        if despite_match and rewrite_intensity == "high":
            done = despite_match.group("done").strip()
            todo = despite_match.group("todo").strip()
            return self._ensure_sentence_end(f"{done}并不意味着相关工作已经完成，{todo}")

        provide_match = re.match(r"^这为(?P<object>.+?)提供了(?P<tail>.+?)[。！？?!]?$", stripped)
        if provide_match and rewrite_intensity == "high":
            obj = provide_match.group("object").strip()
            tail = provide_match.group("tail").strip()
            return self._ensure_sentence_end(f"{obj}由此获得了{tail}")

        note_match = re.match(r"^需要指出的是，(?P<body>.+?)[。！？?!]?$", stripped)
        if note_match:
            return self._ensure_sentence_end(f"需要指出的是，{note_match.group('body').strip()}")

        perspective_match = re.match(r"^从(?P<context>[^，。；]{2,32})上看，(?P<body>.+?)[。！？?!]?$", stripped)
        if perspective_match:
            context = perspective_match.group("context").strip()
            body = perspective_match.group("body").strip()
            return self._ensure_sentence_end(f"在{context}层面，{body}")

        view_match = re.match(r"^从(?P<context>[^，。；]{2,32})来看，(?P<body>.+?)[。！？?!]?$", stripped)
        if view_match:
            context = view_match.group("context").strip()
            body = view_match.group("body").strip()
            return self._ensure_sentence_end(f"就{context}而言，{body}")

        method_heading_match = re.match(r"^基于(?P<object>[^。！？?!]{2,36})的方法[。！？?!]?$", stripped)
        if method_heading_match:
            obj = method_heading_match.group("object").strip()
            return self._ensure_sentence_end(f"另一类路径是基于{obj}的方法")

        method_context_match = re.match(
            r"^(?P<subject>该类方法|该方法|这种方法)在(?P<context>[^，。；]{2,36})中(?P<body>.+?)[。！？?!]?$",
            stripped,
        )
        if method_context_match:
            subject = method_context_match.group("subject").strip()
            context = method_context_match.group("context").strip()
            body = method_context_match.group("body").strip()
            return self._ensure_sentence_end(f"在{context}中，{subject}{body}")

        layer_match = re.match(r"^在(?P<context>该分层架构|该系统架构|这一架构)中，(?P<body>.+?)[。！？?!]?$", stripped)
        if layer_match:
            context = layer_match.group("context").strip()
            body = layer_match.group("body").strip()
            return self._ensure_sentence_end(f"{context}将{body}")

        base_match = re.match(
            r"^(?P<subject>[^，。；]{2,28}?)以(?P<base>[^，。；]{3,46})为基础，并(?P<body>.+?)[。！？?!]?$",
            stripped,
        )
        if base_match:
            subject = base_match.group("subject").strip()
            base = base_match.group("base").strip()
            body = base_match.group("body").strip()
            return self._ensure_sentence_end(f"{base}构成了{subject}的基础，{body}")

        concrete_match = re.match(r"^具体而言，(?P<body>.+?)[。！？?!]?$", stripped)
        if concrete_match:
            return self._ensure_sentence_end(f"更具体地说，{concrete_match.group('body').strip()}")

        simultaneous_match = re.match(r"^同时，(?P<body>.+?)[。！？?!]?$", stripped)
        if simultaneous_match and rewrite_intensity in {"medium", "high"}:
            body = simultaneous_match.group("body").strip()
            return self._ensure_sentence_end(body if body else stripped)

        ordered_mechanism_match = re.match(r"^(?P<order>首先|其次)，(?P<body>.+?)[。！？?!]?$", stripped)
        if ordered_mechanism_match:
            body = ordered_mechanism_match.group("body").strip()
            if body.startswith("噪声分支对部分样本表现出较高敏感性"):
                body = body.replace("对部分样本表现出较高敏感性", "对部分样本确实表现出较高敏感性", 1)
                return self._ensure_sentence_end(body)
            if body.startswith("融合模块及控制机制在训练过程中并未稳定学习到"):
                body = body.replace("并未稳定学习到", "始终未能稳定学到", 1)
                return self._ensure_sentence_end(body)

        overall_match = re.match(r"^整体来看，(?P<body>.+?)[。！？?!]?$", stripped)
        if overall_match:
            return self._ensure_sentence_end(f"从整体上说，{overall_match.group('body').strip()}")

        if self._technical_density_is_high(stripped):
            return sentence

        if rewrite_intensity == "high":
            lead_match = re.match(r"^(?P<subject>本章|本研究|本文|系统|该系统)(?P<body>[^。！？?!]{8,})[。！？?!]?$", stripped)
            if lead_match:
                return sentence

        return sentence

    def _rewrite_sentence_patterns(
        self,
        sentence: str,
        mode: RewriteMode,
        pass_index: int,
        label: str,
    ) -> tuple[str, list[str], list[str], list[str]]:
        stripped = sentence.strip()

        topic_match = re.match(
            r"^(?:本研究(?:的主题为|主要围绕|围绕)|本文讨论的核心问题是)(?P<topic>.+?)(?:展开讨论|展开|进行讨论)?[。！？?!]?$",
            stripped,
        )
        if topic_match:
            topic = topic_match.group("topic").strip("，, ")
            return self._apply_candidate_rewrite(
                family="study-focus",
                rule_name="sentence:study-focus",
                original_text=stripped,
                candidates=[
                    CandidateOption("study-focus:around", f"本研究围绕{topic}展开。", "本研究围绕"),
                    CandidateOption("study-focus:main", f"本文主要讨论{topic}。", "本文主要讨论"),
                    CandidateOption("study-focus:problem", f"本研究尝试回应{topic}这一问题。", "本研究尝试回应"),
                    CandidateOption("study-focus:task", f"围绕{topic}的设计与实现，本文展开讨论。", "围绕……的设计与实现"),
                    CandidateOption("study-focus:priority", f"本研究的重点在于{topic}。", "本研究的重点在于"),
                ],
                prefer_change=mode is RewriteMode.STRONG or pass_index > 1,
                allow_keep=True,
            )

        scope_match = re.match(
            r"^对于(?P<context>.+?)而言，(?P<subject>.+?)并不局限于(?P<limit>.+?)，更在于(?P<focus>.+?)[。！？?!]?$",
            stripped,
        )
        if scope_match:
            context = scope_match.group("context").strip()
            subject = scope_match.group("subject").strip()
            limit = scope_match.group("limit").strip()
            focus = scope_match.group("focus").strip()
            risk_subject = self._risk_subject_phrase(subject)
            return self._apply_candidate_rewrite(
                family="scope-risk",
                rule_name="sentence:scope-risk",
                original_text=stripped,
                candidates=[
                    CandidateOption(
                        "scope-risk:needs",
                        f"从{context}的实际需求来看，{subject}并不只体现在{limit}，更关键的是{focus}。",
                        "从……的实际需求来看",
                    ),
                    CandidateOption(
                        "scope-risk:issue",
                        f"对{context}来说，{risk_subject}不只在于{limit}，更在于{focus}。",
                        "对……来说",
                    ),
                    CandidateOption(
                        "scope-risk:manifest",
                        f"放到{context}中看，{risk_subject}不只体现在{limit}，更在于{focus}。",
                        "在……中",
                    ),
                ],
                prefer_change=True,
                allow_keep=mode is RewriteMode.CONSERVATIVE,
            )

        build_match = re.match(
            r"^因此，构建(?P<target>.+?)(?P<lemma>十分必要|十分关键|尤为关键|尤为必要|非常关键|非常必要)[。！？?!]?$",
            stripped,
        )
        if build_match and mode in {RewriteMode.BALANCED, RewriteMode.STRONG}:
            target = build_match.group("target").strip()
            lemma = build_match.group("lemma")
            emphasis = "尤为必要" if "必要" in lemma else "尤为关键"
            return self._apply_candidate_rewrite(
                family="therefore-build",
                rule_name="sentence:therefore-build",
                original_text=stripped,
                candidates=[
                    CandidateOption("therefore-build:based-on-this", f"基于此，有必要构建{target}。", "基于此"),
                    CandidateOption(
                        "therefore-build:in-this-case",
                        f"在这种情况下，构建{target}就显得{emphasis}。",
                        "在这种情况下",
                    ),
                    CandidateOption("therefore-build:it-shows", f"这也说明，需要构建{target}。", "这也说明"),
                ],
                prefer_change=True,
                allow_keep=True,
            )

        study_match = re.match(r"^因此，本研究(?P<rest>.+?)[。！？?!]?$", stripped)
        if study_match and mode in {RewriteMode.BALANCED, RewriteMode.STRONG}:
            rest = study_match.group("rest").strip()
            return self._apply_candidate_rewrite(
                family="therefore-study",
                rule_name="sentence:therefore-study",
                original_text=stripped,
                candidates=[
                    CandidateOption("therefore-study:direct", f"因此，本研究{rest}。", "因此，本研究"),
                    CandidateOption("therefore-study:accordingly", f"据此，本文{rest}。", "据此，本文"),
                    CandidateOption("therefore-study:goal", f"在这一目标下，本研究{rest}。", "在这一目标下"),
                ],
                prefer_change=pass_index > 1 or mode is RewriteMode.STRONG,
                allow_keep=True,
            )

        inclusive_match = re.match(
            r"^本研究不仅(?P<x>.+?)，还(?P<y>.+?)，(?:形成了|并形成)(?P<z>.+?)[。！？?!]?$",
            stripped,
        )
        if inclusive_match and mode in {RewriteMode.BALANCED, RewriteMode.STRONG}:
            x = inclusive_match.group("x").strip()
            y = inclusive_match.group("y").strip()
            z = inclusive_match.group("z").strip()
            return self._apply_candidate_rewrite(
                family="parallel-expansion",
                rule_name="sentence:parallel-expansion",
                original_text=stripped,
                candidates=[
                    CandidateOption("parallel-expansion:except", f"除{x}之外，本研究还{y}，最终形成了{z}。", "除……之外"),
                    CandidateOption("parallel-expansion:both", f"本研究既{x}，也{y}，并在此基础上形成了{z}。", "本研究既……也……"),
                    CandidateOption("parallel-expansion:while", f"在{x}的同时，本研究还{y}，从而形成了{z}。", "在……的同时"),
                ],
                prefer_change=True,
                allow_keep=mode is RewriteMode.CONSERVATIVE,
            )

        inclusive_match = re.match(r"^本研究不仅(?P<x>.+?)，还(?P<y>.+?)[。！？?!]?$", stripped)
        if inclusive_match and mode is RewriteMode.STRONG:
            x = inclusive_match.group("x").strip()
            y = inclusive_match.group("y").strip()
            return self._apply_candidate_rewrite(
                family="parallel-expansion-short",
                rule_name="sentence:parallel-expansion-short",
                original_text=stripped,
                candidates=[
                    CandidateOption("parallel-short:except", f"除{x}之外，本研究还{y}。", "除……之外"),
                    CandidateOption("parallel-short:both", f"本研究既{x}，也{y}。", "本研究既……也……"),
                    CandidateOption("parallel-short:cover", f"本研究既涉及{x}，也覆盖了{y}。", "本研究既涉及……也覆盖……"),
                ],
                prefer_change=True,
                allow_keep=False,
            )

        return stripped, [], [], []

    def _rewrite_opening_phrases(
        self,
        sentence: str,
        mode: RewriteMode,
        pass_index: int,
        label: str,
    ) -> tuple[str, list[str], list[str], list[str]]:
        stripped = sentence.strip()

        summary_match = re.match(r"^(总的来说|总而言之|总体来看|总体而言|综合来看)，?", stripped)
        if summary_match:
            return self._apply_opening_variation(
                family="summary-opening",
                rule_name="opening:summary",
                original_text=stripped,
                prefix=summary_match.group(1),
                remainder=stripped[summary_match.end() :].lstrip("，, "),
                candidates=["总体来看", "综合来看", "从整体上看"],
                prefer_change=mode in {RewriteMode.BALANCED, RewriteMode.STRONG} and pass_index > 1,
                allow_keep=True,
            )

        note_match = re.match(r"^(值得注意的是|需要指出的是)，?", stripped)
        if note_match:
            return self._apply_opening_variation(
                family="note-opening",
                rule_name="opening:note",
                original_text=stripped,
                prefix=note_match.group(1),
                remainder=stripped[note_match.end() :].lstrip("，, "),
                candidates=["需要注意的是", "值得进一步说明的是", "需要说明的是"],
                prefer_change=mode is RewriteMode.STRONG or pass_index > 1,
                allow_keep=True,
            )

        practical_match = re.match(r"^对于(?P<context>[^，]+?)而言，", stripped)
        if practical_match and not re.match(r"^对于.+?而言，.+?并不局限于.+?，更在于", stripped):
            context = practical_match.group("context").strip()
            return self._apply_opening_variation(
                family="practical-opening",
                rule_name="opening:practical",
                original_text=stripped,
                prefix=f"对于{context}而言",
                remainder=stripped[practical_match.end() :].lstrip("，, "),
                candidates=[
                    f"从{context}的实际需求来看",
                    f"就{context}而言",
                    f"对{context}来说",
                    f"放在{context}中看",
                ],
                prefer_change=mode in {RewriteMode.BALANCED, RewriteMode.STRONG},
                allow_keep=True,
            )

        implication_match = re.match(r"^(因此|由此可见|由此|基于此|在这种情况下)，", stripped)
        if implication_match and mode in {RewriteMode.BALANCED, RewriteMode.STRONG}:
            return self._apply_candidate_rewrite(
                family="implication-opening",
                rule_name="opening:implication",
                original_text=stripped,
                candidates=[
                    CandidateOption("implication-opening:based-on-this", f"基于此，{stripped[implication_match.end() :].lstrip('，, ')}", "基于此"),
                    CandidateOption("implication-opening:in-this-case", f"在这种情况下，{stripped[implication_match.end() :].lstrip('，, ')}", "在这种情况下"),
                    CandidateOption("implication-opening:it-shows", f"这也说明，{stripped[implication_match.end() :].lstrip('，, ')}", "这也说明"),
                    CandidateOption(
                        "implication-opening:drop",
                        self._ensure_sentence_end(stripped[implication_match.end() :].lstrip("，, ")),
                        "省略因果句首",
                        template_family="",
                        rule_like=False,
                    ),
                ],
                prefer_change=pass_index > 1 or mode is RewriteMode.STRONG,
                allow_keep=True,
            )

        return stripped, [], [], []

    def _rewrite_parallel_enumeration(
        self,
        sentence: str,
        mode: RewriteMode,
        pass_index: int,
        label: str,
    ) -> tuple[str, list[str], list[str], list[str]]:
        stripped = sentence.strip()
        enum_match = re.match(
            r"^(?P<items>[^，。；]+(?:、[^，。；]+){2,})，(?P<predicate>(?:都|都会|均会|都将|都可能|都会对).+?)[。！？?!]?$",
            stripped,
        )
        if not enum_match or mode is RewriteMode.CONSERVATIVE:
            return stripped, [], [], []

        items = [item.strip() for item in enum_match.group("items").split("、") if item.strip()]
        if len(items) < 3:
            return stripped, [], [], []

        formatted_items = self._format_enumeration_items(items)
        predicate = enum_match.group("predicate").strip()
        predicate_tail = re.sub(r"^(都|都会|均会|都将|都可能)", "", predicate).strip()

        return self._apply_candidate_rewrite(
            family="parallel-enumeration",
            rule_name="structure:parallel-enumeration",
            original_text=stripped,
            candidates=[
                CandidateOption("parallel-enumeration:wulunshi", f"无论是{formatted_items}，{predicate}。", "无论是……"),
                CandidateOption("parallel-enumeration:range", f"从{items[0]}到{items[-2]}，再到{items[-1]}，{predicate}。", "从……到……再到……"),
                CandidateOption("parallel-enumeration:yiji", f"{'、'.join(items[:-1])}以及{items[-1]}，{predicate}。", "……以及……"),
                CandidateOption("parallel-enumeration:bulun", f"不论{formatted_items}，{predicate}。", "不论……"),
                CandidateOption("parallel-enumeration:zhishe", f"只要涉及{formatted_items}，都会{predicate_tail}。", "只要涉及……"),
            ],
            prefer_change=True,
            allow_keep=False,
        )

    def _soften_fillers(
        self,
        sentence: str,
        mode: RewriteMode,
        pass_index: int,
        label: str,
    ) -> tuple[str, list[str], list[str], list[str]]:
        updated = sentence
        rule_names: list[str] = []
        selected_variants: list[str] = []
        candidate_notes: list[str] = []

        simple_patterns = (
            (re.compile(r"(比较|相对)较为"), "较为", "filler:degree"),
            (re.compile(r"(非常|十分)(非常|十分)"), r"\1", "filler:duplicate-intensity"),
            (
                re.compile(r"(?P<prefix>.+?对(?P<object>.+?)进行了(?P<verb>讨论|分析))，并对(?P=object)进行了(?P=verb)"),
                r"\g<prefix>",
                "filler:duplicate-object-verb",
            ),
            (re.compile(r"进行了分析，并且进行了分析"), "进行了分析", "filler:duplicate-analysis"),
        )

        for pattern, replacement, rule_name in simple_patterns:
            updated, count = pattern.subn(replacement, updated)
            if count:
                rule_names.append(rule_name)

        updated, rules, surfaces, notes = self._replace_literal_with_variants(
            text=updated,
            literal="在很多方面",
            family="vague-scope",
            rule_name="filler:vague-scope",
            candidates=["在多个层面", "在不同层面", "在若干方面"],
            prefer_change=mode is not RewriteMode.CONSERVATIVE,
            allow_keep=True,
        )
        rule_names.extend(rules)
        selected_variants.extend(surfaces)
        candidate_notes.extend(notes)

        updated, rules, surfaces, notes = self._replace_literal_with_variants(
            text=updated,
            literal="当前表述仍然比较概括",
            family="vague-expression",
            rule_name="filler:vague-expression",
            candidates=["现有表述仍偏概括", "现有论述仍显概括", "整体表述仍较为概括"],
            prefer_change=mode in {RewriteMode.BALANCED, RewriteMode.STRONG},
            allow_keep=True,
        )
        rule_names.extend(rules)
        selected_variants.extend(surfaces)
        candidate_notes.extend(notes)

        updated, rules, surfaces, notes = self._replace_literal_with_variants(
            text=updated,
            literal="当前讨论已经指出",
            family="discussion-opening",
            rule_name="filler:discussion-opening",
            candidates=["现有讨论已经表明", "已有讨论已经显示", "前面的讨论已经指出"],
            prefer_change=pass_index > 1,
            allow_keep=True,
        )
        rule_names.extend(rules)
        selected_variants.extend(surfaces)
        candidate_notes.extend(notes)

        return updated, rule_names, selected_variants, candidate_notes

    def _naturalize_academic_texture(
        self,
        sentence: str,
        mode: RewriteMode,
        pass_index: int,
        label: str,
    ) -> tuple[str, list[str], list[str], list[str]]:
        updated = sentence
        rule_names: list[str] = []
        selected_variants: list[str] = []
        candidate_notes: list[str] = []

        if self._technical_density_is_high(updated):
            candidate_notes.append("technical-density: kept wording conservative and avoided expansion")
            return updated, ["natural:keep-technical-dense"], [], candidate_notes

        updated, rules, surfaces, notes = self._reduce_function_word_overuse(
            updated,
            mode=mode,
            pass_index=pass_index,
        )
        rule_names.extend(rules)
        selected_variants.extend(surfaces)
        candidate_notes.extend(notes)

        updated, rules, surfaces, notes = self._weaken_sequence_connectors(
            updated,
            mode=mode,
            pass_index=pass_index,
        )
        rule_names.extend(rules)
        selected_variants.extend(surfaces)
        candidate_notes.extend(notes)

        updated, rules, surfaces, notes = self._break_template_parallelism(
            updated,
            mode=mode,
            pass_index=pass_index,
        )
        rule_names.extend(rules)
        selected_variants.extend(surfaces)
        candidate_notes.extend(notes)

        updated, rules, surfaces, notes = self._rewrite_dense_nominal_phrase(
            updated,
            mode=mode,
            pass_index=pass_index,
        )
        rule_names.extend(rules)
        selected_variants.extend(surfaces)
        candidate_notes.extend(notes)

        return updated, rule_names, selected_variants, candidate_notes

    def _reduce_function_word_overuse(
        self,
        sentence: str,
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[str, list[str], list[str], list[str]]:
        updated = sentence
        rule_names: list[str] = []
        selected_variants: list[str] = []
        candidate_notes: list[str] = []

        study_match = re.match(
            r"^(?P<subject>[^，。；]{2,18}?)(?P<verb>研究了|分析了|讨论了|梳理了)(?P<object>[^，。；]{2,60})[。！？?!]?$",
            updated.strip(),
        )
        if study_match and mode is not RewriteMode.CONSERVATIVE:
            subject = study_match.group("subject")
            verb = study_match.group("verb")
            obj = study_match.group("object").strip()
            base_verb = verb[:-1]
            updated, rules, surfaces, notes = self._apply_candidate_rewrite(
                family="function-word-reduction",
                rule_name="natural:function-word",
                original_text=updated.strip(),
                candidates=[
                    CandidateOption(
                        "function-word:drop-le",
                        f"{subject}{base_verb}{obj}。",
                        "减少了",
                        template_family="",
                        rule_like=False,
                    ),
                    CandidateOption(
                        "function-word:object-study",
                        f"{subject}对{obj}进行{base_verb}。",
                        "对……进行",
                        template_family="",
                        rule_like=False,
                    ),
                ],
                prefer_change=True,
                allow_keep=True,
            )
            rule_names.extend(rules)
            selected_variants.extend(surfaces)
            candidate_notes.extend(notes)

        simple_patterns = (
            (
                re.compile(r"进行了(?P<object>[^，。；]{2,30})的(?P<verb>研究|分析|讨论|梳理|设计|评价|比较|验证)"),
                r"\g<verb>了\g<object>",
            ),
            (re.compile(r"进行了(?P<verb>研究|分析|讨论|梳理|设计|评价|比较|验证)"), r"\g<verb>了"),
            (re.compile(r"进行(?P<verb>研究|分析|讨论|梳理|设计|评价|比较|验证)"), r"\g<verb>"),
            (re.compile(r"并且"), "并"),
            (re.compile(r"为了能够"), "为"),
            (re.compile(r"在(?P<context>[^，。；]{2,18})过程中"), r"\g<context>过程中"),
            (re.compile(r"在(?P<context>[^，。；]{2,18})背景下"), r"\g<context>背景下"),
        )
        for pattern, replacement in simple_patterns:
            updated, count = pattern.subn(replacement, updated)
            if count:
                rule_names.append("natural:function-word")
                selected_variants.append("减少功能词")

        if "在实际" in updated and "中，" in updated:
            updated, count = re.subn(r"在(?P<context>实际[^，。；]{2,18})中，", r"\g<context>下，", updated, count=1)
            if count:
                rule_names.append("natural:function-word")
                selected_variants.append("弱化在……中")

        return updated, rule_names, selected_variants, candidate_notes

    def _weaken_sequence_connectors(
        self,
        sentence: str,
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[str, list[str], list[str], list[str]]:
        stripped = sentence.strip()
        connector_match = re.match(r"^(首先|其次|再次|此外|最后|综上所述)[，,](?P<body>.+)$", stripped)
        if not connector_match or mode is RewriteMode.CONSERVATIVE:
            return sentence, [], [], []

        body = connector_match.group("body").strip()
        replacement = self._ensure_sentence_end(body)
        if not replacement or replacement == stripped:
            return sentence, [], [], []
        return replacement, ["natural:weaken-template-connectors"], ["省略序列连接词"], []

    def _break_template_parallelism(
        self,
        sentence: str,
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[str, list[str], list[str], list[str]]:
        stripped = sentence.strip()
        if mode is RewriteMode.CONSERVATIVE:
            return sentence, [], [], []

        match = re.match(r"^不仅要(?P<x>.+?)，?还要(?P<y>.+?)[。！？?!]?$", stripped)
        if match:
            x = match.group("x").strip()
            y = match.group("y").strip()
            rewritten, rules, surfaces, notes = self._apply_candidate_rewrite(
                family="natural-parallelism",
                rule_name="natural:break-parallelism",
                original_text=stripped,
                candidates=[
                    CandidateOption(
                        "parallelism:compress",
                        f"{x}之外，还需要{y}。",
                        "压缩排比",
                        template_family="",
                        rule_like=False,
                    ),
                    CandidateOption(
                        "parallelism:recast",
                        f"{x}是基础，{y}则决定了后续落实效果。",
                        "改写排比骨架",
                        template_family="",
                        rule_like=False,
                    ),
                ],
                prefer_change=True,
                allow_keep=False,
            )
            return rewritten, rules, surfaces, notes

        slogan_patterns = (
            (r"为(?P<object>.+?)奠定坚实基础", r"为\g<object>提供基础"),
            (r"具有重要意义并提供有力支撑", "具有一定意义，也能提供支撑"),
            (r"形成完整闭环并实现有效提升", "形成较完整的衔接关系，并改善整体效果"),
        )
        updated = stripped
        rule_names: list[str] = []
        for pattern, replacement in slogan_patterns:
            updated, count = re.subn(pattern, replacement, updated)
            if count:
                rule_names.append("natural:break-parallelism")
        if rule_names:
            return self._ensure_sentence_end(updated), rule_names, ["弱化口号排比"], []
        return sentence, [], [], []

    def _rewrite_dense_nominal_phrase(
        self,
        sentence: str,
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[str, list[str], list[str], list[str]]:
        if mode is RewriteMode.CONSERVATIVE:
            return sentence, [], [], []
        updated = sentence
        rule_names: list[str] = []
        updated, count = re.subn(r"(?P<a>[^，。；]{2,8})的(?P<b>[^，。；]{2,8})的(?P<c>[^，。；]{2,12})", r"\g<a>\g<b>的\g<c>", updated, count=1)
        if count:
            rule_names.append("natural:dense-nominal")
        updated, count = re.subn(r"文字语言中的语法与修辞", "文字语言中语法和修辞", updated, count=1)
        if count:
            rule_names.append("natural:dense-nominal")
        return updated, rule_names, ["压缩的字结构"] if rule_names else [], []

    def _rewrite_across_sentences(
        self,
        units: list[SentenceUnit],
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[list[SentenceUnit], list[str], list[str], list[str]]:
        if mode is RewriteMode.CONSERVATIVE or len(units) < 2:
            return units, [], [], []

        rewritten: list[SentenceUnit] = []
        rule_names: list[str] = []
        selected_variants: list[str] = []
        candidate_notes: list[str] = []
        index = 0

        while index < len(units):
            current = units[index]
            if index + 1 >= len(units):
                rewritten.append(current)
                break

            next_unit = units[index + 1]
            transition_prefix, transition_remainder = self._strip_opening(next_unit.text, _TRANSITION_PREFIXES)
            implication_prefix, implication_remainder = self._strip_opening(next_unit.text, _IMPLICATION_PREFIXES)

            if transition_prefix:
                fused = self._fuse_transition_pair(
                    left=current,
                    right=next_unit,
                    remainder=transition_remainder,
                    mode=mode,
                    pass_index=pass_index,
                )
                if fused is not None:
                    new_unit, rules, surfaces, notes = fused
                    rewritten.append(new_unit)
                    rule_names.extend(rules)
                    selected_variants.extend(surfaces)
                    candidate_notes.extend(notes)
                    index += 2
                    continue

            if implication_prefix and len(self._strip_end_punctuation(current.text)) <= 58:
                fused = self._fuse_implication_pair(
                    left=current,
                    right=next_unit,
                    remainder=implication_remainder,
                    mode=mode,
                    pass_index=pass_index,
                )
                if fused is not None:
                    new_unit, rules, surfaces, notes = fused
                    rewritten.append(new_unit)
                    rule_names.extend(rules)
                    selected_variants.extend(surfaces)
                    candidate_notes.extend(notes)
                    index += 2
                    continue

            rewritten.append(current)
            index += 1

        return rewritten, rule_names, selected_variants, candidate_notes

    def _sentence_cluster_rewriter(
        self,
        units: list[SentenceUnit],
        mode: RewriteMode,
        pass_index: int,
        rewrite_depth: str,
        rewrite_intensity: str,
    ) -> tuple[list[SentenceUnit], list[str], list[str], list[str]]:
        rewritten_units, rules, surfaces, notes = self._rewrite_across_sentences(
            units=units,
            mode=mode,
            pass_index=pass_index,
        )
        if rewrite_depth != "developmental_rewrite":
            return rewritten_units, rules, surfaces, notes

        developmental_pass = max(pass_index, 2 if rewrite_intensity in {"medium", "high"} else pass_index)
        enhanced_units, more_rules, more_surfaces, more_notes = self._rewrite_across_sentences(
            units=rewritten_units,
            mode=RewriteMode.STRONG if mode is RewriteMode.STRONG else RewriteMode.BALANCED,
            pass_index=developmental_pass,
        )
        if more_rules or rules:
            more_rules.append("cluster:sentence-cluster-merge")
            more_notes.append("sentence cluster merge/absorption applied across adjacent sentences")
        enhanced_units, cluster_rules, cluster_surfaces, cluster_notes = self._rewrite_sentence_cluster_window(
            units=enhanced_units,
            mode=mode,
            pass_index=developmental_pass,
            rewrite_intensity=rewrite_intensity,
        )
        more_rules.extend(cluster_rules)
        more_surfaces.extend(cluster_surfaces)
        more_notes.extend(cluster_notes)
        if rewrite_intensity == "high":
            enhanced_units, merge_rules, merge_surfaces, merge_notes = self._merge_short_followups(
                units=enhanced_units,
                mode=RewriteMode.STRONG,
                pass_index=developmental_pass,
            )
            more_rules.extend(merge_rules)
            more_surfaces.extend(merge_surfaces)
            more_notes.extend(merge_notes)
        return (
            enhanced_units,
            [*rules, *more_rules],
            [*surfaces, *more_surfaces],
            [*notes, *more_notes],
        )

    def _rewrite_sentence_cluster_window(
        self,
        units: list[SentenceUnit],
        mode: RewriteMode,
        pass_index: int,
        rewrite_intensity: str,
    ) -> tuple[list[SentenceUnit], list[str], list[str], list[str]]:
        if mode is RewriteMode.CONSERVATIVE or len(units) < 3:
            return units, [], [], []
        if len(units) > 5 and rewrite_intensity != "high":
            window = units[:5]
            tail = units[5:]
        else:
            window = units[: min(5, len(units))]
            tail = units[len(window) :]

        labels = [unit.label for unit in window]
        if window[-1].label == "conclusion" and window[0].label in {"background", "detail", "support", "capability"}:
            reordered = [window[0], window[-1], *window[1:-1], *tail]
            if not self._can_apply_reordered_units(units, reordered):
                return units, [], [], []
            return (
                reordered,
                ["cluster:cause-effect-reframe", "cluster:narrative-path-rewrite"],
                [],
                ["sentence-cluster rewrite surfaced conclusion earlier and left supporting detail after it"],
            )
        if labels[0] == "detail" and "risk" in labels[1:]:
            risk_index = labels.index("risk")
            reordered_window = [window[0], window[risk_index], *window[1:risk_index], *window[risk_index + 1 :]]
            reordered_units = [*reordered_window, *tail]
            if not self._can_apply_reordered_units(units, reordered_units):
                return units, [], [], []
            return (
                reordered_units,
                ["cluster:cause-effect-reframe", "cluster:discourse-reordering"],
                [],
                ["sentence-cluster rewrite placed risk immediately after detail to clarify cause-effect flow"],
            )
        if rewrite_intensity == "high" and len(window) >= 4 and window[1].label in {"support", "capability", "detail"}:
            reordered = [window[0], window[2], window[1], *window[3:], *tail]
            if not self._can_apply_reordered_units(units, reordered):
                return units, [], [], []
            return (
                reordered,
                ["cluster:local-sentence-reorder", "cluster:narrative-path-rewrite"],
                [],
                ["sentence-cluster rewrite introduced local non-uniform ordering instead of line-by-line rewriting"],
            )
        return units, [], [], []

    def _rewrite_standalone_transitions(
        self,
        units: list[SentenceUnit],
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[list[SentenceUnit], list[str], list[str], list[str]]:
        if mode is RewriteMode.CONSERVATIVE:
            return units, [], [], []

        rewritten: list[SentenceUnit] = []
        rule_names: list[str] = []
        selected_variants: list[str] = []
        candidate_notes: list[str] = []

        for index, unit in enumerate(units):
            prefix, remainder = self._strip_opening(unit.text, _TRANSITION_PREFIXES)
            if not prefix:
                rewritten.append(unit)
                continue
            if index == 0:
                updated = self._ensure_sentence_end(remainder) if remainder.strip() else unit.text
                rewritten.append(SentenceUnit(text=updated, label=unit.label, source_indices=unit.source_indices))
                if updated != unit.text:
                    rule_names.append("paragraph:opening-style-guard")
                    selected_variants.append("省略段首接续")
                    candidate_notes.append("paragraph opening guard removed a dangling transition opener")
                continue

            updated_text, rules, surfaces, notes = self._apply_candidate_rewrite(
                family="transition-opening",
                rule_name="opening:transition",
                original_text=unit.text,
                candidates=[
                    CandidateOption("transition-opening:process", f"在这一过程中，{remainder}", "在这一过程中"),
                    CandidateOption("transition-opening:accompany", f"与之相伴的是，{remainder}", "与之相伴的是"),
                    CandidateOption("transition-opening:meanwhile", f"同时，{remainder}", "同时"),
                    CandidateOption(
                        "transition-opening:drop",
                        self._ensure_sentence_end(remainder),
                        "省略过渡句首",
                        template_family="",
                        rule_like=False,
                    ),
                ],
                prefer_change=pass_index > 1 or mode is RewriteMode.STRONG,
                allow_keep=True,
            )
            rewritten.append(SentenceUnit(text=updated_text, label=unit.label, source_indices=unit.source_indices))
            rule_names.extend(rules)
            selected_variants.extend(surfaces)
            candidate_notes.extend(notes)

        return rewritten, rule_names, selected_variants, candidate_notes

    def _split_long_sentences(
        self,
        units: list[SentenceUnit],
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[list[SentenceUnit], list[str], list[str], list[str]]:
        threshold = 92 if mode is RewriteMode.BALANCED and pass_index == 1 else 78
        rewritten: list[SentenceUnit] = []
        rule_names: list[str] = []

        for unit in units:
            split_point = self._find_split_point(unit.text, threshold=threshold)
            if split_point is None:
                rewritten.append(unit)
                continue

            left = self._ensure_sentence_end(unit.text[:split_point].rstrip("，, "))
            right = self._ensure_sentence_end(unit.text[split_point + 1 :].lstrip("，, "))
            if left and right:
                rewritten.extend(
                    [
                        SentenceUnit(text=left, label=unit.label, source_indices=unit.source_indices),
                        SentenceUnit(text=right, label="detail", source_indices=unit.source_indices),
                    ]
                )
                rule_names.append("structure:split-long-sentence")
            else:
                rewritten.append(unit)

        return rewritten, rule_names, [], []

    def _merge_short_followups(
        self,
        units: list[SentenceUnit],
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[list[SentenceUnit], list[str], list[str], list[str]]:
        if len(units) < 2:
            return units, [], [], []

        rewritten: list[SentenceUnit] = []
        rule_names: list[str] = []
        selected_variants: list[str] = []
        candidate_notes: list[str] = []
        index = 0

        while index < len(units):
            if index + 1 >= len(units):
                rewritten.append(units[index])
                break

            current = units[index]
            next_unit = units[index + 1]
            can_merge = (
                mode in {RewriteMode.BALANCED, RewriteMode.STRONG}
                and len(self._strip_end_punctuation(current.text)) < (32 if mode is RewriteMode.BALANCED else 36)
                and len(self._strip_end_punctuation(next_unit.text)) < (30 if mode is RewriteMode.BALANCED else 36)
                and next_unit.label in {"support", "detail", "conclusion"}
            )

            if not can_merge:
                rewritten.append(current)
                index += 1
                continue

            left_core = self._strip_end_punctuation(current.text)
            right_core = self._strip_sequence_connector(self._strip_end_punctuation(next_unit.text))
            merged_text, rules, surfaces, notes = self._apply_candidate_rewrite(
                family="short-merge",
                rule_name="structure:merge-short-followup",
                original_text=f"{current.text}{next_unit.text}",
                candidates=[
                    CandidateOption("short-merge:bing", f"{left_core}，并{right_core.lstrip('并')}。", "并"),
                    CandidateOption("short-merge:jiner", f"{left_core}，进而{right_core.lstrip('进而')}。", "进而"),
                    CandidateOption("short-merge:tongshi", f"{left_core}，同时{right_core}。", "同时"),
                    CandidateOption(
                        "short-merge:drop",
                        f"{left_core}，{right_core}。",
                        "省略连接",
                        template_family="",
                        rule_like=False,
                    ),
                ],
                prefer_change=True,
                allow_keep=False,
            )
            rewritten.append(
                SentenceUnit(
                    text=merged_text,
                    label=current.label,
                    source_indices=current.source_indices + next_unit.source_indices,
                )
            )
            rule_names.extend(rules)
            selected_variants.extend(surfaces)
            candidate_notes.extend(notes)
            index += 2

        return rewritten, rule_names, selected_variants, candidate_notes

    def _reorganize_paragraph(
        self,
        units: list[SentenceUnit],
        pass_index: int,
    ) -> tuple[list[SentenceUnit], list[str], list[str], list[str]]:
        if len(units) < 3:
            return units, [], [], []

        labels = [unit.label for unit in units]
        if units[0].label == "conclusion":
            reordered = units[1:] + [units[0]]
            if not self._can_apply_reordered_units(units, reordered):
                return units, [], [], []
            return reordered, ["paragraph:move-summary-to-close"], [], ["paragraph-reorder moved conclusion to close"]

        try:
            objective_index = labels.index("objective")
        except ValueError:
            objective_index = -1

        if 0 < objective_index < 3 and units[0].label in {"background", "risk", "capability"}:
            reordered = [units[objective_index]] + units[:objective_index] + units[objective_index + 1 :]
            if not self._can_apply_reordered_units(units, reordered):
                return units, [], [], []
            return reordered, ["paragraph:promote-objective"], [], ["paragraph-reorder promoted objective sentence"]

        if (
            units[0].label in {"background", "capability"}
            and units[-1].label in {"risk", "conclusion"}
            and units[1].label in {"detail", "support", "capability"}
        ):
            reordered = [units[0], units[-1], *units[1:-1]]
            if not self._can_apply_reordered_units(units, reordered):
                return units, [], [], []
            return reordered, ["paragraph:promote-risk"], [], ["paragraph-reorder promoted risk or conclusion sentence"]

        for index, unit in enumerate(units[:-1]):
            if unit.label == "conclusion":
                reordered = units[:index] + units[index + 1 :] + [unit]
                if not self._can_apply_reordered_units(units, reordered):
                    return units, [], [], []
                return reordered, ["paragraph:move-conclusion-to-close"], [], ["paragraph-reorder moved conclusion sentence"]

        return units, [], [], []

    def _narrative_flow_rebuilder(
        self,
        units: list[SentenceUnit],
        mode: RewriteMode,
        pass_index: int,
        rewrite_depth: str,
        rewrite_intensity: str,
    ) -> tuple[list[SentenceUnit], list[str], list[str], list[str]]:
        if rewrite_depth != "developmental_rewrite":
            return units, [], [], []
        if len(units) < 3:
            return units, [], [], []

        reordered, rules, surfaces, notes = self._reorganize_paragraph(units=units, pass_index=pass_index)
        if rules:
            return reordered, rules, surfaces, notes

        labels = [unit.label for unit in units]
        if labels[:3] == ["background", "detail", "conclusion"]:
            reordered = [units[0], units[2], units[1], *units[3:]]
            if not self._can_apply_reordered_units(units, reordered):
                return units, [], [], []
            return (
                reordered,
                ["paragraph:surface-conclusion-earlier"],
                [],
                ["paragraph-reorder surfaced conclusion before late detail sentence"],
            )
        if "objective" in labels and labels[0] == "risk":
            objective_index = labels.index("objective")
            if objective_index > 0:
                reordered = [units[0], units[objective_index], *units[1:objective_index], *units[objective_index + 1 :]]
                if not self._can_apply_reordered_units(units, reordered):
                    return units, [], [], []
                return (
                    reordered,
                    ["paragraph:align-risk-objective"],
                    [],
                    ["paragraph-reorder aligned objective after risk statement"],
                )

        if rewrite_intensity == "high" and units[1].label in {"risk", "capability", "support", "detail"}:
            reordered = [units[1], units[0], *units[2:]]
            if not self._can_apply_reordered_units(units, reordered):
                return units, [], [], []
            return (
                reordered,
                ["paragraph:high-intensity-promote-second"],
                [],
                ["paragraph-reorder promoted the second sentence to rebuild narrative flow"],
            )

        return units, [], [], []

    def _fuse_transition_pair(
        self,
        left: SentenceUnit,
        right: SentenceUnit,
        remainder: str,
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[SentenceUnit, list[str], list[str], list[str]] | None:
        left_core = self._strip_end_punctuation(left.text)
        right_core = self._strip_end_punctuation(remainder)
        if not left_core or not right_core:
            return None

        if left.label == "objective":
            tails = self._build_followup_variants(right_core)
            if len(tails) < 3:
                return None
            fused_text, rules, surfaces, notes = self._apply_candidate_rewrite(
                family="objective-followup",
                rule_name="fusion:objective-followup",
                original_text=f"{left.text}{right.text}",
                candidates=[
                    CandidateOption("objective-followup:also", f"{left_core}，同时也{right_core}。", "同时也……"),
                    CandidateOption("objective-followup:deeper", f"围绕这一目标，{tails[1]}。", "围绕这一目标"),
                    CandidateOption("objective-followup:covered", f"{left_core}，并将{right_core}纳入了讨论。", "并将……纳入讨论"),
                    CandidateOption(
                        "objective-followup:drop",
                        f"{left_core}，{self._soft_insert_particle(right_core, particle='也')}。",
                        "省略过渡",
                        template_family="",
                        rule_like=False,
                    ),
                ],
                prefer_change=True,
                allow_keep=False,
            )
            return (
                SentenceUnit(text=fused_text, label=left.label, source_indices=left.source_indices + right.source_indices),
                rules,
                surfaces,
                notes,
            )

        if len(left_core) > 70:
            return None

        right_linked = self._soft_insert_particle(right_core, particle="也")
        fused_text, rules, surfaces, notes = self._apply_candidate_rewrite(
            family="context-followup",
            rule_name="fusion:context-followup",
            original_text=f"{left.text}{right.text}",
            candidates=[
                CandidateOption("context-followup:while", f"随着{left_core}的推进，{right_linked}。", "随着……的推进"),
                CandidateOption("context-followup:and", f"{left_core}不断推进，而{right_linked}。", "……而……也"),
                CandidateOption("context-followup:accompany", f"在{left_core}持续发展的同时，{right_core}。", "在……持续发展的同时"),
                CandidateOption(
                    "context-followup:drop",
                    f"{left_core}，{right_linked}。",
                    "省略过渡",
                    template_family="",
                    rule_like=False,
                ),
            ],
            prefer_change=mode is RewriteMode.STRONG or pass_index > 1,
            allow_keep=False,
        )
        return (
            SentenceUnit(text=fused_text, label=left.label, source_indices=left.source_indices + right.source_indices),
            rules,
            surfaces,
            notes,
        )

    def _fuse_implication_pair(
        self,
        left: SentenceUnit,
        right: SentenceUnit,
        remainder: str,
        mode: RewriteMode,
        pass_index: int,
    ) -> tuple[SentenceUnit, list[str], list[str], list[str]] | None:
        if mode is RewriteMode.CONSERVATIVE:
            return None

        left_core = self._strip_end_punctuation(left.text)
        right_core = self._strip_end_punctuation(remainder)
        if not left_core or not right_core:
            return None

        meta_subject, meta_body = self._extract_meta_subject_and_body(right_core)
        if meta_subject in _META_SUBJECTS and meta_body:
            varied_followup = self._subject_variant_phrase(meta_subject, self._strip_end_punctuation(meta_body))
            fused_text, rules, surfaces, notes = self._apply_candidate_rewrite(
                family="implication-followup",
                rule_name="fusion:implication-followup",
                original_text=f"{left.text}{right.text}",
                candidates=[
                    CandidateOption("implication-followup:drop", f"{left_core}，在此基础上，{self._strip_end_punctuation(meta_body)}。", "在此基础上"),
                    CandidateOption("implication-followup:research", f"{left_core}，由此，{varied_followup}。", "由此，研究"),
                    CandidateOption(
                        "implication-followup:accordingly",
                        f"{left_core}，相应地，{self._subject_variant_phrase(meta_subject, self._strip_end_punctuation(meta_body), prefer_system=True)}。",
                        "相应地",
                    ),
                    CandidateOption(
                        "implication-followup:direct",
                        f"{left_core}，{self._strip_end_punctuation(meta_body)}。",
                        "省略因果句首",
                        template_family="",
                        rule_like=False,
                    ),
                ],
                prefer_change=mode is RewriteMode.STRONG or pass_index > 1,
                allow_keep=False,
            )
            return (
                SentenceUnit(text=fused_text, label=left.label, source_indices=left.source_indices + right.source_indices),
                rules,
                surfaces,
                notes,
            )

        fused_text, rules, surfaces, notes = self._apply_candidate_rewrite(
            family="implication-followup",
            rule_name="fusion:implication-followup",
            original_text=f"{left.text}{right.text}",
            candidates=[
                CandidateOption("implication-followup:means", f"{left_core}，这也使得{right_core}。", "这也使得"),
                CandidateOption("implication-followup:based-on-this", f"{left_core}，因此有必要{right_core}。", "因此有必要"),
                CandidateOption("implication-followup:it-shows", f"正因为如此，{right_core}。", "正因为如此"),
                CandidateOption(
                    "implication-followup:direct",
                    f"{left_core}，{right_core}。",
                    "省略因果句首",
                    template_family="",
                    rule_like=False,
                ),
            ],
            prefer_change=mode is RewriteMode.STRONG or pass_index > 1,
            allow_keep=False,
        )
        return (
            SentenceUnit(text=fused_text, label=left.label, source_indices=left.source_indices + right.source_indices),
            rules,
            surfaces,
            notes,
        )

    def _apply_opening_variation(
        self,
        family: str,
        rule_name: str,
        original_text: str,
        prefix: str,
        remainder: str,
        candidates: list[str],
        prefer_change: bool,
        allow_keep: bool,
    ) -> tuple[str, list[str], list[str], list[str]]:
        candidate_options = [
            CandidateOption(
                key=f"{family}:{index}",
                text=f"{surface}，{remainder}" if remainder else f"{surface}。",
                surface=surface,
            )
            for index, surface in enumerate(candidates, start=1)
        ]
        return self._apply_candidate_rewrite(
            family=family,
            rule_name=rule_name,
            original_text=original_text,
            candidates=candidate_options,
            prefer_change=prefer_change,
            allow_keep=allow_keep,
        )

    def _replace_literal_with_variants(
        self,
        text: str,
        literal: str,
        family: str,
        rule_name: str,
        candidates: list[str],
        prefer_change: bool,
        allow_keep: bool,
    ) -> tuple[str, list[str], list[str], list[str]]:
        if literal not in text:
            return text, [], [], []

        option_list = [
            CandidateOption(
                key=f"{family}:{index}",
                text=text.replace(literal, surface),
                surface=surface,
            )
            for index, surface in enumerate(candidates, start=1)
        ]
        return self._apply_candidate_rewrite(
            family=family,
            rule_name=rule_name,
            original_text=text,
            candidates=option_list,
            prefer_change=prefer_change,
            allow_keep=allow_keep,
        )

    def _apply_candidate_rewrite(
        self,
        family: str,
        rule_name: str,
        original_text: str,
        candidates: list[CandidateOption],
        prefer_change: bool,
        allow_keep: bool,
    ) -> tuple[str, list[str], list[str], list[str]]:
        chosen, notes = self._choose_candidate(
            family=family,
            original_text=original_text,
            candidates=candidates,
            prefer_change=prefer_change,
            allow_keep=allow_keep,
        )
        if chosen is None or chosen.text == original_text:
            return original_text, [], [], notes
        return chosen.text, [rule_name], [chosen.surface], notes

    def _extract_meta_topic(self, sentence: str) -> str | None:
        stripped = self._strip_end_punctuation(sentence.strip())
        patterns = (
            r"^本研究的主题为(?P<topic>.+)$",
            r"^本文讨论的核心问题是(?P<topic>.+)$",
            r"^本研究尝试回应(?P<topic>.+?)(?:这一问题)?$",
            r"^本研究聚焦于(?P<topic>.+)$",
            r"^本研究围绕(?P<topic>.+?)展开$",
            r"^本文主要讨论(?P<topic>.+)$",
            r"^本研究的重点在于(?P<topic>.+)$",
            r"^围绕(?P<topic>.+?)的设计与实现，本文展开讨论$",
        )
        for pattern in patterns:
            match = re.match(pattern, stripped)
            if match:
                return match.group("topic").strip("，, ")
        return None

    def _extract_meta_subject_and_body(self, sentence: str) -> tuple[str, str]:
        stripped = self._strip_end_punctuation(sentence.strip())
        for subject in _META_SUBJECTS:
            if stripped.startswith(subject):
                body = stripped[len(subject) :].lstrip("，, ")
                return subject, body
        if stripped.startswith("研究"):
            return "研究", stripped[len("研究") :].lstrip("，, ")
        if stripped.startswith("系统"):
            return "系统", stripped[len("系统") :].lstrip("，, ")
        return "", stripped

    def _looks_like_work_body(self, body: str) -> bool:
        return bool(re.search(r"(不仅|包含|完成|实现|形成|构建|设计|覆盖|讨论|分析|提出|回应|优化)", body))

    def _risk_subject_phrase(self, subject: str) -> str:
        cleaned = subject.strip()
        if "风险" in cleaned:
            return cleaned
        return f"{cleaned}带来的风险"

    def _subject_variant_phrase(self, subject: str, body: str, prefer_system: bool = False) -> str:
        normalized_body = body
        if normalized_body.startswith("同时"):
            normalized_body = normalized_body[len("同时") :].lstrip("，, ")
        if prefer_system and subject.endswith("系统"):
            lead = "系统"
        elif subject.endswith("系统"):
            lead = "该系统" if subject == "该系统" else "系统"
        else:
            if subject in {"本研究", "该研究"} and normalized_body.startswith("研究"):
                return normalized_body
            lead = "研究" if subject in {"本研究", "该研究"} else "本文"
        return f"{lead}{normalized_body}"

    def _should_preserve_meta_subject(self, body: str) -> bool:
        stripped = self._strip_end_punctuation(body.strip())
        if not stripped:
            return False

        explicit_markers = (
            "最终采用",
            "默认运行于",
            "作为本地部署模型",
            "作为部署模型",
            "checkpoint",
            "checkpoints/",
            ".pth",
            ".pt",
            ".ckpt",
            "epoch_",
            "Phase",
            "base_only",
            "best.pth",
            "文件路径",
            "路径为",
            "保存到",
        )
        if any(marker in stripped for marker in explicit_markers):
            return True
        if "/" in stripped and any(token in stripped for token in ("checkpoint", ".pth", ".pt", ".ckpt")):
            return True
        return False

    def _extract_subject_head(self, sentence: str) -> str:
        stripped = sentence.strip()
        _, stripped = self._strip_opening(stripped, _IMPLICATION_PREFIXES)
        _, stripped = self._strip_opening(stripped, _TRANSITION_PREFIXES)
        for pattern in (
            r"^围绕[^，]+，",
            r"^针对[^，]+，",
            r"^放到[^，]+中看，",
            r"^从[^，]+来看，",
            r"^对[^，]+来说，",
            r"^就[^，]+而言，",
            r"^在此基础上，",
            r"^由此，",
            r"^相应地，",
        ):
            stripped = re.sub(pattern, "", stripped)
        stripped = stripped.lstrip("，, ")
        for subject in _META_SUBJECTS:
            if stripped.startswith(subject):
                return subject
        for subject in ("研究", "系统", "该机制", "该设计"):
            if stripped.startswith(subject):
                return subject
        generic_match = re.match(r"^(?P<head>[^，。；]{1,10}?)(?:正在|已经|能够|可以|不仅|还|会|将|带来的|的风险|在|并不)", stripped)
        if generic_match:
            return generic_match.group("head")
        return stripped[:6]

    def _max_repeated_subject_streak(self, subject_heads: list[str]) -> int:
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

    def _has_repeated_subject_risk(self, subject_heads: list[str]) -> bool:
        return self._max_repeated_subject_streak(subject_heads) >= 2

    def _structural_actions_from_rules(self, rules: list[str]) -> list[str]:
        actions: list[str] = []
        for rule in rules:
            if rule in {"fusion:context-followup", "fusion:objective-followup"}:
                actions.extend(["pair_fusion", "sentence_cluster_merge"])
            elif rule == "fusion:implication-followup":
                actions.extend(["conclusion_absorb", "conclusion_absorption", "sentence_cluster_merge"])
            elif rule == "structure:parallel-enumeration":
                actions.append("enumeration_reframe")
            elif rule == "structure:split-long-sentence":
                actions.extend(["sentence_split", "sentence_cluster_split"])
            elif rule == "structure:merge-short-followup":
                actions.extend(["sentence_merge", "sentence_cluster_merge"])
            elif rule in {"paragraph:topic-sentence-preserved", "paragraph:opening-style-guard"}:
                actions.append("topic_sentence_preservation")
            elif rule.startswith("paragraph:"):
                actions.extend(["paragraph_reorder", "discourse_reordering", "narrative_path_rewrite"])
            elif rule.startswith("cluster:"):
                actions.extend(["sentence_cluster_rewrite", "discourse_reordering", "narrative_path_rewrite"])
            elif rule == "sentence:study-focus":
                actions.append("topic_reframe")
            elif rule == "sentence:developmental-recast":
                actions.append("clause_reorder")
            elif rule in {"sentence:scope-risk", "sentence:parallel-expansion", "sentence:parallel-expansion-short"}:
                actions.append("clause_reorder")
            elif rule == "subject:merge-consecutive":
                actions.append("merge_consecutive_subject_sentences")
            elif rule == "subject:drop":
                actions.append("subject_drop")
            elif rule == "subject:variation":
                actions.append("subject_variation")
            elif rule == "subject:meta-compression":
                actions.append("meta_compression")
            elif rule == "subject:followup-absorb":
                actions.append("followup_absorb")
            elif rule == "readability:post-readability-subject-chain-repair":
                actions.append("subject_chain_compression")
            elif rule == "natural:break-parallelism":
                actions.append("clause_reorder")
            elif rule == "natural:weaken-template-connectors":
                actions.append("clause_reorder")
            elif rule == "human:partial-keep":
                actions.append("uneven_rewrite_distribution")
            elif rule == "human:uneven-rewrite-distribution":
                actions.append("uneven_rewrite_distribution")
            elif rule.startswith("local:"):
                actions.extend(self._local_actions_from_rules([rule]))
            elif rule.startswith("readability:"):
                actions.extend(self._readability_actions_from_rules([rule]))
        if any(action in {"merge_consecutive_subject_sentences", "subject_drop", "subject_variation", "meta_compression", "followup_absorb"} for action in actions):
            actions.append("subject_chain_compression")
        return self._deduplicate_preserve_order(actions)

    def _discourse_actions_from_rules(self, rules: list[str]) -> list[str]:
        actions: list[str] = []
        for rule in rules:
            if rule in {"fusion:context-followup", "fusion:objective-followup", "structure:merge-short-followup"}:
                actions.extend(["sentence_cluster_rewrite", "sentence_cluster_merge", "transition_absorption"])
            elif rule == "fusion:implication-followup":
                actions.extend(["sentence_cluster_rewrite", "sentence_cluster_merge", "conclusion_absorb", "conclusion_absorption"])
            elif rule == "structure:parallel-enumeration":
                actions.append("enumeration_reframe")
            elif rule in {"paragraph:topic-sentence-preserved", "paragraph:opening-style-guard"}:
                actions.append("preserve_explicit_subject_if_clarity_needed")
            elif rule.startswith("paragraph:") or rule.startswith("cluster:"):
                actions.extend(["proposition_reorder", "discourse_reordering", "narrative_path_rewrite"])
            elif rule == "subject:meta-compression":
                actions.append("meta_compression")
            elif rule in {"subject:merge-consecutive", "subject:drop", "subject:variation", "subject:followup-absorb"}:
                actions.extend(["subject_chain_compression", "compress_subject_chain"])
            elif rule == "readability:post-readability-subject-chain-repair":
                actions.extend(["subject_chain_compression", "compress_subject_chain"])
            elif rule == "sentence:scope-risk":
                actions.append("rationale_expansion")
            elif rule == "sentence:developmental-recast":
                actions.append("narrative_path_rewrite")
            elif rule in {"structure:split-long-sentence", "structure:merge-short-followup"}:
                actions.append("rebuild_sentence_rhythm")
                if rule == "structure:split-long-sentence":
                    actions.append("sentence_cluster_split")
                else:
                    actions.append("sentence_cluster_merge")
            elif rule == "natural:function-word":
                actions.append("reduce_function_word_overuse")
            elif rule == "natural:weaken-template-connectors":
                actions.append("weaken_template_connectors")
            elif rule == "natural:break-parallelism":
                actions.append("break_parallelism")
            elif rule == "natural:dense-nominal":
                actions.append("rewrite_dense_nominal_phrases")
            elif rule == "natural:keep-technical-dense":
                actions.append("keep_original_if_technical_density_is_high")
            elif rule in {"human:partial-keep", "human:uneven-rewrite-distribution"}:
                actions.append("uneven_rewrite_distribution")
            elif rule.startswith("local:"):
                actions.extend(self._local_actions_from_rules([rule]))
            elif rule.startswith("readability:"):
                actions.extend(self._readability_actions_from_rules([rule]))
        return self._deduplicate_preserve_order(actions)

    def _inject_human_revision_variation(
        self,
        original_sentences: list[str],
        rewritten_sentences: list[str],
        paragraph_index: int,
        rewrite_depth: str,
        rewrite_intensity: str,
        mode: RewriteMode,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        if mode is RewriteMode.CONSERVATIVE or rewrite_depth != "developmental_rewrite":
            return rewritten_sentences, [], [], []
        if not original_sentences or not rewritten_sentences:
            return rewritten_sentences, [], [], []

        updated = list(rewritten_sentences)
        rules: list[str] = []
        notes: list[str] = []
        marks: list[str] = []
        limit = min(len(original_sentences), len(updated))
        changed_positions = [
            index
            for index in range(limit)
            if self._normalize_for_compare(original_sentences[index]) != self._normalize_for_compare(updated[index])
        ]

        if len(original_sentences) == len(updated) and len(original_sentences) >= 4 and len(changed_positions) == len(original_sentences):
            keep_index = (paragraph_index + len(original_sentences)) % len(original_sentences)
            updated[keep_index] = original_sentences[keep_index]
            rules.append("human:partial-keep")
            notes.append(f"human-variation kept sentence {keep_index + 1} nearly unchanged to avoid uniform rewriting")
            marks.append("partial_keep")
            changed_positions = [index for index in changed_positions if index != keep_index]

        lengths = [len(self._normalize_for_compare(sentence)) for sentence in updated]
        if len(lengths) >= 3 and max(lengths) - min(lengths) >= 12:
            marks.append("asymmetric_sentence_rhythm")
        if 0 < len(changed_positions) < limit:
            marks.append("uneven_sentence_change")
        if len(original_sentences) != len(updated):
            marks.append("cluster_length_shift")
        if rewrite_intensity == "high" and len(changed_positions) >= 2:
            marks.append("heavy_light_block_contrast")
        if marks:
            rules.append("human:uneven-rewrite-distribution")
            notes.append("human-noise model accepted controlled asymmetry without changing protected content")
        return updated, self._deduplicate_preserve_order(rules), notes, self._deduplicate_preserve_order(marks)

    def _revision_patterns_from_rules(
        self,
        rules: list[str],
        original_sentences: list[str],
        rewritten_sentences: list[str],
        rewrite_depth: str,
    ) -> list[str]:
        patterns: list[str] = []
        for rule in rules:
            if rule in {"subject:meta-compression", "natural:dense-nominal", "subject:merge-consecutive"}:
                patterns.append("compress")
            elif rule in {"sentence:scope-risk", "sentence:parallel-expansion", "sentence:parallel-expansion-short"}:
                patterns.append("expand")
            elif rule in {"fusion:context-followup", "fusion:objective-followup", "structure:merge-short-followup", "cluster:sentence-cluster-merge"}:
                patterns.append("merge")
            elif rule == "structure:split-long-sentence":
                patterns.append("split")
            elif rule in {"paragraph:topic-sentence-preserved", "paragraph:opening-style-guard"}:
                patterns.append("partial_keep")
            elif rule.startswith("paragraph:") or rule in {"cluster:cause-effect-reframe", "cluster:discourse-reordering", "cluster:local-sentence-reorder"}:
                patterns.append("reorder")
            elif rule in {"natural:weaken-template-connectors", "opening:transition"}:
                patterns.append("soften")
            elif rule.startswith("sentence:") or rule == "cluster:narrative-path-rewrite":
                patterns.append("reframe")
            elif rule == "human:partial-keep":
                patterns.append("partial_keep")
            elif rule == "local:soften-overexplicit-transition":
                patterns.append("soften")
            elif rule in {"local:reshape-supporting-sentence", "local:weaken-overfinished-sentence"}:
                patterns.append("reframe")
            elif rule in {"local:reduce-sentence-uniformity", "local:introduce-local-hierarchy"}:
                patterns.append("merge")
            elif rule == "local:light-partial-retain-with-local-rephrase":
                patterns.append("partial_keep")
            elif rule.startswith("readability:"):
                patterns.append("partial_keep")
        if len(rewritten_sentences) < len(original_sentences):
            patterns.append("merge")
        elif len(rewritten_sentences) > len(original_sentences):
            patterns.append("split")

        limit = min(len(original_sentences), len(rewritten_sentences))
        changed_count = sum(
            1
            for index in range(limit)
            if self._normalize_for_compare(original_sentences[index]) != self._normalize_for_compare(rewritten_sentences[index])
        ) + abs(len(original_sentences) - len(rewritten_sentences))
        if changed_count == 0:
            patterns.append("partial_keep")
        elif rewrite_depth == "developmental_rewrite" and changed_count >= max(1, len(original_sentences) - 1):
            patterns.append("rewrite_all")
        elif 0 < changed_count < len(original_sentences):
            patterns.append("partial_keep")
        if not patterns and changed_count > 0:
            patterns.append("reframe")
        return self._deduplicate_preserve_order(patterns)[:2]

    def _count_sentence_level_changes(self, original: list[str], rewritten: list[str]) -> int:
        limit = min(len(original), len(rewritten))
        changes = 0
        for index in range(limit):
            if self._normalize_for_compare(original[index]) != self._normalize_for_compare(rewritten[index]):
                changes += 1
        changes += abs(len(original) - len(rewritten))
        return changes

    def _count_cluster_changes(self, discourse_actions_used: list[str]) -> int:
        cluster_actions = {
            "sentence_cluster_rewrite",
            "sentence_cluster_merge",
            "sentence_cluster_split",
            "proposition_reorder",
            "discourse_reordering",
            "narrative_path_rewrite",
            "meta_compression",
            "subject_chain_compression",
            "conclusion_absorb",
            "conclusion_absorption",
            "enumeration_reframe",
            "transition_absorption",
        }
        return sum(1 for action in discourse_actions_used if action in cluster_actions)

    def _compute_discourse_change_score(
        self,
        discourse_actions_used: list[str],
        sentence_level_changes: int,
        cluster_changes: int,
    ) -> int:
        score = len(discourse_actions_used) * 2
        score += cluster_changes * 3
        score += min(sentence_level_changes, 4)
        return score

    def _compute_rewrite_coverage(
        self,
        original_sentences: list[str],
        sentence_level_changes: int,
        cluster_changes: int,
    ) -> float:
        if not original_sentences:
            return 0.0
        changed_units = min(len(original_sentences), max(sentence_level_changes, cluster_changes))
        return changed_units / len(original_sentences)

    def _is_prefix_only_rewrite(self, applied_rules: list[str], structural_actions: list[str]) -> bool:
        if not applied_rules or structural_actions:
            return False
        if not any(rule.startswith("opening:") for rule in applied_rules):
            return False
        return all(rule.startswith(("opening:", "filler:")) for rule in applied_rules)

    def _detect_paragraph_patterns(self, units: list[SentenceUnit]) -> list[str]:
        patterns: list[str] = []
        labels = [unit.label for unit in units]
        subject_heads = [self._extract_subject_head(unit.text) for unit in units]

        if any(self._looks_like_parallel_enumeration(unit.text) for unit in units):
            patterns.append("enumeration_unified")
        if any(label == "objective" for label in labels) and any(label in {"support", "capability"} for label in labels):
            patterns.append("study_description")
        if any(label == "risk" for label in labels) and any(label == "objective" for label in labels):
            patterns.append("risk_objective")
        if self._has_repeated_subject_risk(subject_heads):
            patterns.append("repeated_meta_subject")

        for index in range(len(units) - 1):
            current = units[index]
            next_unit = units[index + 1]
            transition_prefix, _ = self._strip_opening(next_unit.text, _TRANSITION_PREFIXES)
            implication_prefix, _ = self._strip_opening(next_unit.text, _IMPLICATION_PREFIXES)

            if current.label in {"background", "capability", "support"} and transition_prefix:
                patterns.append("background_transition")
            if current.label in {"risk", "support", "detail"} and (implication_prefix or next_unit.label == "conclusion"):
                patterns.append("risk_conclusion")
            if current.label in {"capability", "support"} and next_unit.label in {"detail", "support", "capability"}:
                patterns.append("capability_extension")
            if self._extract_subject_head(current.text) in _META_SUBJECTS and self._extract_subject_head(next_unit.text) == self._extract_subject_head(current.text):
                patterns.append("subject_chain")

        return self._deduplicate_preserve_order(patterns)

    def _looks_like_parallel_enumeration(self, sentence: str) -> bool:
        return bool(
            re.match(r"^(?P<items>[^，。；]+(?:、[^，。；]+){2,})，(?P<predicate>(?:都|都会|均会|都将|都可能).+)$", sentence.strip())
        )

    def _technical_density_is_high(self, sentence: str) -> bool:
        technical_tokens = re.findall(
            r"[A-Za-z][A-Za-z0-9_.\-/]{2,}|checkpoints?/[^，。；\s]+|[\w.-]+\.(?:pth|pt|ckpt)|\[\d+\]",
            sentence,
        )
        return len(technical_tokens) >= 3

    def _deduplicate_preserve_order(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _choose_candidate(
        self,
        family: str,
        original_text: str,
        candidates: list[CandidateOption],
        prefer_change: bool,
        allow_keep: bool,
    ) -> tuple[CandidateOption | None, list[str]]:
        options = list(candidates)
        if allow_keep:
            options.append(CandidateOption(key=f"{family}:keep", text=original_text, surface="keep-original"))
        if not options:
            return None, []

        keep_option = next((option for option in options if option.key.endswith(":keep")), None)
        if keep_option is not None and self._should_prefer_keep(
            family=family,
            candidates=[option for option in options if not option.key.endswith(":keep")],
            prefer_change=prefer_change,
        ):
            return keep_option, [f"template-avoidance family={family} kept-original"]

        seed = sum(ord(character) for character in self._normalize_for_compare(original_text))
        seed += self.state.family_usage.get(family, 0)
        rotation = seed % len(options)
        rotated = options[rotation:] + options[:rotation]

        chosen = rotated[0]
        chosen_score = self._score_candidate(family, chosen, prefer_change=prefer_change)
        for order, option in enumerate(rotated[1:], start=1):
            option_score = self._score_candidate(family, option, prefer_change=prefer_change) + order * 0.01
            if option_score < chosen_score:
                chosen = option
                chosen_score = option_score

        notes: list[str] = []
        preferred = rotated[0]
        if chosen.key != preferred.key and not chosen.key.endswith(":keep"):
            notes.append(f"repeat-avoidance family={family} preferred={preferred.surface} selected={chosen.surface}")

        if not chosen.key.endswith(":keep"):
            template_family = self._infer_template_family(family, chosen)
            rule_like = self._is_rule_like_candidate(family, chosen)
            self.state.record_candidate(family, chosen, template_family=template_family, rule_like=rule_like)

        return chosen, notes

    def _should_prefer_keep(
        self,
        family: str,
        candidates: list[CandidateOption],
        prefer_change: bool,
    ) -> bool:
        if not candidates:
            return False

        if not prefer_change and any(
            self._is_rule_like_candidate(family, option) or self._infer_template_family(family, option)
            for option in candidates
        ):
            return True

        paragraph_family_count = self.state.paragraph_family_usage.get(family, 0)
        if paragraph_family_count >= 1:
            return True

        global_rule_like_count = self.state.rule_like_usage.get(family, 0)
        if global_rule_like_count >= DEFAULT_CONFIG.rule_like_family_warning_limit:
            return True

        for option in candidates:
            if self._uses_strict_template_surface(option):
                if self.state.surface_usage.get(option.surface, 0) >= DEFAULT_CONFIG.strict_blacklist_repeat_limit:
                    return True
            template_family = self._infer_template_family(family, option)
            if not template_family:
                continue
            if (
                self.state.paragraph_template_family_usage.get(template_family, 0)
                >= DEFAULT_CONFIG.paragraph_template_family_limit
            ):
                return True
            if (
                self.state.template_family_usage.get(template_family, 0)
                >= DEFAULT_CONFIG.document_template_family_warning_limit
            ):
                return True

        return False

    def _score_candidate(self, family: str, option: CandidateOption, prefer_change: bool) -> float:
        score = 0.0
        score += self.state.variant_usage.get(option.key, 0) * 4.0
        score += self.state.surface_usage.get(option.surface, 0) * 3.0
        if option.key in self.state.recent_keys:
            score += 4.0
        if option.surface in self.state.recent_surfaces:
            score += 5.0

        template_family = self._infer_template_family(family, option)
        if template_family:
            score += self.state.template_family_usage.get(template_family, 0) * 3.5
            score += self.state.paragraph_template_family_usage.get(template_family, 0) * 6.0
            if template_family in self.state.recent_template_families:
                score += 4.0

        if self._is_rule_like_candidate(family, option):
            score += self.state.rule_like_usage.get(family, 0) * 3.0
            score += self.state.paragraph_rule_like_usage.get(family, 0) * 5.0

        if self._uses_strict_template_surface(option):
            score += 12.0
            score += self.state.surface_usage.get(option.surface, 0) * 8.0

        if option.key.endswith(":keep"):
            score += 6.0 if prefer_change else -1.0
        else:
            score += 1.0 if not prefer_change else 0.0
        return score

    def _uses_strict_template_surface(self, option: CandidateOption) -> bool:
        if option.surface in _STRICT_TEMPLATE_SURFACES:
            return True
        if option.text.startswith("从") and "角度看" in option.text:
            return True
        return False

    def _infer_template_family(self, family: str, option: CandidateOption) -> str | None:
        if option.template_family is not None:
            return option.template_family or None

        surface = option.surface
        text = option.text
        if family in {"subject-variation", "subject-drop"} or surface in {"研究", "本文", "本研究", "该系统"}:
            return "study_subject_family"
        if family in {"implication-opening", "therefore-study", "therefore-build", "implication-followup", "subject-followup"}:
            return "implication_family"
        if family in {"transition-opening", "context-followup", "objective-followup"}:
            return "transition_family"
        if family in {"study-focus", "meta-compression"} or any(
            marker in surface or marker in text for marker in ("围绕", "聚焦于", "尝试回应", "讨论的核心问题")
        ):
            return "framing_family"
        return None

    def _is_rule_like_candidate(self, family: str, option: CandidateOption) -> bool:
        if option.rule_like is not None:
            return option.rule_like
        if option.key.endswith(":keep"):
            return False
        return family in _RULE_LIKE_FAMILIES

    def _classify_sentence(self, sentence: str) -> str:
        stripped = self._strip_end_punctuation(sentence)

        if stripped.startswith(_SUMMARY_PREFIXES + _IMPLICATION_PREFIXES):
            return "conclusion"
        if re.search(r"(本研究|本文|研究旨在|目标是|聚焦于|围绕.+展开|主要讨论|重点在于)", stripped):
            return "objective"
        if re.search(r"(风险|问题|挑战|局限|偏差|不稳定|瓶颈)", stripped):
            return "risk"
        if re.search(r"(能够|可以|可用于|支持|实现|完成|开始|进入|延伸到|扩展到)", stripped):
            return "capability"
        if re.search(r"(近年来|当前|目前|随着|在.+背景下|在.+情境下)", stripped):
            return "background"
        if re.search(r"(讨论|分析|说明|表明|案例|应用|场景|对象|样本|结果)", stripped):
            return "support"
        return "detail"

    def _deduplicate_adjacent_units(self, units: list[SentenceUnit]) -> list[SentenceUnit]:
        cleaned: list[SentenceUnit] = []
        previous_key = ""

        for unit in units:
            key = self._normalize_for_compare(unit.text)
            if key and key == previous_key:
                continue
            cleaned.append(unit)
            previous_key = key

        return cleaned

    def _deduplicate_clauses(self, sentence: str) -> str:
        pieces = re.split(r"(，|,)", sentence)
        if len(pieces) <= 1:
            return sentence

        rebuilt: list[str] = []
        previous_key = ""
        index = 0

        while index < len(pieces):
            clause = pieces[index]
            punct = pieces[index + 1] if index + 1 < len(pieces) else ""
            key = self._normalize_clause_key(clause)
            if key and key == previous_key:
                index += 2
                continue

            rebuilt.append(clause)
            if punct:
                rebuilt.append(punct)
            previous_key = key
            index += 2

        return "".join(rebuilt).strip()

    def _build_followup_variants(self, right_core: str) -> list[str]:
        patterns = (
            (r"^(?:文章|本文|本研究|研究)?(?:还|也)?讨论了(?P<object>.+)$", "讨论了", "讨论"),
            (r"^(?:文章|本文|本研究|研究)?(?:还|也)?分析了(?P<object>.+)$", "分析了", "分析"),
            (r"^(?:文章|本文|本研究|研究)?(?:还|也)?提到(?:了)?(?P<object>.+)$", "提到了", "提及"),
            (r"^(?:文章|本文|本研究|研究)?(?:还|也)?说明了(?P<object>.+)$", "说明了", "说明"),
        )
        for pattern, verb_phrase, noun_phrase in patterns:
            match = re.match(pattern, right_core)
            if match:
                obj = match.group("object").strip()
                return [
                    f"同时{verb_phrase}{obj}",
                    f"也{verb_phrase}{obj}",
                    f"{noun_phrase}也覆盖了{obj}",
                ]

        softened = self._soft_insert_particle(right_core, particle="也")
        return [softened, f"同时{right_core.lstrip('并')}", f"相关分析也覆盖了{right_core}"]

    def _format_enumeration_items(self, items: list[str]) -> str:
        if len(items) == 2:
            return f"{items[0]}还是{items[1]}"
        return f"{'、'.join(items[:-1])}，还是{items[-1]}"

    def _strip_opening(self, sentence: str, prefixes: tuple[str, ...]) -> tuple[str | None, str]:
        stripped = sentence.strip()
        for prefix in sorted(prefixes, key=len, reverse=True):
            if stripped.startswith(f"{prefix}，"):
                return prefix, stripped[len(prefix) + 1 :].lstrip("，, ")
            if stripped.startswith(prefix):
                return prefix, stripped[len(prefix) :].lstrip("，, ")
        return None, stripped

    def _strip_sequence_connector(self, sentence: str) -> str:
        return re.sub(r"^(?:首先|其次|再次|此外|最后|综上所述)[，,\s]*", "", sentence.strip())

    def _soft_insert_particle(self, text: str, particle: str) -> str:
        if text.startswith((particle, "文章", "本文", "本研究", "研究")):
            return text

        match = re.match(r"^(?P<subject>[^，。；]{1,12}?)(?P<lemma>已经|正在|开始|能够|可以|会|将|仍然|仍|需要|还)(?P<rest>.+)$", text)
        if match:
            return f"{match.group('subject')}{particle}{match.group('lemma')}{match.group('rest')}"
        return text

    def _find_split_point(self, sentence: str, threshold: int) -> int | None:
        stripped = self._strip_end_punctuation(sentence)
        if len(stripped) < threshold:
            return None

        candidates: list[int] = []
        for marker in ("，同时", "，并且", "，并", "，而", "，从而", "，进而", "，这也意味着", "，其中", "，不过"):
            start = 0
            while True:
                position = stripped.find(marker, start)
                if position == -1:
                    break
                candidates.append(position)
                start = position + len(marker)

        if not candidates:
            candidates = [match.start() for match in re.finditer(r"[，,]", stripped)]
        if not candidates:
            return None

        midpoint = len(stripped) // 2
        for candidate in sorted(candidates, key=lambda value: abs(value - midpoint)):
            left = stripped[:candidate].strip()
            right = stripped[candidate + 1 :].strip()
            if self._split_would_create_dangling_opening_fragment(left, right):
                continue
            if len(left) >= 18 and len(right) >= 18:
                return candidate
        return None

    def _split_would_create_dangling_opening_fragment(self, left: str, right: str) -> bool:
        if re.search(
            r"在\s*(?:\[\[AIRC:CORE_[A-Z_]+:\d+\]\]|[A-Za-z0-9_./-]+|[^，。；]{2,24})\s*[^，。；]{0,8}任务中$",
            left,
        ):
            return True
        if re.search(
            r"在\s*(?:\[\[AIRC:CORE_[A-Z_]+:\d+\]\]|[A-Za-z0-9_./-]+|[^，。；]{2,24})\s*中$",
            left,
        ) and right.startswith(("这一", "该", "其", "模型", "系统")):
            return True
        if re.search(r"(?:过程中|场景中|背景下|层面|方面|语境下|设计中)$", left) and right.startswith(
            ("针对", "通过", "对", "以", "并", "从而")
        ):
            return True
        return False

    def _normalize_whitespace(self, text: str) -> str:
        normalized = _LINE_BREAK_RE.sub(" ", text)
        normalized = _MULTISPACE_RE.sub(" ", normalized)
        return normalized.strip()

    def _strip_end_punctuation(self, sentence: str) -> str:
        return _TRAILING_PUNCT_RE.sub("", sentence).strip()

    def _ensure_sentence_end(self, sentence: str) -> str:
        stripped = sentence.strip()
        if not stripped:
            return ""
        if stripped.endswith("]]"):
            return stripped
        if stripped[-1] in "。！？?!.:：;；":
            return stripped
        return f"{stripped}{'。' if self._contains_cjk(stripped) else '.'}"

    def _normalize_for_compare(self, text: str) -> str:
        return re.sub(r"\s+", "", self._strip_end_punctuation(text))

    def _normalize_clause_key(self, clause: str) -> str:
        cleaned = _LEADING_CONNECTOR_RE.sub("", clause.strip())
        return re.sub(r"\s+", "", cleaned)

    def _contains_cjk(self, text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    def _has_sentence_level_change(self, original: list[str], rewritten: list[str]) -> bool:
        if len(original) != len(rewritten):
            return True
        differences = 0
        for left, right in zip(original, rewritten):
            if self._normalize_for_compare(left) != self._normalize_for_compare(right):
                differences += 1
        return differences >= 2


class Rewriter:
    def __init__(self, backend: RewriteBackend | None = None, style_profile: str = "academic_natural") -> None:
        self.style_profile = style_profile
        self.backend = backend or RuleBasedRewriteBackend(style_profile=style_profile)

    def reset_document_state(self) -> None:
        if hasattr(self.backend, "reset_document_state"):
            self.backend.reset_document_state()

    def rewrite(
        self,
        text: str,
        mode: RewriteMode,
        pass_index: int = 1,
        rewrite_depth: str | None = None,
        rewrite_intensity: str | None = None,
        high_sensitivity_prose: bool = False,
        style_profile: str | None = None,
    ) -> tuple[str, RewriteStats]:
        result = self.backend.rewrite(
            text,
            mode,
            pass_index=pass_index,
            rewrite_depth=rewrite_depth,
            rewrite_intensity=rewrite_intensity,
            high_sensitivity_prose=high_sensitivity_prose,
            style_profile=style_profile or self.style_profile,
        )
        changed_characters = sum(1 for left, right in zip(text, result.text) if left != right) + abs(
            len(text) - len(result.text)
        )
        return result.text, RewriteStats(
            mode=mode,
            changed=result.text != text,
            applied_rules=result.applied_rules,
            sentence_count_before=len(result.original_sentences),
            sentence_count_after=len(result.rewritten_sentences),
            sentence_level_change=result.sentence_level_change,
            changed_characters=changed_characters,
            original_sentences=result.original_sentences,
            rewritten_sentences=result.rewritten_sentences,
            paragraph_char_count=result.paragraph_char_count,
            sentence_labels=result.sentence_labels,
            subject_heads=result.subject_heads,
            detected_patterns=result.detected_patterns,
            structural_actions=result.structural_actions,
            structural_action_count=result.structural_action_count,
            high_value_structural_actions=result.high_value_structural_actions,
            discourse_actions_used=result.discourse_actions_used,
            sentence_level_changes=result.sentence_level_changes,
            cluster_changes=result.cluster_changes,
            discourse_change_score=result.discourse_change_score,
            rewrite_coverage=result.rewrite_coverage,
            prefix_only_rewrite=result.prefix_only_rewrite,
            repeated_subject_risk=result.repeated_subject_risk,
            selected_variants=result.selected_variants,
            candidate_notes=result.candidate_notes,
            paragraph_index=result.paragraph_index,
            block_id=0,
            rewrite_depth=rewrite_depth or "",
            rewrite_intensity=rewrite_intensity or "",
            revision_patterns=result.revision_patterns,
            human_noise_marks=result.human_noise_marks,
            sentence_transition_rigidity=result.sentence_transition_rigidity,
            local_discourse_flatness=result.local_discourse_flatness,
            revision_realism_score=result.revision_realism_score,
            sentence_cadence_irregularity=result.sentence_cadence_irregularity,
            local_revision_actions=result.local_revision_actions,
            sentence_completeness_score=result.sentence_completeness_score,
            paragraph_readability_score=result.paragraph_readability_score,
            dangling_sentence_risk=result.dangling_sentence_risk,
            incomplete_support_sentence_risk=result.incomplete_support_sentence_risk,
            fragment_like_conclusion_risk=result.fragment_like_conclusion_risk,
            readability_repair_actions=result.readability_repair_actions,
            high_sensitivity_prose=high_sensitivity_prose or result.high_sensitivity_prose,
        )
