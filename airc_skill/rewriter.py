from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .config import DEFAULT_CONFIG, RewriteMode

_LINE_BREAK_RE = re.compile(r"[ \t]*\n[ \t]*")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_TRAILING_PUNCT_RE = re.compile(r"[。！？?!]+$")
_LEADING_CONNECTOR_RE = re.compile(r"^(?:并且|而且|并|还|同时|此外|另外|与此同时|因此|所以|由此|基于此)\s*")

_SUMMARY_PREFIXES = ("总的来说", "总而言之", "总体来看", "总体而言", "综合来看", "综上")
_TRANSITION_PREFIXES = ("与此同时", "同时", "此外", "另外", "在这一过程中", "与之相伴的是")
_IMPLICATION_PREFIXES = ("因此", "由此可见", "由此", "这也意味着", "这也说明", "基于此", "在这种情况下")
_META_SUBJECTS = ("本研究", "本文", "该研究", "该系统")
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
    "clause_reorder",
    "pair_fusion",
    "enumeration_reframe",
    "conclusion_absorb",
    "paragraph_reorder",
    "topic_reframe",
    "merge_consecutive_subject_sentences",
    "subject_drop",
    "subject_variation",
    "meta_compression",
    "followup_absorb",
    "subject_chain_compression",
)
DISCOURSE_ACTION_TYPES = (
    "proposition_reorder",
    "sentence_cluster_rewrite",
    "meta_compression",
    "subject_chain_compression",
    "conclusion_absorb",
    "enumeration_reframe",
    "rationale_expansion",
    "transition_absorption",
)
HIGH_IMPACT_ACTION_TYPES = (
    "pair_fusion",
    "conclusion_absorb",
    "paragraph_reorder",
    "subject_chain_compression",
    "sentence_cluster_rewrite",
    "proposition_reorder",
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
    ) -> BackendRewriteResult:
        raise NotImplementedError("No online backend is configured in the local MVP.")


class RuleBasedRewriteBackend:
    """A local rewriter that focuses on sentence fusion, controlled variation, and paragraph flow."""

    def __init__(self) -> None:
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
    ) -> BackendRewriteResult:
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

        if mode in {RewriteMode.BALANCED, RewriteMode.STRONG}:
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

        rewritten_units, rules, surfaces, notes = self._rewrite_standalone_transitions(
            units=polished_units,
            mode=mode,
            pass_index=pass_index,
        )
        applied_rules.extend(rules)
        selected_variants.extend(surfaces)
        candidate_notes.extend(notes)

        rewritten_sentences = [self._ensure_sentence_end(unit.text) for unit in rewritten_units if unit.text.strip()]
        final_core = "".join(rewritten_sentences)
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
        )

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
                    "subject-variation:work",
                    f"这一工作{body_core}。",
                    "这一工作",
                ),
                CandidateOption(
                    "subject-variation:transition",
                    f"在此基础上，{self._subject_variant_phrase(subject, body_core)}。",
                    "在此基础上，研究",
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

        return (
            SentenceUnit(text=updated_text, label=unit.label, source_indices=unit.source_indices),
            rule_names,
            selected_variants,
            candidate_notes,
        )

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
                    CandidateOption("therefore-study:based-on-this", f"在此基础上，研究进一步{rest}。", "在此基础上，研究"),
                    CandidateOption("therefore-study:in-this-case", f"由此，研究{rest}。", "由此，研究"),
                    CandidateOption("therefore-study:it-shows", f"相应地，本文{rest}。", "相应地，本文"),
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

        for unit in units:
            prefix, remainder = self._strip_opening(unit.text, _TRANSITION_PREFIXES)
            if not prefix:
                rewritten.append(unit)
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
            right_core = self._strip_end_punctuation(next_unit.text)
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
            return reordered, ["paragraph:move-summary-to-close"], [], ["paragraph-reorder moved conclusion to close"]

        try:
            objective_index = labels.index("objective")
        except ValueError:
            objective_index = -1

        if 0 < objective_index < 3 and units[0].label in {"background", "risk", "capability"}:
            reordered = [units[objective_index]] + units[:objective_index] + units[objective_index + 1 :]
            return reordered, ["paragraph:promote-objective"], [], ["paragraph-reorder promoted objective sentence"]

        if (
            units[0].label in {"background", "capability"}
            and units[-1].label in {"risk", "conclusion"}
            and units[1].label in {"detail", "support", "capability"}
        ):
            reordered = [units[0], units[-1], *units[1:-1]]
            return reordered, ["paragraph:promote-risk"], [], ["paragraph-reorder promoted risk or conclusion sentence"]

        for index, unit in enumerate(units[:-1]):
            if unit.label == "conclusion":
                reordered = units[:index] + units[index + 1 :] + [unit]
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
                return (
                    reordered,
                    ["paragraph:align-risk-objective"],
                    [],
                    ["paragraph-reorder aligned objective after risk statement"],
                )

        if rewrite_intensity == "high" and units[1].label in {"risk", "capability", "support", "detail"}:
            reordered = [units[1], units[0], *units[2:]]
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
        for subject in ("研究", "系统", "这一工作"):
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
                actions.append("pair_fusion")
            elif rule == "fusion:implication-followup":
                actions.append("conclusion_absorb")
            elif rule == "structure:parallel-enumeration":
                actions.append("enumeration_reframe")
            elif rule == "structure:split-long-sentence":
                actions.append("sentence_split")
            elif rule == "structure:merge-short-followup":
                actions.append("sentence_merge")
            elif rule.startswith("paragraph:"):
                actions.append("paragraph_reorder")
            elif rule == "sentence:study-focus":
                actions.append("topic_reframe")
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
        if any(action in {"merge_consecutive_subject_sentences", "subject_drop", "subject_variation", "meta_compression", "followup_absorb"} for action in actions):
            actions.append("subject_chain_compression")
        return actions

    def _discourse_actions_from_rules(self, rules: list[str]) -> list[str]:
        actions: list[str] = []
        for rule in rules:
            if rule in {"fusion:context-followup", "fusion:objective-followup", "structure:merge-short-followup"}:
                actions.extend(["sentence_cluster_rewrite", "transition_absorption"])
            elif rule == "fusion:implication-followup":
                actions.extend(["sentence_cluster_rewrite", "conclusion_absorb"])
            elif rule == "structure:parallel-enumeration":
                actions.append("enumeration_reframe")
            elif rule.startswith("paragraph:"):
                actions.append("proposition_reorder")
            elif rule == "subject:meta-compression":
                actions.append("meta_compression")
            elif rule in {"subject:merge-consecutive", "subject:drop", "subject:variation", "subject:followup-absorb"}:
                actions.append("subject_chain_compression")
            elif rule == "sentence:scope-risk":
                actions.append("rationale_expansion")
        return self._deduplicate_preserve_order(actions)

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
            "proposition_reorder",
            "meta_compression",
            "subject_chain_compression",
            "conclusion_absorb",
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
                    f"同时也{verb_phrase}{obj}",
                    f"并进一步{verb_phrase}{obj}",
                    f"也将{obj}纳入了{noun_phrase}",
                ]

        softened = self._soft_insert_particle(right_core, particle="也")
        return [softened, f"并进一步{right_core.lstrip('并')}", f"也将{right_core}纳入了分析"]

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
            if len(left) >= 18 and len(right) >= 18:
                return candidate
        return None

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
    def __init__(self, backend: RewriteBackend | None = None) -> None:
        self.backend = backend or RuleBasedRewriteBackend()

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
    ) -> tuple[str, RewriteStats]:
        result = self.backend.rewrite(
            text,
            mode,
            pass_index=pass_index,
            rewrite_depth=rewrite_depth,
            rewrite_intensity=rewrite_intensity,
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
        )
