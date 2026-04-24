from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_BUREAUCRATIC_OPENING_RE = re.compile(
    r"^(?:本研究面向[^。；，,]{2,36}需求|在方法上|围绕这一目标|在解决[^。；]{2,60}目标下|"
    r"在完成[^。；]{2,48}后|在[^。；，,]{2,24}层面|本研究具有显著[^。；]{0,24}价值|"
    r"本研究的主题为)"
)
_SLOGAN_GOAL_RE = re.compile(
    r"(?:旨在构建|旨在通过|以实现[^。；]{1,36}(?:落地|闭环)|形成(?:了)?完整闭环|"
    r"具有显著[^。；]{0,24}价值|提供清晰方法论路径|筑牢第一道防线)"
)
_OVERSTRUCTURED_RE = re.compile(
    r"(?:并不是[^。；]{2,80}而是|不仅[^。；]{2,80}(?:同时|还)|在结构上[^。；]{2,80}在决策层面|"
    r"第一[^。；]{2,60}进而第二)"
)
_SUBJECT_RE = re.compile(r"^(本研究|本文|研究|系统|模型|该系统|该模型)")
_DELAYED_MAIN_RE = re.compile(
    r"^在(?:解决|完成|推进|实现|[^，。；]{2,16})(?:[^，。；]{18,90})[，,](?:本研究|本文|研究|系统|模型)"
)
_DIRECT_WRAPPER_RE = re.compile(
    r"(?:^在[^，。；]{2,28}(?:中|下|过程中)[，,]|^具体而言[，,]|^围绕[^，。；]{2,28}[，,]|"
    r"^从[^，。；]{2,28}角度(?:来看)?[，,]|^就[^，。；]{2,28}而言[，,])"
)
_CONNECTOR_RE = re.compile(r"(?:不仅[^。；]{2,80}(?:还|也|同时)|并且|从而|进而|同时)")
_NOMINALIZATION_RE = re.compile(r"(?:进行(?:评估|分析|验证|控制|处理)|实现(?:控制|部署|落地|融合)|提供(?:支撑|支持|基础|路径))")
_PASSIVE_VOICE_RE = re.compile(r"(?:被视为|属于|得到体现)")


@dataclass(frozen=True)
class AcademicSentenceNaturalizationSignals:
    bureaucratic_opening_density: float
    repeated_explicit_subject_risk: float
    overstructured_syntax_risk: float
    delayed_main_clause_risk: float
    slogan_like_goal_risk: float
    directness_score: float
    connector_overuse_risk: float
    nominalization_density: float
    passive_voice_ratio: float
    overlong_sentence_risk: float
    subject_monotony_risk: float
    bureaucratic_opening_controlled: bool
    explicit_subject_chain_controlled: bool
    overstructured_syntax_controlled: bool
    main_clause_position_reasonable: bool
    slogan_like_goal_phrase_controlled: bool
    author_style_alignment_controlled: bool
    bureaucratic_paragraph_ids: list[int]
    repeated_subject_paragraph_ids: list[int]
    overstructured_paragraph_ids: list[int]
    delayed_main_clause_paragraph_ids: list[int]
    slogan_like_goal_paragraph_ids: list[int]
    low_directness_paragraph_ids: list[int]
    connector_overuse_paragraph_ids: list[int]
    nominalization_paragraph_ids: list[int]
    passive_voice_paragraph_ids: list[int]
    overlong_sentence_paragraph_ids: list[int]
    subject_monotony_paragraph_ids: list[int]


def analyze_academic_sentence_naturalization(
    sentences: list[str],
    *,
    paragraph_index: int = 0,
) -> AcademicSentenceNaturalizationSignals:
    """Measure project-style academic syntax that makes rewritten prose feel mechanically polished."""

    visible = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not visible:
        return AcademicSentenceNaturalizationSignals(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            True,
            True,
            True,
            True,
            True,
            True,
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

    bureaucratic_hits = sum(1 for sentence in visible if _BUREAUCRATIC_OPENING_RE.search(sentence))
    slogan_hits = sum(1 for sentence in visible if _SLOGAN_GOAL_RE.search(sentence))
    overstructured_hits = sum(1 for sentence in visible if _OVERSTRUCTURED_RE.search(sentence))
    delayed_hits = sum(1 for sentence in visible if _delayed_main_clause(sentence))
    direct_wrapper_hits = sum(1 for sentence in visible if _DIRECT_WRAPPER_RE.search(sentence))
    connector_hits = sum(len(_CONNECTOR_RE.findall(sentence)) for sentence in visible)
    nominalization_hits = sum(len(_NOMINALIZATION_RE.findall(sentence)) for sentence in visible)
    passive_hits = sum(len(_PASSIVE_VOICE_RE.findall(sentence)) for sentence in visible)
    overlong_hits = sum(1 for sentence in visible if _overlong_sentence(sentence))
    subject_chain_hits = _repeated_explicit_subject_hits(visible)
    denominator = max(1, len(visible))

    bureaucratic_density = round(min(1.0, bureaucratic_hits / denominator), 4)
    repeated_subject_risk = round(min(1.0, subject_chain_hits / max(1, denominator - 1)), 4)
    overstructured_risk = round(min(1.0, overstructured_hits / denominator), 4)
    delayed_risk = round(min(1.0, delayed_hits / denominator), 4)
    slogan_risk = round(min(1.0, slogan_hits / denominator), 4)
    connector_risk = round(min(1.0, connector_hits / max(1, denominator * 2)), 4)
    nominalization_density = round(min(1.0, nominalization_hits / max(1, denominator * 2)), 4)
    passive_ratio = round(min(1.0, passive_hits / denominator), 4)
    overlong_risk = round(min(1.0, overlong_hits / denominator), 4)
    subject_monotony_risk = repeated_subject_risk
    directness_penalty = min(
        1.0,
        (direct_wrapper_hits / denominator * 0.28)
        + (delayed_risk * 0.22)
        + (connector_risk * 0.16)
        + (nominalization_density * 0.14)
        + (passive_ratio * 0.10)
        + (overlong_risk * 0.10),
    )
    directness_score = round(max(0.0, 1.0 - directness_penalty), 4)
    author_style_ok = (
        directness_score >= 0.82
        and connector_risk <= 0.18
        and nominalization_density <= 0.16
        and passive_ratio <= 0.14
        and overlong_risk <= 0.22
        and subject_monotony_risk <= 0.20
    )

    return AcademicSentenceNaturalizationSignals(
        bureaucratic_opening_density=bureaucratic_density,
        repeated_explicit_subject_risk=repeated_subject_risk,
        overstructured_syntax_risk=overstructured_risk,
        delayed_main_clause_risk=delayed_risk,
        slogan_like_goal_risk=slogan_risk,
        directness_score=directness_score,
        connector_overuse_risk=connector_risk,
        nominalization_density=nominalization_density,
        passive_voice_ratio=passive_ratio,
        overlong_sentence_risk=overlong_risk,
        subject_monotony_risk=subject_monotony_risk,
        bureaucratic_opening_controlled=bureaucratic_density <= 0.14,
        explicit_subject_chain_controlled=repeated_subject_risk <= 0.20,
        overstructured_syntax_controlled=overstructured_risk <= 0.16,
        main_clause_position_reasonable=delayed_risk <= 0.12,
        slogan_like_goal_phrase_controlled=slogan_risk <= 0.12,
        author_style_alignment_controlled=author_style_ok,
        bureaucratic_paragraph_ids=[paragraph_index] if bureaucratic_density > 0.14 else [],
        repeated_subject_paragraph_ids=[paragraph_index] if repeated_subject_risk > 0.20 else [],
        overstructured_paragraph_ids=[paragraph_index] if overstructured_risk > 0.16 else [],
        delayed_main_clause_paragraph_ids=[paragraph_index] if delayed_risk > 0.12 else [],
        slogan_like_goal_paragraph_ids=[paragraph_index] if slogan_risk > 0.12 else [],
        low_directness_paragraph_ids=[paragraph_index] if directness_score < 0.82 else [],
        connector_overuse_paragraph_ids=[paragraph_index] if connector_risk > 0.18 else [],
        nominalization_paragraph_ids=[paragraph_index] if nominalization_density > 0.16 else [],
        passive_voice_paragraph_ids=[paragraph_index] if passive_ratio > 0.14 else [],
        overlong_sentence_paragraph_ids=[paragraph_index] if overlong_risk > 0.22 else [],
        subject_monotony_paragraph_ids=[paragraph_index] if subject_monotony_risk > 0.20 else [],
    )


def aggregate_academic_sentence_naturalization(rewrite_stats: list[Any]) -> dict[str, object]:
    """Aggregate sentence-naturalization metrics across rewritten body prose."""

    checked = [
        stats
        for stats in rewrite_stats
        if getattr(stats, "changed", False)
        and getattr(stats, "rewrite_depth", "") in {"developmental_rewrite", "light_edit"}
        and getattr(stats, "rewritten_sentences", None)
    ]
    if not checked:
        return {
            "paragraphs_checked": 0,
            "bureaucratic_opening_density": 0.0,
            "repeated_explicit_subject_risk": 0.0,
            "overstructured_syntax_risk": 0.0,
            "delayed_main_clause_risk": 0.0,
            "slogan_like_goal_risk": 0.0,
            "directness_score": 1.0,
            "connector_overuse_risk": 0.0,
            "nominalization_density": 0.0,
            "passive_voice_ratio": 0.0,
            "overlong_sentence_risk": 0.0,
            "subject_monotony_risk": 0.0,
            "bureaucratic_opening_controlled": True,
            "explicit_subject_chain_controlled": True,
            "overstructured_syntax_controlled": True,
            "main_clause_position_reasonable": True,
            "slogan_like_goal_phrase_controlled": True,
            "author_style_alignment_controlled": True,
            "bureaucratic_paragraph_ids": [],
            "repeated_subject_paragraph_ids": [],
            "overstructured_paragraph_ids": [],
            "delayed_main_clause_paragraph_ids": [],
            "slogan_like_goal_paragraph_ids": [],
            "low_directness_paragraph_ids": [],
            "connector_overuse_paragraph_ids": [],
            "nominalization_paragraph_ids": [],
            "passive_voice_paragraph_ids": [],
            "overlong_sentence_paragraph_ids": [],
            "subject_monotony_paragraph_ids": [],
        }

    signals = [
        analyze_academic_sentence_naturalization(
            list(getattr(stats, "rewritten_sentences", [])),
            paragraph_index=int(getattr(stats, "paragraph_index", 0) or 0),
        )
        for stats in checked
    ]
    paragraphs_checked = len(signals)
    bureaucratic_ids = _merge_ids(signal.bureaucratic_paragraph_ids for signal in signals)
    repeated_ids = _merge_ids(signal.repeated_subject_paragraph_ids for signal in signals)
    overstructured_ids = _merge_ids(signal.overstructured_paragraph_ids for signal in signals)
    delayed_ids = _merge_ids(signal.delayed_main_clause_paragraph_ids for signal in signals)
    slogan_ids = _merge_ids(signal.slogan_like_goal_paragraph_ids for signal in signals)
    low_directness_ids = _merge_ids(signal.low_directness_paragraph_ids for signal in signals)
    connector_ids = _merge_ids(signal.connector_overuse_paragraph_ids for signal in signals)
    nominalization_ids = _merge_ids(signal.nominalization_paragraph_ids for signal in signals)
    passive_ids = _merge_ids(signal.passive_voice_paragraph_ids for signal in signals)
    overlong_ids = _merge_ids(signal.overlong_sentence_paragraph_ids for signal in signals)
    subject_monotony_ids = _merge_ids(signal.subject_monotony_paragraph_ids for signal in signals)
    average_bureaucratic = round(_average(signal.bureaucratic_opening_density for signal in signals), 4)
    average_repeated = round(_average(signal.repeated_explicit_subject_risk for signal in signals), 4)
    average_overstructured = round(_average(signal.overstructured_syntax_risk for signal in signals), 4)
    average_delayed = round(_average(signal.delayed_main_clause_risk for signal in signals), 4)
    average_slogan = round(_average(signal.slogan_like_goal_risk for signal in signals), 4)
    average_directness = round(_average(signal.directness_score for signal in signals), 4)
    average_connector = round(_average(signal.connector_overuse_risk for signal in signals), 4)
    average_nominalization = round(_average(signal.nominalization_density for signal in signals), 4)
    average_passive = round(_average(signal.passive_voice_ratio for signal in signals), 4)
    average_overlong = round(_average(signal.overlong_sentence_risk for signal in signals), 4)
    average_subject_monotony = round(_average(signal.subject_monotony_risk for signal in signals), 4)
    author_style_ok = (
        average_directness >= 0.88
        and average_connector <= 0.12
        and average_nominalization <= 0.10
        and average_passive <= 0.08
        and average_overlong <= 0.16
        and average_subject_monotony <= 0.12
        and len(low_directness_ids) <= max(5, paragraphs_checked // 5)
    )

    return {
        "paragraphs_checked": paragraphs_checked,
        "bureaucratic_opening_density": average_bureaucratic,
        "repeated_explicit_subject_risk": average_repeated,
        "overstructured_syntax_risk": average_overstructured,
        "delayed_main_clause_risk": average_delayed,
        "slogan_like_goal_risk": average_slogan,
        "directness_score": average_directness,
        "connector_overuse_risk": average_connector,
        "nominalization_density": average_nominalization,
        "passive_voice_ratio": average_passive,
        "overlong_sentence_risk": average_overlong,
        "subject_monotony_risk": average_subject_monotony,
        "bureaucratic_opening_controlled": (
            average_bureaucratic <= 0.08 and len(bureaucratic_ids) <= max(3, paragraphs_checked // 6)
        ),
        "explicit_subject_chain_controlled": (
            average_repeated <= 0.12 and len(repeated_ids) <= max(4, paragraphs_checked // 5)
        ),
        "overstructured_syntax_controlled": (
            average_overstructured <= 0.10 and len(overstructured_ids) <= max(4, paragraphs_checked // 5)
        ),
        "main_clause_position_reasonable": (
            average_delayed <= 0.08 and len(delayed_ids) <= max(3, paragraphs_checked // 7)
        ),
        "slogan_like_goal_phrase_controlled": (
            average_slogan <= 0.08 and len(slogan_ids) <= max(3, paragraphs_checked // 7)
        ),
        "author_style_alignment_controlled": author_style_ok,
        "bureaucratic_paragraph_ids": bureaucratic_ids,
        "repeated_subject_paragraph_ids": repeated_ids,
        "overstructured_paragraph_ids": overstructured_ids,
        "delayed_main_clause_paragraph_ids": delayed_ids,
        "slogan_like_goal_paragraph_ids": slogan_ids,
        "low_directness_paragraph_ids": low_directness_ids,
        "connector_overuse_paragraph_ids": connector_ids,
        "nominalization_paragraph_ids": nominalization_ids,
        "passive_voice_paragraph_ids": passive_ids,
        "overlong_sentence_paragraph_ids": overlong_ids,
        "subject_monotony_paragraph_ids": subject_monotony_ids,
    }


def _repeated_explicit_subject_hits(sentences: list[str]) -> int:
    heads = []
    for sentence in sentences:
        match = _SUBJECT_RE.match(sentence)
        heads.append(match.group(1) if match else "")
    return sum(1 for left, right in zip(heads, heads[1:], strict=False) if left and right and left == right)


def _delayed_main_clause(sentence: str) -> bool:
    stripped = sentence.strip()
    if _DELAYED_MAIN_RE.search(stripped):
        return True
    comma = re.search(r"[，,]", stripped)
    if not comma:
        return False
    prefix = stripped[: comma.start()]
    return len(prefix) >= 34 and bool(re.search(r"(目标|过程|背景|层面|基础|情况下|完成|解决)", prefix))


def _overlong_sentence(sentence: str) -> bool:
    compact = re.sub(r"\s+", "", sentence)
    if len(compact) < 86:
        return False
    logic_marks = len(re.findall(r"(?:，|；|并且|从而|进而|同时|不仅|还|以便|用于|通过)", compact))
    return logic_marks >= 3


def _average(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def _merge_ids(groups: Any) -> list[int]:
    merged: list[int] = []
    for group in groups:
        for value in group:
            if value not in merged:
                merged.append(value)
    return merged
