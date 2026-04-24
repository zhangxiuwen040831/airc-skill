from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .sentence_readability import analyze_paragraph_readability_sentences

_SENTENCE_RE = re.compile(r"[^。！？?!]+[。！？?!]?")
_SPACE_RE = re.compile(r"\s+")
_TRANSITION_RE = re.compile(
    r"^(?:同时|此外|另外|与此同时|因此|由此|由此可见|基于此|在此基础上|在这种情况下|"
    r"具体而言|更具体地说|进一步来看|总体来看|整体来看|综上所述)[，,\s]*"
)
_OVERFINISHED_RE = re.compile(
    r"(提供(?:了)?(?:重要)?(?:基础|支撑|保障|参考)|具有(?:重要)?意义|发挥(?:重要)?作用|"
    r"形成(?:完整)?闭环|实现(?:有效)?提升|具有(?:一定)?价值)[。！？?!]?$"
)
_COMPLETE_CLAIM_RE = re.compile(
    r"(能够|可以|有助于|用于|从而|因此|由此|实现|提升|保证|确保|提供|表明|说明|体现)"
)
_TECH_MARKER_RE = re.compile(
    r"\[\[AIRC:CORE_[A-Z_]+:\d+\]\]|[A-Za-z][A-Za-z0-9_.\-/]{2,}|"
    r"checkpoint[s]?/[^，。；\s]+|[\w.-]+\.(?:pth|pt|ckpt)|\[\d+\]"
)
_ACADEMIC_CLICHE_PATTERNS = (
    re.compile(r"需要说明的是"),
    re.compile(r"值得进一步说明的是"),
    re.compile(r"需要指出的是"),
    re.compile(r"重点在于"),
    re.compile(r"(?:本研究的)?核心价值在于"),
    re.compile(r"具有重要意义"),
    re.compile(r"从而提升"),
    re.compile(r"实现了(?:一个|完整)?[^。；]{0,16}(?:闭环|体系|流程|框架)"),
    re.compile(r"为[^。；]{2,24}提供(?:了)?(?:重要)?(?:基础|支撑|保障|路径)"),
    re.compile(r"在此基础上"),
    re.compile(r"因此可以看出"),
    re.compile(r"有助于(?:提升|缓解|减少|增强|改善|保持)"),
)
_TEXTURE_JUDGMENT_RE = re.compile(r"(表明|说明|意味着|并非|更关键|关键是|更重要|更能|更适合)")
_TEXTURE_EXPLANATORY_RE = re.compile(r"(通过|用于|以便|使得|借助|围绕|依赖|构成|形成|由.+组成)")
_TEXTURE_SUPPLEMENT_RE = re.compile(r"^(?:其中|尤其|同时|另外|此外|换言之|相较之下|作为补充)[，,\s]*")


@dataclass(frozen=True)
class LocalRevisionSignals:
    sentence_transition_rigidity: float
    local_discourse_flatness: float
    revision_realism_score: float
    sentence_cadence_irregularity: float
    stylistic_uniformity_score: float
    support_sentence_texture_variation: float
    academic_cliche_density: float
    paragraph_voice_signature: str
    sentence_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "sentence_transition_rigidity": self.sentence_transition_rigidity,
            "local_discourse_flatness": self.local_discourse_flatness,
            "revision_realism_score": self.revision_realism_score,
            "sentence_cadence_irregularity": self.sentence_cadence_irregularity,
            "stylistic_uniformity_score": self.stylistic_uniformity_score,
            "support_sentence_texture_variation": self.support_sentence_texture_variation,
            "academic_cliche_density": self.academic_cliche_density,
            "paragraph_voice_signature": self.paragraph_voice_signature,
            "sentence_count": self.sentence_count,
        }


def split_sentences_for_realism(text: str) -> list[str]:
    """Split text into sentence-like units for local revision realism scoring."""

    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]


def analyze_local_revision_text(text: str) -> LocalRevisionSignals:
    """Analyze local human-revision signals from raw paragraph text."""

    return analyze_local_revision_sentences(split_sentences_for_realism(text))


def analyze_local_revision_sentences(sentences: list[str]) -> LocalRevisionSignals:
    """Measure local transition rigidity, flatness, cadence, and realism."""

    visible = [sentence.strip() for sentence in sentences if sentence.strip()]
    sentence_count = len(visible)
    if sentence_count == 0:
        return LocalRevisionSignals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "", 0)
    if sentence_count == 1:
        single_cliche_density = _academic_cliche_density(visible)
        return LocalRevisionSignals(
            0.0,
            0.0,
            round(_clamp(0.72 - single_cliche_density * 0.08), 4),
            0.35,
            round(_clamp(single_cliche_density * 0.28), 4),
            0.48,
            round(single_cliche_density, 4),
            _paragraph_voice_signature(visible, ["plain"], 0.35, single_cliche_density),
            1,
        )

    lengths = [_visible_length(sentence) for sentence in visible]
    cadence = _cadence_irregularity(lengths)
    transition_ratio = sum(1 for sentence in visible[1:] if _TRANSITION_RE.match(sentence)) / max(1, sentence_count - 1)
    opener_ratio = _max_opener_family_ratio(visible)
    overfinished_ratio = sum(1 for sentence in visible if _OVERFINISHED_RE.search(sentence)) / sentence_count
    complete_ratio = sum(1 for sentence in visible if _COMPLETE_CLAIM_RE.search(sentence)) / sentence_count
    hierarchy_ratio = _hierarchy_marker_ratio(visible)
    support_textures = _support_sentence_textures(visible)
    support_variation = _support_texture_variation(support_textures)
    cliche_density = _academic_cliche_density(visible)
    readability = analyze_paragraph_readability_sentences(visible)

    sentence_transition_rigidity = _clamp(
        transition_ratio * 0.46
        + opener_ratio * 0.24
        + overfinished_ratio * 0.18
        + max(0.0, 0.35 - cadence) * 0.35
    )
    local_discourse_flatness = _clamp(
        max(0.0, 0.40 - cadence) * 0.70
        + opener_ratio * 0.24
        + complete_ratio * 0.18
        + max(0.0, 0.25 - hierarchy_ratio) * 0.22
    )
    stylistic_uniformity = _clamp(
        sentence_transition_rigidity * 0.24
        + local_discourse_flatness * 0.26
        + max(0.0, 0.42 - support_variation) * 0.40
        + cliche_density * 0.28
        + overfinished_ratio * 0.12
        + max(0.0, 0.34 - cadence) * 0.10
    )
    revision_realism_score = _clamp(
        0.70
        - sentence_transition_rigidity * 0.34
        - local_discourse_flatness * 0.30
        - stylistic_uniformity * 0.18
        + cadence * 0.28
        + hierarchy_ratio * 0.14
        + support_variation * 0.16
        - overfinished_ratio * 0.10
        - cliche_density * 0.12
        + (readability.sentence_completeness_score - 0.72) * 0.12
        + (readability.paragraph_readability_score - 0.70) * 0.10
        - readability.dangling_sentence_risk * 0.28
        - readability.incomplete_support_sentence_risk * 0.22
    )
    return LocalRevisionSignals(
        sentence_transition_rigidity=round(sentence_transition_rigidity, 4),
        local_discourse_flatness=round(local_discourse_flatness, 4),
        revision_realism_score=round(revision_realism_score, 4),
        sentence_cadence_irregularity=round(cadence, 4),
        stylistic_uniformity_score=round(stylistic_uniformity, 4),
        support_sentence_texture_variation=round(support_variation, 4),
        academic_cliche_density=round(cliche_density, 4),
        paragraph_voice_signature=_paragraph_voice_signature(visible, support_textures, cadence, cliche_density),
        sentence_count=sentence_count,
    )


def aggregate_local_revision_realism(rewrite_stats: list[Any]) -> dict[str, object]:
    """Aggregate local realism signals over changed body rewrite stats."""

    checked = [
        stats for stats in rewrite_stats
        if getattr(stats, "changed", False)
        and getattr(stats, "rewrite_depth", "") in {"developmental_rewrite", "light_edit"}
        and getattr(stats, "rewritten_sentences", None)
    ]
    if not checked:
        return {
            "paragraphs_checked": 0,
            "sentence_transition_rigidity": 0.0,
            "local_discourse_flatness": 0.0,
            "revision_realism_score": 0.0,
            "sentence_cadence_irregularity": 0.0,
            "stylistic_uniformity_score": 0.0,
            "support_sentence_texture_variation": 1.0,
            "paragraph_voice_variation": 1.0,
            "academic_cliche_density": 0.0,
            "sentence_completeness_score": 1.0,
            "paragraph_readability_score": 1.0,
            "readability_preserved": True,
            "sentence_completeness_preserved": True,
            "local_transition_natural": True,
            "local_discourse_not_flat": True,
            "sentence_uniformity_reduced": True,
            "revision_realism_present": True,
            "stylistic_uniformity_controlled": True,
            "support_sentence_texture_varied": True,
            "paragraph_voice_variation_present": True,
            "academic_cliche_density_controlled": True,
            "low_realism_paragraph_ids": [],
            "flat_paragraph_ids": [],
            "rigid_paragraph_ids": [],
            "uniform_paragraph_ids": [],
            "low_texture_variation_paragraph_ids": [],
            "cliche_dense_paragraph_ids": [],
            "high_sensitivity_cliche_paragraph_ids": [],
        }

    revised_signals = [analyze_local_revision_sentences(list(stats.rewritten_sentences)) for stats in checked]
    original_signals = [analyze_local_revision_sentences(list(stats.original_sentences)) for stats in checked]
    action_bonus_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats in checked
        if any(str(action).startswith("local:") for action in getattr(stats, "applied_rules", []))
    ]
    average_rigidity = _average(signal.sentence_transition_rigidity for signal in revised_signals)
    average_flatness = _average(signal.local_discourse_flatness for signal in revised_signals)
    average_realism = _average(signal.revision_realism_score for signal in revised_signals)
    average_cadence = _average(signal.sentence_cadence_irregularity for signal in revised_signals)
    average_uniformity = _average(signal.stylistic_uniformity_score for signal in revised_signals)
    average_texture_variation = _average(signal.support_sentence_texture_variation for signal in revised_signals)
    average_cliche_density = _average(signal.academic_cliche_density for signal in revised_signals)
    original_flatness = _average(signal.local_discourse_flatness for signal in original_signals)
    original_cadence = _average(signal.sentence_cadence_irregularity for signal in original_signals)
    original_uniformity = _average(signal.stylistic_uniformity_score for signal in original_signals)
    original_cliche_density = _average(signal.academic_cliche_density for signal in original_signals)
    revised_readability = [
        analyze_paragraph_readability_sentences(list(stats.rewritten_sentences)) for stats in checked
    ]
    original_readability = [
        analyze_paragraph_readability_sentences(list(stats.original_sentences)) for stats in checked
    ]
    average_completeness = _average(signal.sentence_completeness_score for signal in revised_readability)
    average_paragraph_readability = _average(signal.paragraph_readability_score for signal in revised_readability)
    original_completeness = _average(signal.sentence_completeness_score for signal in original_readability)
    original_paragraph_readability = _average(signal.paragraph_readability_score for signal in original_readability)
    completeness_preserved = average_completeness >= max(0.72, original_completeness - 0.06)
    readability_preserved = average_paragraph_readability >= max(0.70, original_paragraph_readability - 0.06)
    if not completeness_preserved or not readability_preserved:
        average_realism = min(average_realism, 0.47)
    improved_uniformity = sum(
        1
        for original, revised in zip(original_signals, revised_signals, strict=False)
        if revised.sentence_cadence_irregularity >= original.sentence_cadence_irregularity
        or revised.local_discourse_flatness <= original.local_discourse_flatness
    )
    low_realism_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, revised_signals, strict=False)
        if signal.revision_realism_score < 0.42 and signal.sentence_count >= 2
    ]
    flat_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, revised_signals, strict=False)
        if signal.local_discourse_flatness > 0.62 and signal.sentence_count >= 3
    ]
    rigid_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, revised_signals, strict=False)
        if signal.sentence_transition_rigidity > 0.30 and signal.sentence_count >= 2
    ]
    uniform_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, revised_signals, strict=False)
        if signal.stylistic_uniformity_score > 0.58 and signal.sentence_count >= 2
    ]
    low_texture_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, revised_signals, strict=False)
        if signal.support_sentence_texture_variation < 0.28 and signal.sentence_count >= 3
    ]
    cliche_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, revised_signals, strict=False)
        if signal.academic_cliche_density > 0.22
    ]
    high_sensitivity_cliche_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, revised_signals, strict=False)
        if bool(getattr(stats, "high_sensitivity_prose", False)) and signal.academic_cliche_density > 0.14
    ]
    voice_signatures = [signal.paragraph_voice_signature for signal in revised_signals if signal.paragraph_voice_signature]
    voice_counts = Counter(voice_signatures)
    paragraph_voice_variation = (
        1.0
        if len(voice_signatures) < 3
        else len(voice_counts) / len(voice_signatures)
    )
    dominant_voice_ratio = (
        0.0
        if not voice_signatures
        else max(voice_counts.values(), default=0) / len(voice_signatures)
    )
    paragraphs_checked = len(checked)
    varied_support_paragraphs = sum(
        1 for signal in revised_signals if signal.support_sentence_texture_variation >= 0.34 or signal.sentence_count <= 2
    )
    stylistic_uniformity_controlled = (
        average_uniformity <= max(0.52, original_uniformity + 0.04)
        and len(uniform_ids) <= max(2, paragraphs_checked // 5)
    )
    support_sentence_texture_varied = (
        average_texture_variation >= 0.30
        or varied_support_paragraphs >= max(1, paragraphs_checked // 3)
    ) and len(low_texture_ids) <= max(2, paragraphs_checked // 4)
    paragraph_voice_variation_present = (
        len(voice_signatures) < 3
        or (paragraph_voice_variation >= 0.35 and dominant_voice_ratio <= 0.72)
    )
    academic_cliche_density_controlled = (
        average_cliche_density <= max(0.18, original_cliche_density + 0.03)
        and len(cliche_ids) <= max(2, paragraphs_checked // 6)
        and not high_sensitivity_cliche_ids
    )
    return {
        "paragraphs_checked": paragraphs_checked,
        "sentence_transition_rigidity": round(average_rigidity, 4),
        "local_discourse_flatness": round(average_flatness, 4),
        "revision_realism_score": round(average_realism, 4),
        "sentence_cadence_irregularity": round(average_cadence, 4),
        "stylistic_uniformity_score": round(average_uniformity, 4),
        "support_sentence_texture_variation": round(average_texture_variation, 4),
        "paragraph_voice_variation": round(paragraph_voice_variation, 4),
        "academic_cliche_density": round(average_cliche_density, 4),
        "original_local_discourse_flatness": round(original_flatness, 4),
        "original_sentence_cadence_irregularity": round(original_cadence, 4),
        "original_stylistic_uniformity_score": round(original_uniformity, 4),
        "original_academic_cliche_density": round(original_cliche_density, 4),
        "sentence_completeness_score": round(average_completeness, 4),
        "paragraph_readability_score": round(average_paragraph_readability, 4),
        "readability_preserved": readability_preserved,
        "sentence_completeness_preserved": completeness_preserved,
        "local_transition_natural": average_rigidity <= 0.48,
        "local_discourse_not_flat": average_flatness <= 0.56 and len(flat_ids) <= max(2, paragraphs_checked // 5),
        "sentence_uniformity_reduced": (
            improved_uniformity >= max(1, paragraphs_checked // 2)
            or average_cadence >= original_cadence + 0.015
            or bool(action_bonus_ids)
        ),
        "revision_realism_present": average_realism >= 0.48 and len(low_realism_ids) <= max(2, paragraphs_checked // 4),
        "stylistic_uniformity_controlled": stylistic_uniformity_controlled,
        "support_sentence_texture_varied": support_sentence_texture_varied,
        "paragraph_voice_variation_present": paragraph_voice_variation_present,
        "academic_cliche_density_controlled": academic_cliche_density_controlled,
        "low_realism_paragraph_ids": low_realism_ids,
        "flat_paragraph_ids": flat_ids,
        "rigid_paragraph_ids": rigid_ids,
        "uniform_paragraph_ids": uniform_ids,
        "low_texture_variation_paragraph_ids": low_texture_ids,
        "cliche_dense_paragraph_ids": cliche_ids,
        "high_sensitivity_cliche_paragraph_ids": high_sensitivity_cliche_ids,
        "local_realism_action_paragraph_ids": action_bonus_ids,
    }


def _visible_length(sentence: str) -> int:
    return len(_SPACE_RE.sub("", sentence))


def _cadence_irregularity(lengths: list[int]) -> float:
    if not lengths:
        return 0.0
    if len(lengths) == 1:
        return 0.35
    mean = sum(lengths) / len(lengths)
    if mean <= 0:
        return 0.0
    variance = sum((length - mean) ** 2 for length in lengths) / len(lengths)
    coefficient = math.sqrt(variance) / mean
    short_long_mix = 0.14 if min(lengths) <= mean * 0.72 and max(lengths) >= mean * 1.20 else 0.0
    return round(_clamp(coefficient * 2.25 + short_long_mix), 4)


def _max_opener_family_ratio(sentences: list[str]) -> float:
    families: dict[str, int] = {}
    for sentence in sentences:
        opener = _opener_family(sentence)
        if not opener:
            continue
        families[opener] = families.get(opener, 0) + 1
    return max(families.values(), default=0) / max(1, len(sentences))


def _opener_family(sentence: str) -> str:
    stripped = sentence.strip()
    transition = _TRANSITION_RE.match(stripped)
    if transition:
        return transition.group(0).strip("，, ")
    for marker in ("本研究", "本文", "该系统", "系统", "该模型", "模型", "这一", "该"):
        if stripped.startswith(marker):
            return marker
    return ""


def _hierarchy_marker_ratio(sentences: list[str]) -> float:
    markers = ("其中", "尤其", "更具体", "这使", "这意味着", "换言之", "相较之下", "但", "不过", "一方面")
    return sum(1 for sentence in sentences if any(marker in sentence for marker in markers)) / max(1, len(sentences))


def _support_sentence_textures(sentences: list[str]) -> list[str]:
    textures: list[str] = []
    for index, sentence in enumerate(sentences):
        if index == 0:
            textures.append("topic")
            continue
        if index == len(sentences) - 1:
            textures.append("conclusion")
            continue
        stripped = sentence.strip()
        if _TEXTURE_SUPPLEMENT_RE.match(stripped) or _visible_length(stripped) < 22:
            textures.append("supplement")
        elif _ACADEMIC_CLICHE_COUNT(stripped) > 0:
            textures.append("cliche")
        elif _TEXTURE_JUDGMENT_RE.search(stripped):
            textures.append("judgment")
        elif _TEXTURE_EXPLANATORY_RE.search(stripped) or any(marker in stripped for marker in ("，", "；", "：")):
            textures.append("explanatory")
        else:
            textures.append("plain")
    return textures


def _support_texture_variation(textures: list[str]) -> float:
    supports = [texture for texture in textures[1:-1] if texture] if len(textures) > 2 else [texture for texture in textures[1:] if texture]
    if not supports:
        return 0.48
    if len(supports) == 1:
        return 0.56 if supports[0] in {"plain", "explanatory", "judgment"} else 0.46
    unique_ratio = (len(set(supports)) - 1) / max(1, len(supports) - 1)
    plain_presence = 0.12 if "plain" in supports else 0.0
    supplement_presence = 0.08 if "supplement" in supports else 0.0
    non_cliche_presence = 0.10 if not all(texture == "cliche" for texture in supports) else 0.0
    return _clamp(unique_ratio * 0.68 + plain_presence + supplement_presence + non_cliche_presence)


def _academic_cliche_density(sentences: list[str]) -> float:
    occurrences = sum(_ACADEMIC_CLICHE_COUNT(sentence) for sentence in sentences)
    return _clamp(occurrences / max(1, len(sentences) * 2))


def _ACADEMIC_CLICHE_COUNT(sentence: str) -> int:
    return sum(1 for pattern in _ACADEMIC_CLICHE_PATTERNS if pattern.search(sentence))


def _paragraph_voice_signature(
    sentences: list[str],
    textures: list[str],
    cadence: float,
    cliche_density: float,
) -> str:
    opener = _opener_family(sentences[0]) or "direct"
    support_signature = "-".join(texture for texture in textures[1:3] if texture) or "plain"
    cadence_band = "high" if cadence >= 0.60 else ("mid" if cadence >= 0.38 else "low")
    cliche_band = "cliche" if cliche_density >= 0.18 else "plain"
    return f"{opener}|{support_signature}|{cadence_band}|{cliche_band}"


def _average(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(float(item) for item in items) / len(items)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def technical_density_is_high(sentence: str) -> bool:
    """Return true when a sentence is too technical for naturalness rewrites."""

    return len(_TECH_MARKER_RE.findall(sentence)) >= 3
