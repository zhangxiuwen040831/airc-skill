from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .local_revision_realism import technical_density_is_high

_UNSUPPORTED_EXPANSION_PATTERNS = (
    re.compile(r"\b20\d{2}年(?:AIGC|生成式|内容生态|传播环境|舆论场|虚假信息)"),
    re.compile(r"(?:超过|高于|接近)(?:八成|半数|九成|七成|六成|五成|四成|三成)"),
    re.compile(r"(?:绝大多数|大多数|主流观点|业界通常认为|普遍认为|领域共识|业内普遍认为)"),
    re.compile(r"(?:投诉|内容生态|真实世界中|行业现状|平台治理|运营人员)"),
)
_COMMENTARY_TONE_PATTERNS = (
    re.compile(r"(?:正如[^。；]{0,40}所认为)"),
    re.compile(r"(?:终于摆脱|终于脱离|挣脱出来|被击穿|逼近满分|拖进来|走向终点)"),
    re.compile(r"(?:宁可错杀|宁可放过)"),
    re.compile(r"(?:如同|仿佛)[^。；]{0,36}(?:手术|判决书|幻想|剧情|故事)"),
)
_METAPHOR_PATTERNS = (
    re.compile(r"(?:幻想的破灭|边界修复手术|判决书|挣脱出来|终于摆脱|路径终点)"),
    re.compile(r"(?:如同一次|像一次)[^。；]{0,24}(?:手术|修复|判决|拉锯|较量)"),
)
_UNJUSTIFIED_AUTHORIAL_PATTERNS = (
    re.compile(r"^(?:我们|我们的)[，,\s]*"),
    re.compile(r"(?:本工作证明了|这项工作的终点|我们的工程价值|我们选择)"),
    re.compile(r"(?:主流观点|业界通常认为|普遍认为)"),
)
_ACADEMIC_REGISTER_PATTERNS = (
    re.compile(r"^(?:本研究|本文|研究|模型|系统|语义分支|频域分支|噪声分支|课程采样器|判别路径|实验|结果)"),
    re.compile(r"(?:表明|说明|显示|采用|通过|将|把|负责|实现|约束|缓解|保持|比较|构建|设计)"),
)


@dataclass(frozen=True)
class EvidenceFidelitySignals:
    evidence_fidelity_score: float
    unsupported_expansion_risk: float
    thesis_tone_restraint_score: float
    metaphor_or_storytelling_risk: float
    unjustified_authorial_claim_risk: float
    evidence_drift: bool
    unsupported_expansion_visible: bool
    thesis_tone_drift: bool
    metaphor_visible: bool
    authorial_claim_visible: bool

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "evidence_fidelity_score": self.evidence_fidelity_score,
            "unsupported_expansion_risk": self.unsupported_expansion_risk,
            "thesis_tone_restraint_score": self.thesis_tone_restraint_score,
            "metaphor_or_storytelling_risk": self.metaphor_or_storytelling_risk,
            "unjustified_authorial_claim_risk": self.unjustified_authorial_claim_risk,
            "evidence_drift": self.evidence_drift,
            "unsupported_expansion_visible": self.unsupported_expansion_visible,
            "thesis_tone_drift": self.thesis_tone_drift,
            "metaphor_visible": self.metaphor_visible,
            "authorial_claim_visible": self.authorial_claim_visible,
        }


def analyze_evidence_fidelity(
    original_sentences: list[str],
    revised_sentences: list[str],
    *,
    high_sensitivity: bool = False,
) -> EvidenceFidelitySignals:
    """Assess whether revision stays inside the source evidence boundary and thesis register."""

    original_visible = [sentence.strip() for sentence in original_sentences if sentence.strip()]
    revised_visible = [sentence.strip() for sentence in revised_sentences if sentence.strip()]
    if not revised_visible:
        return EvidenceFidelitySignals(0.0, 0.0, 0.0, 0.0, 0.0, True, False, True, False, False)

    unsupported_hits = _residual_pattern_count(
        original_visible,
        revised_visible,
        _UNSUPPORTED_EXPANSION_PATTERNS,
    )
    commentary_hits = _residual_pattern_count(
        original_visible,
        revised_visible,
        _COMMENTARY_TONE_PATTERNS,
    )
    metaphor_hits = _residual_pattern_count(
        original_visible,
        revised_visible,
        _METAPHOR_PATTERNS,
    )
    authorial_claim_hits = _residual_pattern_count(
        original_visible,
        revised_visible,
        _UNJUSTIFIED_AUTHORIAL_PATTERNS,
    )
    unsupported_risk = round(_clamp(unsupported_hits / max(1, len(revised_visible))), 4)
    metaphor_risk = round(_clamp((metaphor_hits + commentary_hits * 0.5) / max(1, len(revised_visible))), 4)
    authorial_claim_risk = round(_clamp(authorial_claim_hits / max(1, len(revised_visible))), 4)

    thesis_register_score = _thesis_register_score(revised_visible)
    thesis_register_score -= unsupported_risk * (0.34 if high_sensitivity else 0.24)
    thesis_register_score -= metaphor_risk * 0.30
    thesis_register_score -= authorial_claim_risk * 0.22
    thesis_register_score = round(_clamp(thesis_register_score), 4)

    evidence_score = 0.94
    evidence_score -= unsupported_risk * (0.58 if high_sensitivity else 0.44)
    evidence_score -= metaphor_risk * 0.24
    evidence_score -= authorial_claim_risk * 0.18
    if not original_visible:
        evidence_score -= 0.06
    evidence_score = round(_clamp(evidence_score), 4)

    return EvidenceFidelitySignals(
        evidence_fidelity_score=evidence_score,
        unsupported_expansion_risk=unsupported_risk,
        thesis_tone_restraint_score=thesis_register_score,
        metaphor_or_storytelling_risk=metaphor_risk,
        unjustified_authorial_claim_risk=authorial_claim_risk,
        evidence_drift=evidence_score < (0.78 if high_sensitivity else 0.74),
        unsupported_expansion_visible=unsupported_risk > (0.04 if high_sensitivity else 0.06),
        thesis_tone_drift=thesis_register_score < (0.82 if high_sensitivity else 0.78),
        metaphor_visible=metaphor_risk > (0.02 if high_sensitivity else 0.04),
        authorial_claim_visible=authorial_claim_risk > (0.03 if high_sensitivity else 0.05),
    )


def aggregate_evidence_fidelity(rewrite_stats: list[Any]) -> dict[str, object]:
    """Aggregate evidence-boundary and thesis-register signals across rewritten body paragraphs."""

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
            "evidence_fidelity_score": 1.0,
            "unsupported_expansion_risk": 0.0,
            "thesis_tone_restraint_score": 1.0,
            "metaphor_or_storytelling_risk": 0.0,
            "unjustified_authorial_claim_risk": 0.0,
            "evidence_fidelity_preserved": True,
            "unsupported_expansion_controlled": True,
            "thesis_tone_restrained": True,
            "metaphor_or_storytelling_controlled": True,
            "authorial_claim_risk_controlled": True,
            "evidence_drift_paragraph_ids": [],
            "unsupported_expansion_paragraph_ids": [],
            "high_sensitivity_unsupported_paragraph_ids": [],
            "metaphor_paragraph_ids": [],
            "authorial_claim_paragraph_ids": [],
        }

    signals = [
        analyze_evidence_fidelity(
            list(getattr(stats, "original_sentences", [])),
            list(getattr(stats, "rewritten_sentences", [])),
            high_sensitivity=bool(getattr(stats, "high_sensitivity_prose", False)),
        )
        for stats in checked
    ]
    average_evidence = round(_average(signal.evidence_fidelity_score for signal in signals), 4)
    average_unsupported = round(_average(signal.unsupported_expansion_risk for signal in signals), 4)
    average_tone = round(_average(signal.thesis_tone_restraint_score for signal in signals), 4)
    average_metaphor = round(_average(signal.metaphor_or_storytelling_risk for signal in signals), 4)
    average_authorial_claim = round(_average(signal.unjustified_authorial_claim_risk for signal in signals), 4)
    evidence_drift_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, signals, strict=False)
        if signal.evidence_drift
    ]
    unsupported_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, signals, strict=False)
        if signal.unsupported_expansion_visible
    ]
    high_sensitivity_unsupported_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, signals, strict=False)
        if bool(getattr(stats, "high_sensitivity_prose", False)) and signal.unsupported_expansion_visible
    ]
    metaphor_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, signals, strict=False)
        if signal.metaphor_visible
    ]
    authorial_claim_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, signals, strict=False)
        if signal.authorial_claim_visible
    ]
    paragraphs_checked = len(checked)
    return {
        "paragraphs_checked": paragraphs_checked,
        "evidence_fidelity_score": average_evidence,
        "unsupported_expansion_risk": average_unsupported,
        "thesis_tone_restraint_score": average_tone,
        "metaphor_or_storytelling_risk": average_metaphor,
        "unjustified_authorial_claim_risk": average_authorial_claim,
        "evidence_fidelity_preserved": (
            average_evidence >= 0.84
            and len(evidence_drift_ids) <= max(3, paragraphs_checked // 5)
        ),
        "unsupported_expansion_controlled": (
            average_unsupported <= 0.05
            and len(unsupported_ids) <= max(2, paragraphs_checked // 6)
            and not high_sensitivity_unsupported_ids
        ),
        "thesis_tone_restrained": (
            average_tone >= 0.84
            and len(metaphor_ids) <= max(1, paragraphs_checked // 8)
        ),
        "metaphor_or_storytelling_controlled": (
            average_metaphor <= 0.03
            and len(metaphor_ids) <= max(1, paragraphs_checked // 10)
        ),
        "authorial_claim_risk_controlled": (
            average_authorial_claim <= 0.04
            and len(authorial_claim_ids) <= max(1, paragraphs_checked // 10)
        ),
        "evidence_drift_paragraph_ids": evidence_drift_ids,
        "unsupported_expansion_paragraph_ids": unsupported_ids,
        "high_sensitivity_unsupported_paragraph_ids": high_sensitivity_unsupported_ids,
        "metaphor_paragraph_ids": metaphor_ids,
        "authorial_claim_paragraph_ids": authorial_claim_ids,
    }


def _residual_pattern_count(
    original_sentences: list[str],
    revised_sentences: list[str],
    patterns: tuple[re.Pattern[str], ...],
) -> int:
    original_count = sum(sum(1 for _ in pattern.finditer(sentence)) for pattern in patterns for sentence in original_sentences)
    revised_count = sum(sum(1 for _ in pattern.finditer(sentence)) for pattern in patterns for sentence in revised_sentences)
    return max(0, revised_count - original_count)


def _thesis_register_score(sentences: list[str]) -> float:
    if not sentences:
        return 0.0
    score = 0.78
    academic_hits = sum(1 for sentence in sentences if any(pattern.search(sentence) for pattern in _ACADEMIC_REGISTER_PATTERNS))
    if academic_hits:
        score += min(0.16, academic_hits / max(1, len(sentences)) * 0.18)
    if any(technical_density_is_high(sentence) for sentence in sentences):
        score += 0.04
    return _clamp(score)


def _average(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))
