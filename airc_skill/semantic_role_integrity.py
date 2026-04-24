from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_SCAFFOLDING_PATTERNS = (
    re.compile(r"^(?:这一工作|这项工作|该段论述|相关内容|该部分|这一部分|这一设置|这一调整|这一诊断结果)"),
    re.compile(r"^(?:这一点需要指出|这一点需要说明|这一点需要强调)"),
    re.compile(r"^(?:在此基础上，?研究|相应地，?本文|由此，?研究)"),
    re.compile(r"(?:还包括|也包括|进一步包括|相关内容还包括|相关内容包括|进一步还包括)"),
)
_ABSTRACTED_SUBJECT_RE = re.compile(
    r"^(?:这一工作|这项工作|这一点|这一机制|这一设计|该部分|相关内容|这一部分|这一设置|"
    r"这一调整|这一诊断结果|这一策略|这一过程|这一路径)[，,\s]*"
)
_ENUMERATION_ITEM_RE = re.compile(r"^(?:[（(]?\d+[）)]|第一|第二|第三|第四|第五|第六|其一|其二|其三)[，:\s]*")
_ENUMERATION_HEAD_RE = re.compile(
    r"(?:主要包括以下|主要分为|主要包含|主要创新点如下|创新点如下|主要内容如下|包括以下几类|"
    r"包含三类关键组成部分|主要包含三类关键组成部分)"
)
_MECHANISM_RE = re.compile(
    r"(?:模型|系统|模块|分支|接口|机制|路径|判别机制|课程学习|损失函数|主融合模块|语义分支|频域分支|"
    r"base_only|logit|特征|决策|推理)"
)
_MECHANISM_VERB_RE = re.compile(
    r"(?:采用|通过|将|把|用于|负责|生成|输出|融合|提取|建模|约束|对外提供|保留|剥离|实施|"
    r"构成|形成|限制|连接|由[^。；]{0,24}得到)"
)
_CONCLUSION_HEAD_RE = re.compile(
    r"^(?:总结而言|总的来说|总体而言|总体来看|整体来看|综上|由此可见|从最终效果看|从整体上看|"
    r"从整体上说)[，,\s]*"
)
_APPENDIX_LIKE_SUPPORT_RE = re.compile(
    r"^(?:还包括|也包括|进一步包括|相关内容包括|相关内容还包括|这一(?:机制|设计|工作|部分|路径)|"
    r"该(?:机制|设计|部分)).*(?:还包括|也包括|进一步)"
)


@dataclass(frozen=True)
class SemanticRoleSignals:
    semantic_role_integrity_score: float
    enumeration_integrity_score: float
    scaffolding_phrase_density: float
    over_abstracted_subject_risk: float
    has_enumeration: bool
    semantic_role_drift: bool
    enumeration_drift: bool

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "semantic_role_integrity_score": self.semantic_role_integrity_score,
            "enumeration_integrity_score": self.enumeration_integrity_score,
            "scaffolding_phrase_density": self.scaffolding_phrase_density,
            "over_abstracted_subject_risk": self.over_abstracted_subject_risk,
            "has_enumeration": self.has_enumeration,
            "semantic_role_drift": self.semantic_role_drift,
            "enumeration_drift": self.enumeration_drift,
        }


def analyze_semantic_role_integrity(
    original_sentences: list[str],
    revised_sentences: list[str],
    *,
    high_sensitivity: bool = False,
) -> SemanticRoleSignals:
    """Assess whether rewrite output preserves semantic role and enumeration behavior."""

    original_visible = [sentence.strip() for sentence in original_sentences if sentence.strip()]
    revised_visible = [sentence.strip() for sentence in revised_sentences if sentence.strip()]
    if not revised_visible:
        return SemanticRoleSignals(0.0, 0.0, 0.0, 0.0, False, True, bool(original_visible))

    original_roles = [
        _sentence_role(sentence, index=index, total=len(original_visible))
        for index, sentence in enumerate(original_visible)
    ]
    revised_roles = [
        _sentence_role(sentence, index=index, total=len(revised_visible))
        for index, sentence in enumerate(revised_visible)
    ]
    has_enumeration = any(role == "enumeration" for role in original_roles) or any(
        _looks_like_enumeration_sentence(sentence) for sentence in original_visible
    )
    original_enumeration_count = sum(1 for role in original_roles if role == "enumeration")
    revised_enumeration_count = sum(1 for role in revised_roles if role == "enumeration")
    semantic_weight = 0.0
    semantic_penalty = 0.0
    aligned = max(len(original_visible), len(revised_visible))

    for index in range(aligned):
        original_sentence = original_visible[index] if index < len(original_visible) else ""
        revised_sentence = revised_visible[index] if index < len(revised_visible) else ""
        original_role = original_roles[index] if index < len(original_roles) else "support"
        revised_role = revised_roles[index] if index < len(revised_roles) else "support"
        weight = _role_weight(original_role)
        semantic_weight += weight
        if not _role_compatible(
            original_role,
            revised_role,
            original_sentence,
            revised_sentence,
            revised_enumeration_count=revised_enumeration_count,
            original_enumeration_count=original_enumeration_count,
        ):
            semantic_penalty += weight
        if original_role == "mechanism" and _APPENDIX_LIKE_SUPPORT_RE.search(revised_sentence):
            semantic_penalty += 0.35
        if original_role == "conclusion" and revised_role == "support":
            semantic_penalty += 0.25

    scaffolding_density = round(_scaffolding_phrase_density(revised_visible), 4)
    abstracted_subject_risk = round(_over_abstracted_subject_risk(revised_visible), 4)
    semantic_score = 1.0 - (semantic_penalty / max(1.0, semantic_weight))
    semantic_score -= scaffolding_density * (0.24 if high_sensitivity else 0.16)
    semantic_score -= abstracted_subject_risk * 0.18
    semantic_score = round(_clamp(semantic_score), 4)

    if has_enumeration:
        enumeration_penalty = 0.0
        if revised_enumeration_count < original_enumeration_count:
            enumeration_penalty += min(0.65, (original_enumeration_count - revised_enumeration_count) * 0.22)
        appendix_like_enum_count = sum(
            1 for sentence in revised_visible if _looks_like_enumeration_sentence(sentence) and _APPENDIX_LIKE_SUPPORT_RE.search(sentence)
        )
        if appendix_like_enum_count:
            enumeration_penalty += min(0.45, appendix_like_enum_count * 0.18)
        collapse_penalty = 0.18 if any(
            re.search(r"第一，.+?(?:，进而|，随后|，接着)?第二，", sentence) for sentence in revised_visible
        ) else 0.0
        enumeration_score = round(_clamp(1.0 - min(1.0, enumeration_penalty + collapse_penalty)), 4)
    else:
        enumeration_score = 1.0

    return SemanticRoleSignals(
        semantic_role_integrity_score=semantic_score,
        enumeration_integrity_score=enumeration_score,
        scaffolding_phrase_density=scaffolding_density,
        over_abstracted_subject_risk=abstracted_subject_risk,
        has_enumeration=has_enumeration,
        semantic_role_drift=semantic_score < (0.68 if high_sensitivity else 0.64),
        enumeration_drift=has_enumeration and enumeration_score < 0.80,
    )


def aggregate_semantic_role_integrity(rewrite_stats: list[Any]) -> dict[str, object]:
    """Aggregate semantic-role preservation signals across changed body paragraphs."""

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
            "semantic_role_integrity_score": 1.0,
            "enumeration_integrity_score": 1.0,
            "scaffolding_phrase_density": 0.0,
            "over_abstracted_subject_risk": 0.0,
            "semantic_role_integrity_preserved": True,
            "enumeration_integrity_preserved": True,
            "scaffolding_phrase_density_controlled": True,
            "over_abstracted_subject_risk_controlled": True,
            "semantic_role_drift_paragraph_ids": [],
            "enumeration_drift_paragraph_ids": [],
            "scaffolding_paragraph_ids": [],
            "high_sensitivity_scaffolding_paragraph_ids": [],
            "abstracted_subject_paragraph_ids": [],
        }

    signals = [
        analyze_semantic_role_integrity(
            list(getattr(stats, "original_sentences", [])),
            list(getattr(stats, "rewritten_sentences", [])),
            high_sensitivity=bool(getattr(stats, "high_sensitivity_prose", False)),
        )
        for stats in checked
    ]
    semantic_role_scores: list[float] = []
    for stats, signal in zip(checked, signals, strict=False):
        action_bonus = any(
            action in {"conclusion_absorb", "conclusion_absorption", "sentence_cluster_merge", "subject_chain_compression", "meta_compression"}
            for action in getattr(stats, "structural_actions", [])
        ) or any(
            action in {"conclusion_absorb", "conclusion_absorption", "sentence_cluster_rewrite", "sentence_cluster_merge", "meta_compression"}
            for action in getattr(stats, "discourse_actions_used", [])
        )
        adjusted_score = signal.semantic_role_integrity_score
        if action_bonus and adjusted_score >= 0.55:
            adjusted_score = min(1.0, adjusted_score + 0.18)
        semantic_role_scores.append(adjusted_score)
    avg_semantic_role = round(_average(semantic_role_scores), 4)
    enum_signals = [signal.enumeration_integrity_score for signal in signals if signal.has_enumeration]
    avg_enumeration = round(_average(enum_signals) if enum_signals else 1.0, 4)
    avg_scaffolding = round(_average(signal.scaffolding_phrase_density for signal in signals), 4)
    avg_abstracted_subject = round(_average(signal.over_abstracted_subject_risk for signal in signals), 4)
    semantic_drift_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, signals, strict=False)
        if signal.semantic_role_drift
        and not any(
            action in {"conclusion_absorb", "conclusion_absorption", "sentence_cluster_merge", "subject_chain_compression", "meta_compression"}
            for action in [*getattr(stats, "structural_actions", []), *getattr(stats, "discourse_actions_used", [])]
        )
    ]
    enumeration_drift_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, signals, strict=False)
        if signal.enumeration_drift
    ]
    scaffolding_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, signals, strict=False)
        if signal.scaffolding_phrase_density > 0.10
    ]
    high_sensitivity_scaffolding_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, signals, strict=False)
        if bool(getattr(stats, "high_sensitivity_prose", False)) and signal.scaffolding_phrase_density > 0.04
    ]
    abstracted_subject_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, signals, strict=False)
        if signal.over_abstracted_subject_risk > 0.10
    ]
    paragraphs_checked = len(checked)
    return {
        "paragraphs_checked": paragraphs_checked,
        "semantic_role_integrity_score": avg_semantic_role,
        "enumeration_integrity_score": avg_enumeration,
        "scaffolding_phrase_density": avg_scaffolding,
        "over_abstracted_subject_risk": avg_abstracted_subject,
        "semantic_role_integrity_preserved": (
            avg_semantic_role >= 0.78 and len(semantic_drift_ids) <= max(6, paragraphs_checked // 3)
        ),
        "enumeration_integrity_preserved": (
            not enum_signals
            or (avg_enumeration >= 0.82 and len(enumeration_drift_ids) <= max(1, paragraphs_checked // 6))
        ),
        "scaffolding_phrase_density_controlled": (
            avg_scaffolding <= 0.08
            and len(scaffolding_ids) <= max(2, paragraphs_checked // 6)
            and not high_sensitivity_scaffolding_ids
        ),
        "over_abstracted_subject_risk_controlled": (
            avg_abstracted_subject <= 0.08
            and len(abstracted_subject_ids) <= max(2, paragraphs_checked // 5)
        ),
        "semantic_role_drift_paragraph_ids": semantic_drift_ids,
        "enumeration_drift_paragraph_ids": enumeration_drift_ids,
        "scaffolding_paragraph_ids": scaffolding_ids,
        "high_sensitivity_scaffolding_paragraph_ids": high_sensitivity_scaffolding_ids,
        "abstracted_subject_paragraph_ids": abstracted_subject_ids,
    }


def _sentence_role(sentence: str, *, index: int, total: int) -> str:
    stripped = sentence.strip()
    if not stripped:
        return "support"
    if _looks_like_enumeration_sentence(stripped):
        return "enumeration"
    if _CONCLUSION_HEAD_RE.match(stripped):
        return "conclusion"
    if index == total - 1 and re.search(r"(说明|表明|意味着|更有效|更稳定|更关键|更有利)", stripped):
        return "conclusion"
    if index == 0 and re.match(r"^(?:本研究|本文|本章|本系统|本工作|本项目)", stripped):
        return "core"
    if _MECHANISM_RE.search(stripped) and _MECHANISM_VERB_RE.search(stripped):
        return "mechanism"
    return "support"


def _looks_like_enumeration_sentence(sentence: str) -> bool:
    stripped = sentence.strip()
    if not stripped:
        return False
    return bool(_ENUMERATION_ITEM_RE.match(stripped) or _ENUMERATION_HEAD_RE.search(stripped))


def _role_weight(role: str) -> float:
    return {
        "core": 1.2,
        "mechanism": 1.3,
        "enumeration": 1.5,
        "conclusion": 1.2,
        "support": 0.8,
    }.get(role, 0.8)


def _role_compatible(
    original_role: str,
    revised_role: str,
    original_sentence: str,
    revised_sentence: str,
    *,
    revised_enumeration_count: int,
    original_enumeration_count: int,
) -> bool:
    if original_role == revised_role:
        return True
    if original_role == "support":
        return True
    if original_role == "core":
        return revised_role in {"core", "support"} and not _ABSTRACTED_SUBJECT_RE.match(revised_sentence.strip())
    if original_role == "conclusion":
        return revised_role in {"conclusion", "support"} and not _APPENDIX_LIKE_SUPPORT_RE.search(revised_sentence)
    if original_role == "mechanism":
        return revised_role == "mechanism" and not _APPENDIX_LIKE_SUPPORT_RE.search(revised_sentence)
    if original_role == "enumeration":
        return (
            (_looks_like_enumeration_sentence(revised_sentence) and not _APPENDIX_LIKE_SUPPORT_RE.search(revised_sentence))
            or revised_enumeration_count >= original_enumeration_count
        )
    return False


def _scaffolding_phrase_density(sentences: list[str]) -> float:
    count = 0
    for sentence in sentences:
        stripped = sentence.strip()
        if any(pattern.search(stripped) for pattern in _SCAFFOLDING_PATTERNS):
            count += 1
    return _clamp(count / max(1, len(sentences)))


def _over_abstracted_subject_risk(sentences: list[str]) -> float:
    count = 0
    for sentence in sentences:
        stripped = sentence.strip()
        if _ABSTRACTED_SUBJECT_RE.match(stripped):
            count += 1
    return _clamp(count / max(1, len(sentences)))


def _average(values: list[float] | Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))
