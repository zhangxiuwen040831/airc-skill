from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .local_revision_realism import technical_density_is_high

_APPENDIX_LIKE_LEAD_RE = re.compile(
    r"^(?:用于进一步|从[^。；]{2,20}角度来看|在该设置下|在当前设置下|通过这种方式|"
    r"结合[^。；]{2,40}可以看出|可以看出|这表明)[，,\s]*"
)
_WEAK_MODAL_RE = re.compile(r"(?:可以|能够|用于|有助于)")
_STRONG_VERB_RE = re.compile(
    r"(?:直接|显式|负责|实现|形成|约束|限制|剥离|建模|融合|输出|保留|引入|完成|"
    r"比较|设计|压缩|避免|选择|决定)"
)
_CONCRETE_SUBJECT_RE = re.compile(
    r"^(?:模型|系统|语义分支|频域分支|噪声分支|主融合模块|融合模块|控制机制|课程采样器|"
    r"判别路径|基础对数几率|默认部署模式|扩展分析模式|训练|实验|NTIRE数据集|photos_test集合|"
    r"该结构|该分支|该模块|该判别路径|该课程采样器|后续研究)"
)
_ABSTRACT_SUBJECT_RE = re.compile(
    r"^(?:本研究|本文|该方法|该机制|这种方式|该设置|这一路径|这一设计|这一机制|该系统)"
)
_MECHANISM_RE = re.compile(
    r"(?:模型|系统|模块|分支|接口|机制|路径|判别机制|课程学习|损失函数|主融合模块|语义分支|"
    r"频域分支|噪声分支|base_only|logit|特征|决策|推理)"
)
_MECHANISM_VERB_RE = re.compile(
    r"(?:采用|通过|将|把|用于|负责|生成|输出|融合|提取|建模|约束|对外提供|保留|剥离|"
    r"实施|构成|形成|限制|连接|刻画|验证|观察)"
)
_STANCE_RE = re.compile(
    r"(?:相比之下|不同于|并非|而是|更关键的是|关键在于|核心在于|本质是|本质在于|"
    r"本研究选择|本工作选择|选择[^。；]{1,48}而非[^。；]{1,48}|重点并非[^。；]{1,96}而是)"
)
_CONCLUSION_MARKER_RE = re.compile(r"^(?:可以看出|这表明|这说明|由此可见|综上)[，,\s]*")


@dataclass(frozen=True)
class AuthorialIntentSignals:
    assertion_strength_score: float
    appendix_like_support_ratio: float
    authorial_stance_presence: float
    weak_assertion: bool
    appendix_like_support_visible: bool
    stance_drop: bool

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "assertion_strength_score": self.assertion_strength_score,
            "appendix_like_support_ratio": self.appendix_like_support_ratio,
            "authorial_stance_presence": self.authorial_stance_presence,
            "weak_assertion": self.weak_assertion,
            "appendix_like_support_visible": self.appendix_like_support_visible,
            "stance_drop": self.stance_drop,
        }


def analyze_authorial_intent(
    original_sentences: list[str],
    revised_sentences: list[str],
    *,
    high_sensitivity: bool = False,
) -> AuthorialIntentSignals:
    """Assess whether rewritten prose keeps direct author-like assertion and stance."""

    original_visible = [sentence.strip() for sentence in original_sentences if sentence.strip()]
    revised_visible = [sentence.strip() for sentence in revised_sentences if sentence.strip()]
    if not revised_visible:
        return AuthorialIntentSignals(0.0, 0.0, 0.0, True, False, bool(original_visible))

    revised_assertion_scores = [
        _assertion_strength(sentence, index=index, total=len(revised_visible))
        for index, sentence in enumerate(revised_visible)
    ]
    original_assertion_scores = [
        _assertion_strength(sentence, index=index, total=len(original_visible))
        for index, sentence in enumerate(original_visible)
    ]
    appendix_count = sum(
        1
        for index, sentence in enumerate(revised_visible)
        if _is_appendix_like_support(sentence, role=_sentence_role(sentence, index=index, total=len(revised_visible)))
    )
    appendix_ratio = round(_clamp(appendix_count / max(1, len(revised_visible))), 4)
    original_stance_count = sum(1 for sentence in original_visible if _STANCE_RE.search(sentence))
    revised_stance_count = sum(1 for sentence in revised_visible if _STANCE_RE.search(sentence))
    if original_stance_count == 0:
        stance_presence = 1.0 if revised_stance_count == 0 else min(1.0, 0.72 + revised_stance_count * 0.08)
        stance_drop = False
    else:
        stance_presence = round(min(1.0, revised_stance_count / original_stance_count), 4)
        stance_drop = revised_stance_count < original_stance_count

    original_assertion = _average(original_assertion_scores) if original_assertion_scores else 0.72
    revised_assertion = _average(revised_assertion_scores)
    assertion_score = revised_assertion
    if revised_assertion + 0.04 < original_assertion:
        assertion_score -= min(0.12, original_assertion - revised_assertion)
    assertion_score -= appendix_ratio * (0.26 if high_sensitivity else 0.18)
    if stance_drop and original_stance_count:
        assertion_score -= 0.06
    assertion_score = round(_clamp(assertion_score), 4)

    return AuthorialIntentSignals(
        assertion_strength_score=assertion_score,
        appendix_like_support_ratio=appendix_ratio,
        authorial_stance_presence=round(_clamp(stance_presence), 4),
        weak_assertion=assertion_score < (0.66 if high_sensitivity else 0.62),
        appendix_like_support_visible=appendix_ratio > (0.06 if high_sensitivity else 0.08),
        stance_drop=stance_drop and stance_presence < 0.92,
    )


def aggregate_authorial_intent(rewrite_stats: list[Any]) -> dict[str, object]:
    """Aggregate author-like assertion and stance signals across changed body paragraphs."""

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
            "assertion_strength_score": 1.0,
            "appendix_like_support_ratio": 0.0,
            "authorial_stance_presence": 1.0,
            "original_assertion_strength_score": 1.0,
            "original_authorial_stance_presence": 1.0,
            "assertion_strength_preserved": True,
            "appendix_like_support_controlled": True,
            "authorial_stance_present": True,
            "weak_assertion_paragraph_ids": [],
            "appendix_like_paragraph_ids": [],
            "stance_drop_paragraph_ids": [],
            "high_sensitivity_appendix_like_paragraph_ids": [],
        }

    revised_signals = [
        analyze_authorial_intent(
            list(getattr(stats, "original_sentences", [])),
            list(getattr(stats, "rewritten_sentences", [])),
            high_sensitivity=bool(getattr(stats, "high_sensitivity_prose", False)),
        )
        for stats in checked
    ]
    original_assertion_strength = round(
        _average(
            _average(
                _assertion_strength(sentence, index=index, total=len(stats.original_sentences))
                for index, sentence in enumerate(stats.original_sentences)
            )
            if getattr(stats, "original_sentences", None)
            else 0.72
            for stats in checked
        ),
        4,
    )
    original_stance_presence = round(
        _average(
            1.0
            if not any(_STANCE_RE.search(sentence) for sentence in getattr(stats, "original_sentences", []))
            else 1.0
            for stats in checked
        ),
        4,
    )
    average_assertion = round(_average(signal.assertion_strength_score for signal in revised_signals), 4)
    average_appendix = round(_average(signal.appendix_like_support_ratio for signal in revised_signals), 4)
    average_stance = round(_average(signal.authorial_stance_presence for signal in revised_signals), 4)
    weak_assertion_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, revised_signals, strict=False)
        if signal.weak_assertion
    ]
    appendix_like_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, revised_signals, strict=False)
        if signal.appendix_like_support_visible
    ]
    stance_drop_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, revised_signals, strict=False)
        if signal.stance_drop
    ]
    high_sensitivity_appendix_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, revised_signals, strict=False)
        if bool(getattr(stats, "high_sensitivity_prose", False)) and signal.appendix_like_support_visible
    ]
    paragraphs_checked = len(checked)
    return {
        "paragraphs_checked": paragraphs_checked,
        "assertion_strength_score": average_assertion,
        "appendix_like_support_ratio": average_appendix,
        "authorial_stance_presence": average_stance,
        "original_assertion_strength_score": original_assertion_strength,
        "original_authorial_stance_presence": original_stance_presence,
        "assertion_strength_preserved": (
            average_assertion >= max(0.66, original_assertion_strength - 0.05)
            and len(weak_assertion_ids) <= max(3, paragraphs_checked // 4)
        ),
        "appendix_like_support_controlled": (
            average_appendix <= 0.08
            and len(appendix_like_ids) <= max(2, paragraphs_checked // 5)
            and not high_sensitivity_appendix_ids
        ),
        "authorial_stance_present": (
            average_stance >= 0.90
            and len(stance_drop_ids) <= max(2, paragraphs_checked // 6)
        ),
        "weak_assertion_paragraph_ids": weak_assertion_ids,
        "appendix_like_paragraph_ids": appendix_like_ids,
        "stance_drop_paragraph_ids": stance_drop_ids,
        "high_sensitivity_appendix_like_paragraph_ids": high_sensitivity_appendix_ids,
    }


def _sentence_role(sentence: str, *, index: int, total: int) -> str:
    stripped = sentence.strip()
    if not stripped:
        return "support"
    if index == 0 and re.match(r"^(?:本研究|本文|本章|本系统|本工作|本项目)", stripped):
        return "core"
    if index == total - 1 and (_CONCLUSION_MARKER_RE.match(stripped) or _STANCE_RE.search(stripped)):
        return "conclusion"
    if _MECHANISM_RE.search(stripped) and _MECHANISM_VERB_RE.search(stripped):
        return "mechanism"
    return "support"


def _assertion_strength(sentence: str, *, index: int, total: int) -> float:
    stripped = sentence.strip()
    if not stripped:
        return 0.0
    if technical_density_is_high(stripped):
        return 0.74
    role = _sentence_role(stripped, index=index, total=total)
    if _is_appendix_like_support(stripped, role=role):
        return 0.28

    score = 0.62
    if _CONCRETE_SUBJECT_RE.match(stripped):
        score += 0.16
    if _STRONG_VERB_RE.search(stripped):
        score += 0.14
    if _STANCE_RE.search(stripped):
        score += 0.18
    if role == "mechanism" and _CONCRETE_SUBJECT_RE.match(stripped) and _STRONG_VERB_RE.search(stripped):
        score += 0.10
    if role in {"core", "conclusion"} and _STANCE_RE.search(stripped):
        score += 0.08
    if _has_weak_abstract_main_predicate(stripped):
        score -= 0.18
    elif _WEAK_MODAL_RE.search(stripped):
        score -= 0.08
    return _clamp(score)


def _is_appendix_like_support(sentence: str, *, role: str) -> bool:
    stripped = sentence.strip()
    if not stripped:
        return False
    if role == "mechanism" and _CONCRETE_SUBJECT_RE.match(stripped) and _STRONG_VERB_RE.search(stripped):
        return False
    if _APPENDIX_LIKE_LEAD_RE.match(stripped):
        return True
    if re.match(r"^(?:本研究|本文|该方法|该机制|这种方式|该设置)用于", stripped):
        return True
    if re.match(r"^通过[^。；]{2,32}(?:可以|能够)", stripped):
        return True
    return _has_weak_abstract_main_predicate(stripped)


def _has_weak_abstract_main_predicate(sentence: str) -> bool:
    stripped = sentence.strip()
    return bool(
        re.match(
            r"^(?:本研究|本文|该方法|该机制|这种方式|该设置|这一路径|这一设计|这一机制|该系统)"
            r"(?:[^，。；]{0,10})?(?:可以|能够|用于|有助于)",
            stripped,
        )
    )


def _average(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(float(item) for item in items) / len(items)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))
