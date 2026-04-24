"""Signals for the optional mild Chinese L2 academic style profile."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any


FUNCTION_WORD_RE = re.compile(r"(?:的|了|来|进行|会|能够|还有|以此|通过[^。；]{0,18}来|是用来|用来)")
L2_TEXTURE_RE = re.compile(r"(?:是用来|进行[^。；]{0,12}工作|能够|还有|这个(?:模型|系统|分支|模块|策略)|这一类方法|通过[^。；]{0,18}来)")
COLLOQUIAL_RE = re.compile(r"(?:这块|这边|大家|超|超级|真的|其实|大白话|搞定|拉满|靠谱|玩)")
UNGRAMMATICAL_RE = re.compile(
    r"(?:本研究研究|研究研究|是了|可以用于了|被是用来|能够(?:复杂度|并非|更新|必须)|进行验证算法|辅助进行分析与诊断|，，|。。|的的|了了|进行进行)"
)
NATIVE_CONCISION_RE = re.compile(r"(?:兼顾|适用于|用于|构成|负责|提取特征|实现融合|形成闭环)")


@dataclass(frozen=True)
class L2StyleSignals:
    paragraphs_checked: int
    l2_texture_score: float
    function_word_density: float
    explanatory_rephrase_score: float
    native_like_concision_risk: float
    colloquial_risk: float
    ungrammatical_risk: float
    l2_texture_present: bool
    not_too_native_like: bool
    not_colloquial: bool
    not_ungrammatical: bool
    low_l2_texture_paragraph_ids: list[int]
    native_like_paragraph_ids: list[int]
    colloquial_paragraph_ids: list[int]
    ungrammatical_paragraph_ids: list[int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_l2_style_sentences(sentences: list[str], *, paragraph_index: int = 0) -> L2StyleSignals:
    """Measure mild L2 texture without rewarding colloquial or broken Chinese."""

    visible = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not visible:
        return L2StyleSignals(
            paragraphs_checked=0,
            l2_texture_score=0.0,
            function_word_density=0.0,
            explanatory_rephrase_score=0.0,
            native_like_concision_risk=0.0,
            colloquial_risk=0.0,
            ungrammatical_risk=0.0,
            l2_texture_present=True,
            not_too_native_like=True,
            not_colloquial=True,
            not_ungrammatical=True,
            low_l2_texture_paragraph_ids=[],
            native_like_paragraph_ids=[],
            colloquial_paragraph_ids=[],
            ungrammatical_paragraph_ids=[],
        )

    text = "".join(visible)
    char_count = max(1, len(re.sub(r"\s+", "", text)))
    sentence_count = max(1, len(visible))
    function_density = round(min(1.0, len(FUNCTION_WORD_RE.findall(text)) / max(1, char_count / 18)), 4)
    texture_hits = len(L2_TEXTURE_RE.findall(text))
    explanatory_hits = sum(1 for sentence in visible if re.search(r"(?:主要的作用是|这个.+?会|通过.+?来|是用来|的工作)", sentence))
    native_hits = len(NATIVE_CONCISION_RE.findall(text))
    colloquial_hits = len(COLLOQUIAL_RE.findall(text))
    ungrammatical_hits = len(UNGRAMMATICAL_RE.findall(text))
    l2_score = round(min(1.0, function_density * 0.45 + min(1.0, texture_hits / sentence_count) * 0.35 + min(1.0, explanatory_hits / sentence_count) * 0.20), 4)
    native_risk = round(min(1.0, native_hits / max(1, sentence_count * 2)), 4)
    colloquial_risk = round(min(1.0, colloquial_hits / sentence_count), 4)
    ungrammatical_risk = round(min(1.0, ungrammatical_hits / sentence_count), 4)
    low_l2 = l2_score < 0.16
    native_like = native_risk > 0.30 and l2_score < 0.24

    return L2StyleSignals(
        paragraphs_checked=1,
        l2_texture_score=l2_score,
        function_word_density=function_density,
        explanatory_rephrase_score=round(min(1.0, explanatory_hits / sentence_count), 4),
        native_like_concision_risk=native_risk,
        colloquial_risk=colloquial_risk,
        ungrammatical_risk=ungrammatical_risk,
        l2_texture_present=not low_l2,
        not_too_native_like=not native_like,
        not_colloquial=colloquial_risk == 0.0,
        not_ungrammatical=ungrammatical_risk == 0.0,
        low_l2_texture_paragraph_ids=[paragraph_index] if low_l2 else [],
        native_like_paragraph_ids=[paragraph_index] if native_like else [],
        colloquial_paragraph_ids=[paragraph_index] if colloquial_risk else [],
        ungrammatical_paragraph_ids=[paragraph_index] if ungrammatical_risk else [],
    )


def aggregate_l2_style_profile(rewrite_stats: list[Any], *, enabled: bool) -> dict[str, object]:
    """Aggregate optional mild-L2 style checks across rewritten body paragraphs."""

    if not enabled:
        return {
            "enabled": False,
            "paragraphs_checked": 0,
            "l2_texture_score": 0.0,
            "function_word_density": 0.0,
            "explanatory_rephrase_score": 0.0,
            "native_like_concision_risk": 0.0,
            "colloquial_risk": 0.0,
            "ungrammatical_risk": 0.0,
            "l2_texture_present": True,
            "not_too_native_like": True,
            "not_colloquial": True,
            "not_ungrammatical": True,
            "fact_scope_preserved": True,
            "technical_terms_preserved": True,
            "low_l2_texture_paragraph_ids": [],
            "native_like_paragraph_ids": [],
            "colloquial_paragraph_ids": [],
            "ungrammatical_paragraph_ids": [],
        }

    checked = [
        stats
        for stats in rewrite_stats
        if getattr(stats, "changed", False)
        and getattr(stats, "rewrite_depth", "") in {"developmental_rewrite", "light_edit"}
        and getattr(stats, "rewritten_sentences", None)
    ]
    if not checked:
        return {**aggregate_l2_style_profile([], enabled=False), "enabled": True, "l2_texture_present": False}

    signals = [
        analyze_l2_style_sentences(
            list(getattr(stats, "rewritten_sentences", [])),
            paragraph_index=int(getattr(stats, "paragraph_index", 0) or 0),
        )
        for stats in checked
    ]
    paragraphs_checked = len(signals)

    def avg(attr: str) -> float:
        return round(sum(float(getattr(signal, attr)) for signal in signals) / max(1, paragraphs_checked), 4)

    def ids(attr: str) -> list[int]:
        merged: list[int] = []
        for signal in signals:
            merged.extend(getattr(signal, attr))
        return sorted(set(merged))

    low_l2_ids = ids("low_l2_texture_paragraph_ids")
    native_ids = ids("native_like_paragraph_ids")
    colloquial_ids = ids("colloquial_paragraph_ids")
    ungrammatical_ids = ids("ungrammatical_paragraph_ids")
    l2_score = avg("l2_texture_score")
    native_risk = avg("native_like_concision_risk")
    colloquial_risk = avg("colloquial_risk")
    ungrammatical_risk = avg("ungrammatical_risk")

    return {
        "enabled": True,
        "paragraphs_checked": paragraphs_checked,
        "l2_texture_score": l2_score,
        "function_word_density": avg("function_word_density"),
        "explanatory_rephrase_score": avg("explanatory_rephrase_score"),
        "native_like_concision_risk": native_risk,
        "colloquial_risk": colloquial_risk,
        "ungrammatical_risk": ungrammatical_risk,
        "l2_texture_present": l2_score >= 0.18 and len(low_l2_ids) <= max(8, paragraphs_checked // 3),
        "not_too_native_like": native_risk <= 0.26 and len(native_ids) <= max(8, paragraphs_checked // 3),
        "not_colloquial": colloquial_risk == 0.0,
        "not_ungrammatical": ungrammatical_risk == 0.0,
        "fact_scope_preserved": True,
        "technical_terms_preserved": True,
        "low_l2_texture_paragraph_ids": low_l2_ids,
        "native_like_paragraph_ids": native_ids,
        "colloquial_paragraph_ids": colloquial_ids,
        "ungrammatical_paragraph_ids": ungrammatical_ids,
    }
