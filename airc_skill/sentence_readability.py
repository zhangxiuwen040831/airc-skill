from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_SENTENCE_RE = re.compile(r"[^。！？?!]+[。！？?!]?")
_SPACE_RE = re.compile(r"\s+")
_PREDICATE_RE = re.compile(
    r"(是|为|有|具有|能够|可以|需要|提出|构建|采用|表明|说明|体现|提升|降低|实现|形成|保持|"
    r"依赖|包括|包含|用于|面向|围绕|描述|展示|验证|支持|导致|反映|解决|完成|提供|增强|"
    r"引入|关注|分析|设计|部署|训练|评价|计算|识别|检测|分类|生成|保留|冻结|保护|约束|避免|使|"
    r"遇到|经历|达到|减少|增加|优于|适合|支撑|承担|作为|展现|呈现|可能|学到|偏向)"
)
_SUBJECT_RE = re.compile(
    r"^(?:本研究|本文|该研究|研究内容|该系统|系统|该模型|模型|该方法|该策略|该设计|这种设计|"
    r"这一过程|这一结果|这一问题|该问题|实验结果|结果|方法|模型结构|检测流程|系统实现|用户|图像|"
    r"门控机制|融合层|分支贡献|某一分支)"
)
_DANGLING_START_RE = re.compile(
    r"^(?P<marker>并进一步|并且进一步|并(?=提出|形成|实现|说明|表明|进一步|完成|用于)|而是|由此|同时|此外|"
    r"在这种情况下|相应地|其中|需要说明的是|从而|进而|随后|之后|以及)[，,\s]*"
)
_COMPARISON_FRAGMENT_RE = re.compile(r"^是比.+更.+")
_PREPOSITION_FRAGMENT_RE = re.compile(r"^(?:在|通过|基于|面向|围绕).+(?:之后|之中|过程中)[。！？?!]?$")
_COLON_FRAGMENT_RE = re.compile(r"：\s*(?:在|通过|基于|面向|围绕)[^，,；;。！？?!]{2,30}[。！？?!]?$")
_RESULT_FRAGMENT_RE = re.compile(r"^(?:结合|根据|基于).{2,35}(?:结果|数据|分析|实验|评测|诊断)[。！？?!]?$")
_VERB_OPEN_FRAGMENT_RE = re.compile(r"^(?:强调|说明|体现|支持|展示|反映|用于|包括|包含|呈现|提供|形成|实现)")
_CAUSATIVE_FRAGMENT_RE = re.compile(r"^(?:使|让)[^。！？?!]{8,}")
_METHOD_TAXONOMY_RE = re.compile(r"^基于.{2,40}的方法(?:\[\[AIRC:CORE_CITATION:\d+\]\])?[。！？?!]?$")
_LIST_OR_FORMULA_LEAD_RE = re.compile(r"(?:如下|如下所示|核心表达式如下|定义如下|包含以下[一二三四五六七八九十\d]+个部分)[:：]?[。！？?!]?$")
_ESPECIALLY_TERM_FRAGMENT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*(?:尤其是|特别是).{2,50}[。！？?!]?$")
_TECH_MARKER_RE = re.compile(
    r"\[\[AIRC:CORE_[A-Z_]+:\d+\]\]|[A-Za-z][A-Za-z0-9_.\-/]{2,}|"
    r"checkpoint[s]?/[^，。；\s]+|[\w.-]+\.(?:pth|pt|ckpt)|\[\d+\]"
)


@dataclass(frozen=True)
class SentenceReadabilitySignals:
    sentence_completeness_score: float
    dangling_sentence_risk: float
    paragraph_readability_score: float
    incomplete_support_sentence_risk: float
    fragment_like_conclusion_risk: float
    sentence_count: int
    dangling_sentence_indexes: list[int] = field(default_factory=list)
    incomplete_support_indexes: list[int] = field(default_factory=list)
    fragment_like_conclusion_indexes: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "sentence_completeness_score": self.sentence_completeness_score,
            "dangling_sentence_risk": self.dangling_sentence_risk,
            "paragraph_readability_score": self.paragraph_readability_score,
            "incomplete_support_sentence_risk": self.incomplete_support_sentence_risk,
            "fragment_like_conclusion_risk": self.fragment_like_conclusion_risk,
            "sentence_count": self.sentence_count,
            "dangling_sentence_indexes": list(self.dangling_sentence_indexes),
            "incomplete_support_indexes": list(self.incomplete_support_indexes),
            "fragment_like_conclusion_indexes": list(self.fragment_like_conclusion_indexes),
        }


def split_sentences_for_readability(text: str) -> list[str]:
    """Split prose into sentence-like units for readability checks."""

    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]


def analyze_paragraph_readability_text(
    text: str,
    *,
    high_sensitivity: bool = False,
) -> SentenceReadabilitySignals:
    """Analyze sentence completeness and paragraph readability from raw text."""

    return analyze_paragraph_readability_sentences(
        split_sentences_for_readability(text),
        high_sensitivity=high_sensitivity,
    )


def analyze_paragraph_readability_sentences(
    sentences: list[str],
    *,
    roles: list[str] | None = None,
    high_sensitivity: bool = False,
) -> SentenceReadabilitySignals:
    """Return aggregate readability signals for a paragraph's sentence list."""

    visible = [sentence.strip() for sentence in sentences if sentence.strip()]
    count = len(visible)
    if count == 0:
        return SentenceReadabilitySignals(1.0, 0.0, 1.0, 0.0, 0.0, 0)

    sentence_scores: list[float] = []
    dangling_indexes: list[int] = []
    incomplete_support_indexes: list[int] = []
    fragment_conclusion_indexes: list[int] = []

    for index, sentence in enumerate(visible):
        role = roles[index] if roles and index < len(roles) else _default_role(index, count)
        score = sentence_completeness_score(sentence, role=role)
        sentence_scores.append(score)
        if dangling_sentence_risk(sentence):
            dangling_indexes.append(index)
        if incomplete_support_sentence_risk(sentence, role=role):
            incomplete_support_indexes.append(index)
        if fragment_like_conclusion_sentence(sentence, is_final=index == count - 1, role=role):
            fragment_conclusion_indexes.append(index)

    completeness = _average(sentence_scores)
    dangling_ratio = len(dangling_indexes) / count
    support_ratio = len(incomplete_support_indexes) / count
    conclusion_ratio = len(fragment_conclusion_indexes) / count
    opening_bonus = 0.05 if count and sentence_scores[0] >= 0.72 and not dangling_sentence_risk(visible[0]) else -0.08
    ending_bonus = 0.04 if sentence_scores[-1] >= 0.72 and not fragment_like_conclusion_sentence(visible[-1], is_final=True) else -0.08
    sensitivity_penalty = 0.08 if high_sensitivity and (dangling_indexes or incomplete_support_indexes) else 0.0
    paragraph_readability = _clamp(
        completeness
        - dangling_ratio * 0.24
        - support_ratio * 0.20
        - conclusion_ratio * 0.16
        + opening_bonus
        + ending_bonus
        - sensitivity_penalty
    )

    return SentenceReadabilitySignals(
        sentence_completeness_score=round(completeness, 4),
        dangling_sentence_risk=round(dangling_ratio, 4),
        paragraph_readability_score=round(paragraph_readability, 4),
        incomplete_support_sentence_risk=round(support_ratio, 4),
        fragment_like_conclusion_risk=round(conclusion_ratio, 4),
        sentence_count=count,
        dangling_sentence_indexes=dangling_indexes,
        incomplete_support_indexes=incomplete_support_indexes,
        fragment_like_conclusion_indexes=fragment_conclusion_indexes,
    )


def sentence_completeness_score(sentence: str, *, role: str = "") -> float:
    """Score whether a sentence can stand as complete academic prose."""

    stripped = sentence.strip()
    if not stripped:
        return 0.0
    implication = re.match(r"^(?:因此|由此|由此可见|基于此|在此基础上)[，,\s]*(?P<tail>.+)$", stripped)
    if implication and _has_complete_clause(implication.group("tail")):
        return 0.82
    if _LIST_OR_FORMULA_LEAD_RE.search(stripped):
        return 0.88
    if _METHOD_TAXONOMY_RE.match(stripped):
        return 0.84
    if technical_density_is_high(stripped):
        return 0.84
    if dangling_sentence_risk(stripped):
        return 0.36
    if _COMPARISON_FRAGMENT_RE.match(stripped):
        return 0.34
    if _PREPOSITION_FRAGMENT_RE.match(stripped):
        return 0.42
    if _COLON_FRAGMENT_RE.search(stripped):
        return 0.46
    if _RESULT_FRAGMENT_RE.match(stripped):
        return 0.44
    if _VERB_OPEN_FRAGMENT_RE.match(stripped):
        return 0.50
    if _CAUSATIVE_FRAGMENT_RE.match(stripped) and not _has_subject_like(stripped):
        return 0.42
    if _ESPECIALLY_TERM_FRAGMENT_RE.match(stripped):
        return 0.58

    has_predicate = bool(_PREDICATE_RE.search(stripped))
    has_subject = _has_subject_like(stripped)
    length = _visible_length(stripped)
    if not has_predicate and length < 18:
        return 0.48
    if not has_predicate:
        return 0.58
    if not has_subject and role in {"support_sentence", "conclusion_sentence"} and length < 22:
        return 0.62
    if stripped.startswith(("而", "并", "从而", "进而")) and not has_subject:
        return 0.52
    return 0.90 if has_subject else 0.78


def dangling_sentence_risk(sentence: str) -> bool:
    """Detect sentences that depend too strongly on a missing previous clause."""

    stripped = sentence.strip()
    if not stripped:
        return False
    if _LIST_OR_FORMULA_LEAD_RE.search(stripped) or _METHOD_TAXONOMY_RE.match(stripped):
        return False
    marker_match = _DANGLING_START_RE.match(stripped)
    if marker_match:
        marker = marker_match.group("marker")
        tail = stripped[marker_match.end():].strip()
        if marker in {"同时", "此外", "由此", "相应地", "其中", "需要说明的是"} and _has_complete_clause(tail):
            return False
        return True
    if technical_density_is_high(stripped):
        return False
    if _COMPARISON_FRAGMENT_RE.match(stripped):
        return True
    if _PREPOSITION_FRAGMENT_RE.match(stripped):
        return True
    if _COLON_FRAGMENT_RE.search(stripped):
        return True
    if _RESULT_FRAGMENT_RE.match(stripped):
        return True
    if _VERB_OPEN_FRAGMENT_RE.match(stripped):
        return True
    if _CAUSATIVE_FRAGMENT_RE.match(stripped) and not _has_subject_like(stripped):
        return True
    return False


def incomplete_support_sentence_risk(sentence: str, *, role: str = "support_sentence") -> bool:
    """Detect support sentences that read like fragments instead of claims."""

    if role == "topic_sentence":
        return False
    stripped = sentence.strip()
    if not stripped or technical_density_is_high(stripped):
        return False
    if _LIST_OR_FORMULA_LEAD_RE.search(stripped) or _METHOD_TAXONOMY_RE.match(stripped):
        return False
    if dangling_sentence_risk(stripped):
        return True
    if sentence_completeness_score(stripped, role=role) < 0.66:
        return True
    if not _has_subject_like(stripped) and _visible_length(stripped) < 24:
        return True
    if stripped.startswith(("而是", "并进一步", "从而", "进而")):
        return True
    return False


def fragment_like_conclusion_sentence(
    sentence: str,
    *,
    is_final: bool = False,
    role: str = "",
) -> bool:
    """Detect final sentences that look like leftover tails or dangling conclusions."""

    stripped = sentence.strip()
    if not stripped or technical_density_is_high(stripped):
        return False
    if _LIST_OR_FORMULA_LEAD_RE.search(stripped):
        return False
    if _COMPARISON_FRAGMENT_RE.match(stripped):
        return True
    if role == "conclusion_sentence" or is_final:
        implication = re.match(r"^(?:因此|由此|从而|进而)[，,\s]*(?P<tail>.+)$", stripped)
        if implication and not _has_complete_clause(implication.group("tail")):
            return True
        if not implication and stripped.startswith(("是比", "而是")):
            return True
        if sentence_completeness_score(stripped, role="conclusion_sentence") < 0.58:
            return True
    return False


def aggregate_sentence_readability(rewrite_stats: list[Any]) -> dict[str, object]:
    """Aggregate sentence-level readability signals over changed body blocks."""

    checked = [
        stats for stats in rewrite_stats
        if getattr(stats, "changed", False)
        and getattr(stats, "rewrite_depth", "") in {"developmental_rewrite", "light_edit"}
        and getattr(stats, "rewritten_sentences", None)
    ]
    if not checked:
        return {
            "paragraphs_checked": 0,
            "sentence_completeness_score": 1.0,
            "paragraph_readability_score": 1.0,
            "dangling_sentence_risk": 0.0,
            "incomplete_support_sentence_risk": 0.0,
            "fragment_like_conclusion_risk": 0.0,
            "sentence_completeness_preserved": True,
            "paragraph_readability_preserved": True,
            "no_dangling_support_sentences": True,
            "no_fragment_like_conclusion_sentences": True,
            "dangling_sentence_paragraph_ids": [],
            "incomplete_support_paragraph_ids": [],
            "fragment_like_conclusion_paragraph_ids": [],
            "high_sensitivity_readability_risk_ids": [],
        }

    revised = [
        analyze_paragraph_readability_sentences(
            list(stats.rewritten_sentences),
            high_sensitivity=bool(getattr(stats, "high_sensitivity_prose", False)),
        )
        for stats in checked
    ]
    original = [
        analyze_paragraph_readability_sentences(list(stats.original_sentences))
        for stats in checked
    ]
    average_completeness = _average(signal.sentence_completeness_score for signal in revised)
    average_readability = _average(signal.paragraph_readability_score for signal in revised)
    average_dangling = _average(signal.dangling_sentence_risk for signal in revised)
    average_support = _average(signal.incomplete_support_sentence_risk for signal in revised)
    average_conclusion = _average(signal.fragment_like_conclusion_risk for signal in revised)
    original_completeness = _average(signal.sentence_completeness_score for signal in original)
    original_readability = _average(signal.paragraph_readability_score for signal in original)

    dangling_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, revised, strict=False)
        if signal.dangling_sentence_indexes
    ]
    support_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, revised, strict=False)
        if signal.incomplete_support_indexes
    ]
    conclusion_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, revised, strict=False)
        if signal.fragment_like_conclusion_indexes
    ]
    high_sensitivity_ids = [
        getattr(stats, "paragraph_index", 0)
        for stats, signal in zip(checked, revised, strict=False)
        if bool(getattr(stats, "high_sensitivity_prose", False))
        and (
            signal.sentence_completeness_score < 0.74
            or signal.paragraph_readability_score < 0.72
            or signal.dangling_sentence_indexes
            or signal.incomplete_support_indexes
            or signal.fragment_like_conclusion_indexes
        )
    ]
    checked_count = len(checked)
    allowed_minor = max(1, checked_count // 12)
    return {
        "paragraphs_checked": checked_count,
        "sentence_completeness_score": round(average_completeness, 4),
        "paragraph_readability_score": round(average_readability, 4),
        "dangling_sentence_risk": round(average_dangling, 4),
        "incomplete_support_sentence_risk": round(average_support, 4),
        "fragment_like_conclusion_risk": round(average_conclusion, 4),
        "original_sentence_completeness_score": round(original_completeness, 4),
        "original_paragraph_readability_score": round(original_readability, 4),
        "sentence_completeness_preserved": (
            average_completeness >= max(0.72, original_completeness - 0.06)
            and len(support_ids) <= allowed_minor
            and not high_sensitivity_ids
        ),
        "paragraph_readability_preserved": (
            average_readability >= max(0.70, original_readability - 0.06)
            and len(dangling_ids) <= allowed_minor
            and not high_sensitivity_ids
        ),
        "no_dangling_support_sentences": len(dangling_ids) <= allowed_minor and not high_sensitivity_ids,
        "no_fragment_like_conclusion_sentences": len(conclusion_ids) <= max(1, checked_count // 16),
        "dangling_sentence_paragraph_ids": dangling_ids,
        "incomplete_support_paragraph_ids": support_ids,
        "fragment_like_conclusion_paragraph_ids": conclusion_ids,
        "high_sensitivity_readability_risk_ids": high_sensitivity_ids,
    }


def technical_density_is_high(sentence: str) -> bool:
    """Return true when protected/technical markers make natural repair unsafe."""

    return len(_TECH_MARKER_RE.findall(sentence)) >= 3


def _default_role(index: int, count: int) -> str:
    if index == 0:
        return "topic_sentence"
    if index == count - 1:
        return "conclusion_sentence"
    return "support_sentence"


def _has_early_nominal_subject(sentence: str) -> bool:
    head = re.split(r"[，,：:；;]", sentence, maxsplit=1)[0]
    return 2 <= _visible_length(head) <= 18 and bool(re.search(r"(研究|系统|模型|方法|结果|问题|任务|设计|策略|流程|结构|数据)", head))


def _has_subject_like(sentence: str) -> bool:
    sentence = re.sub(r"^(?:因此|由此|由此可见|基于此|在此基础上)[，,\s]*", "", sentence.strip())
    sentence = re.sub(r"^第[一二三四五六七八九十\d]+[，,\s]*", "", sentence.strip())
    return bool(_SUBJECT_RE.match(sentence)) or bool(_METHOD_TAXONOMY_RE.match(sentence)) or _has_early_nominal_subject(sentence)


def _has_complete_clause(sentence: str) -> bool:
    stripped = sentence.strip()
    if not stripped:
        return False
    stripped = re.sub(r"^第[一二三四五六七八九十\d]+[，,\s]*", "", stripped)
    return _has_subject_like(stripped) and bool(_PREDICATE_RE.search(stripped))


def _visible_length(sentence: str) -> int:
    return len(_SPACE_RE.sub("", sentence))


def _average(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(float(item) for item in items) / len(items)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
