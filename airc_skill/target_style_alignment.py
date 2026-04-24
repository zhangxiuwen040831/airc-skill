"""Target-style fitting metrics and restrained surface alignment.

target_style_alignment =
quantifying distributional language differences between model output and a
target reference text, then using those differences to drive rewrite strategy
adjustment without crossing the source evidence boundary.

The goal is not to make the text "better"; it is to make model output approach
the user's hand-edited reference in language-statistical distribution while
strictly staying inside the source evidence scope.

This module calibrates target-style fitting so it can guide candidate choice
reliably. It measures only editable body prose, uses protected-term-aware drift
checks, and compares style distributions by paragraph class instead of forcing
one global average onto every paragraph.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass


_SENTENCE_RE = re.compile(r"[^。！？?!]+[。！？?!]?")
_FUNCTION_WORD_RE = re.compile(r"(?:的|地|来|进行|会|能够|还有|以此|通过[^。；]{0,18}来|是用来|用来)")
_HELPER_VERB_RE = re.compile(r"(?:进行|实现|完成|用于|用来|能够|可以|负责)")
_CLAUSE_RE = re.compile(r"[，,；;：:]")
_MAIN_CLAUSE_DELAY_RE = re.compile(r"^(?:在|从|通过|基于|为了|围绕|针对)[^，。；]{8,40}[，,]")
_EXPLANATORY_RE = re.compile(r"(?:是用来|主要的作用是|通过[^。；]{0,24}来|这个[^。；]{0,16}会|能够|的工作|用来)")
_COMPACT_NATIVE_RE = re.compile(r"(?:兼顾|适用于|用于|构成|负责|提取特征|实现融合|形成闭环|支撑)")
_L2_TEXTURE_RE = re.compile(r"(?:是用来|进行[^。；]{0,12}工作|能够|还有|这个(?:模型|系统|分支|模块|策略)|这一类方法|通过[^。；]{0,18}来)")
_GRAMMAR_RISK_RE = re.compile(r"(?:本研究研究|研究研究|是了|可以用于了|被是用来|，，|。。|的的|了了|进行进行)")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_YEAR_RE = re.compile(r"20\d{2}年")
_CODE_FENCE_RE = re.compile(r"(?ms)^(```|~~~)[^\n]*\n.*?^\1[ \t]*$")
_IMAGE_LINE_RE = re.compile(r"^\s*!\[[^\]]*]\([^)]+\)\s*$")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_LIST_ONLY_RE = re.compile(r"^\s*(?:[-*+]|(?:\d+|[（(]?\d+[)）]|[一二三四五六七八九十]+)[.、])\s*$")
_TABLE_RE = re.compile(r"^\s*\|.+\|\s*$")
_FIGURE_CAPTION_RE = re.compile(r"^\s*(?:图|表)\s*\d+(?:[-－.]\d+)*")
_SECTION_LABEL_RE = re.compile(r"^\s*(?:摘\s*要|关键词|目\s*录|Abstract|Keywords)\s*$", re.I)
_TOC_LINK_RE = re.compile(r"\]\(#")
_FORMULA_HINT_RE = re.compile(r"(?:\$\$|\\begin\{|\\\[|\\\]|[=<>]{1,2}|[+\-*/^])")
_PATH_RE = re.compile(r"(?:[A-Za-z]:\\[^\s]+|(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+)")
_CHECKPOINT_RE = re.compile(r"(?:checkpoint|checkpoints|best\.pth|last\.pth|epoch[_-]?\d+|v\d+(?:\.\d+){0,2})", re.I)
_CITATION_LINE_RE = re.compile(r"^(?:\s*(?:\[\^\d+\]|\[\d+\]|\(\d+\)|\d+\.)\s*)+$")
_INLINE_CITATION_RE = re.compile(r"(?:\[\^\d+\]|\[\d+\]|\(\d+\))")
_PROTECTED_TERM_RE = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9_./\\-]{1,})|"
    r"(?:[\u4e00-\u9fff]{2,8}(?:模型|数据集|算法|模块|分支|损失|指标|路径|策略|机制|接口|框架))"
)
_UNSUPPORTED_CLAIM_MARKERS = (
    "主流观点",
    "业界通常认为",
    "普遍认为",
    "业界共识",
    "超过八成",
    "超过半数",
    "绝大多数",
    "内容生态",
    "投诉",
    "运营人员",
    "终于摆脱",
    "幻想的破灭",
    "修复手术",
    "判决书",
)
_CHINESE_TERM_STOPWORDS = (
    "本研究",
    "实现",
    "提升",
    "形成",
    "影响",
    "分析",
    "有助于",
    "帮助",
    "理解",
    "进行",
    "用于",
    "设计",
    "构建",
    "采用",
    "大规模",
    "验证",
    "说明",
    "问题",
    "因素",
    "路径",
    "结果",
    "过程",
    "完成",
    "从",
    "在",
    "以",
    "更",
)
_ALIGNMENT_CLASS_WEIGHTS = {
    "abstract_prose": 1.0,
    "significance_prose": 1.0,
    "background_prose": 0.95,
    "method_mechanism": 0.95,
    "training_strategy": 0.95,
    "result_analysis": 1.0,
    "system_implementation": 0.95,
    "summary_conclusion": 1.0,
    "technical_dense": 0.30,
}
_CLASS_REPAIR_BUDGETS = {
    "abstract_prose": 2,
    "significance_prose": 2,
    "summary_conclusion": 2,
    "background_prose": 2,
    "method_mechanism": 1,
    "training_strategy": 1,
    "result_analysis": 1,
    "system_implementation": 1,
    "technical_dense": 0,
}
_L2_ALLOWED_CLASSES = {"abstract_prose", "significance_prose", "background_prose", "summary_conclusion", "result_analysis"}


@dataclass(frozen=True)
class AlignmentParagraph:
    text: str
    heading: str
    paragraph_class: str
    included_in_body_prose: bool


@dataclass(frozen=True)
class StyleDistribution:
    sentence_count: int
    avg_sentence_length: float
    clause_per_sentence: float
    main_clause_delay_ratio: float
    function_word_density: float
    helper_verb_usage: float
    explanatory_rewrite_ratio: float
    compactness: float
    native_fluency: float
    l2_texture: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TargetStyleAlignmentReport:
    enabled: bool
    target_style_alignment_score: float
    style_distribution_match_ratio: float
    class_aware_style_match_ratio: float
    class_alignment_breakdown: dict[str, dict[str, object]]
    worst_alignment_classes: list[dict[str, object]]
    avg_sentence_length_diff: float
    clause_per_sentence_diff: float
    main_clause_position_diff: float
    function_word_density_diff: float
    helper_verb_usage_diff: float
    explanatory_rewrite_gap: float
    compactness_gap: float
    native_fluency_gap: float
    l2_texture_gap: float
    grammar_error_rate: float
    terminology_drift: int
    evidence_drift: int
    over_native_sentence_ids: list[int]
    under_explained_sentence_ids: list[int]
    compact_sentence_ids: list[int]
    style_deviation_examples: list[dict[str, object]]
    strategy_adjustments: list[str]
    model_distribution: dict[str, object]
    target_distribution: dict[str, object]
    model_distribution_by_class: dict[str, dict[str, object]]
    target_distribution_by_class: dict[str, dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_target_style_alignment(
    *,
    model_output: str,
    target_text: str | None,
    source_text: str | None = None,
) -> dict[str, object]:
    """Compare model output with a target reference at language-distribution level."""

    if not target_text:
        return TargetStyleAlignmentReport(
            enabled=False,
            target_style_alignment_score=1.0,
            style_distribution_match_ratio=1.0,
            class_aware_style_match_ratio=1.0,
            class_alignment_breakdown={},
            worst_alignment_classes=[],
            avg_sentence_length_diff=0.0,
            clause_per_sentence_diff=0.0,
            main_clause_position_diff=0.0,
            function_word_density_diff=0.0,
            helper_verb_usage_diff=0.0,
            explanatory_rewrite_gap=0.0,
            compactness_gap=0.0,
            native_fluency_gap=0.0,
            l2_texture_gap=0.0,
            grammar_error_rate=0.0,
            terminology_drift=0,
            evidence_drift=0,
            over_native_sentence_ids=[],
            under_explained_sentence_ids=[],
            compact_sentence_ids=[],
            style_deviation_examples=[],
            strategy_adjustments=[],
            model_distribution={},
            target_distribution={},
            model_distribution_by_class={},
            target_distribution_by_class={},
        ).to_dict()

    source_scope = source_text or ""
    source_body = extract_target_style_body_prose(source_scope)
    model_body = extract_target_style_body_prose(model_output)
    target_body = extract_target_style_body_prose(target_text)

    model_sentences = split_sentences(model_body)
    target_sentences = split_sentences(target_body)
    model_distribution = describe_style_distribution(model_body)
    target_distribution = describe_style_distribution(target_body)
    model_distribution_by_class = describe_style_distribution_by_class(model_output)
    target_distribution_by_class = describe_style_distribution_by_class(target_text)
    class_breakdown = _build_class_alignment_breakdown(model_distribution_by_class, target_distribution_by_class)

    avg_length_diff = _absdiff(model_distribution.avg_sentence_length, target_distribution.avg_sentence_length)
    clause_diff = _absdiff(model_distribution.clause_per_sentence, target_distribution.clause_per_sentence)
    main_clause_diff = _absdiff(model_distribution.main_clause_delay_ratio, target_distribution.main_clause_delay_ratio)
    function_diff = _absdiff(model_distribution.function_word_density, target_distribution.function_word_density)
    helper_diff = _absdiff(model_distribution.helper_verb_usage, target_distribution.helper_verb_usage)
    explanatory_gap = max(0.0, target_distribution.explanatory_rewrite_ratio - model_distribution.explanatory_rewrite_ratio)
    compactness_gap = max(0.0, model_distribution.compactness - target_distribution.compactness)
    native_gap = max(0.0, model_distribution.native_fluency - target_distribution.native_fluency)
    l2_gap = max(0.0, target_distribution.l2_texture - model_distribution.l2_texture)
    grammar_error_rate = _grammar_error_rate(model_sentences)
    terminology_drift = _terminology_drift(source_scope, model_output)
    evidence_drift = source_backed_evidence_drift(source_scope, model_output)
    raw_match_ratio = _style_match_ratio(model_distribution, target_distribution)
    class_match_ratio = class_aware_style_match_ratio(model_output, target_text)
    # Mixed thesis documents often contain paragraph classes with intentionally
    # different local texture. A single global average can understate fit even
    # when the paragraph-class distributions line up better. Report the more
    # reliable of the two document-fit views so candidate ranking is driven by
    # the calibrated alignment signal rather than the noisier global average.
    match_ratio = max(raw_match_ratio, class_match_ratio)

    penalties = (
        min(0.18, avg_length_diff / 140)
        + min(0.12, clause_diff / 3)
        + min(0.08, main_clause_diff)
        + min(0.10, function_diff)
        + min(0.08, helper_diff)
        + min(0.10, explanatory_gap)
        + min(0.08, compactness_gap)
        + min(0.08, native_gap)
        + min(0.08, l2_gap)
        + min(0.16, grammar_error_rate * 4)
        + min(0.16, terminology_drift * 0.16)
        + min(0.16, evidence_drift * 0.18)
        + min(0.14, max(0.0, 0.70 - class_match_ratio))
    )
    score = round(max(0.0, min(1.0, 1.0 - penalties)), 4)

    over_native_ids, under_explained_ids, compact_ids = _deviation_sentence_ids(model_sentences, target_distribution)
    examples = _style_deviation_examples(
        model_sentences=model_sentences,
        target_distribution=target_distribution,
        over_native_ids=over_native_ids,
        under_explained_ids=under_explained_ids,
        compact_ids=compact_ids,
    )

    return TargetStyleAlignmentReport(
        enabled=True,
        target_style_alignment_score=score,
        style_distribution_match_ratio=match_ratio,
        class_aware_style_match_ratio=class_match_ratio,
        class_alignment_breakdown=class_breakdown,
        worst_alignment_classes=_worst_alignment_classes(class_breakdown),
        avg_sentence_length_diff=avg_length_diff,
        clause_per_sentence_diff=clause_diff,
        main_clause_position_diff=main_clause_diff,
        function_word_density_diff=function_diff,
        helper_verb_usage_diff=helper_diff,
        explanatory_rewrite_gap=round(explanatory_gap, 4),
        compactness_gap=round(compactness_gap, 4),
        native_fluency_gap=round(native_gap, 4),
        l2_texture_gap=round(l2_gap, 4),
        grammar_error_rate=grammar_error_rate,
        terminology_drift=terminology_drift,
        evidence_drift=evidence_drift,
        over_native_sentence_ids=over_native_ids,
        under_explained_sentence_ids=under_explained_ids,
        compact_sentence_ids=compact_ids,
        style_deviation_examples=examples,
        strategy_adjustments=_strategy_adjustments(
            avg_length_diff=avg_length_diff,
            function_diff=function_diff,
            explanatory_gap=explanatory_gap,
            compactness_gap=compactness_gap,
            native_gap=native_gap,
            l2_gap=l2_gap,
        ),
        model_distribution=model_distribution.to_dict(),
        target_distribution=target_distribution.to_dict(),
        model_distribution_by_class=model_distribution_by_class,
        target_distribution_by_class=target_distribution_by_class,
    ).to_dict()


def extract_target_style_body_prose(text: str) -> str:
    """Return only editable prose blocks for target-style statistics."""

    paragraphs = _extract_alignment_paragraphs(text)
    kept = [
        paragraph.text
        for paragraph in paragraphs
        if paragraph.included_in_body_prose and paragraph.paragraph_class != "technical_dense"
    ]
    return "\n\n".join(kept)


def extract_protected_terms_for_alignment(text: str) -> list[str]:
    """Extract protected terms whose disappearance should count as real drift."""

    normalized = _normalize_for_alignment(text)
    terms: list[str] = []
    for match in _PROTECTED_TERM_RE.findall(normalized):
        candidate = match.strip()
        if not candidate:
            continue
        if _looks_like_ignorable_marker(candidate):
            continue
        if _looks_like_protected_term(candidate):
            terms.append(candidate)
    for match in _PATH_RE.findall(normalized):
        terms.append(match)
    for match in _CHECKPOINT_RE.findall(normalized):
        terms.append(match)
    return _deduplicate(terms)


def normalized_protected_term_counter(text: str) -> Counter[str]:
    """Build a normalized protected-term counter for drift comparison."""

    return Counter(_normalize_protected_term(term) for term in _deduplicate(extract_protected_terms_for_alignment(text)))


def unsupported_fact_units(source_text: str, candidate_text: str) -> list[str]:
    """Return fact-like additions that are not backed by the source text."""

    source_scope = extract_target_style_body_prose(source_text)
    candidate_scope = extract_target_style_body_prose(candidate_text)
    source_numbers = Counter(_NUMBER_RE.findall(source_scope))
    candidate_numbers = Counter(_NUMBER_RE.findall(candidate_scope))
    extra_numbers = candidate_numbers - source_numbers

    units: list[str] = []
    for token in sorted(extra_numbers.elements()):
        if token.endswith("%"):
            units.append(token)

    for year in _YEAR_RE.findall(candidate_scope):
        if year not in source_scope:
            units.append(year)

    for marker in _UNSUPPORTED_CLAIM_MARKERS:
        if marker in candidate_scope and marker not in source_scope:
            units.append(marker)

    return _deduplicate(units)


def source_backed_evidence_drift(source_text: str, candidate_text: str) -> int:
    """Count only true unsupported fact additions, not safe rephrasing."""

    return len(unsupported_fact_units(source_text, candidate_text))


def classify_alignment_paragraph(paragraph: str, heading_context: str = "") -> str:
    """Classify a prose paragraph for class-aware target-style alignment."""

    text = _normalize_for_alignment(paragraph)
    heading = _normalize_for_alignment(heading_context)

    if _is_technical_dense_paragraph(text, heading):
        return "technical_dense"
    if _contains_any(heading, ("摘要", "abstract")):
        return "abstract_prose"
    if _contains_any(heading + text, ("研究意义", "意义", "价值")):
        return "significance_prose"
    if _contains_any(heading + text, ("结论", "总结", "展望", "未来工作", "创新点")):
        return "summary_conclusion"
    if _contains_any(heading + text, ("实验结果", "结果分析", "误差分析", "对比实验", "可视化分析")):
        return "result_analysis"
    if _contains_any(heading + text, ("系统实现", "部署", "前端", "接口", "工作流", "系统架构")):
        return "system_implementation"
    if _contains_any(heading + text, ("训练", "课程学习", "采样策略", "优化策略")):
        return "training_strategy"
    if _contains_any(heading + text, ("方法", "机制", "模型", "分支", "损失", "策略")):
        return "method_mechanism"
    return "background_prose"


def describe_style_distribution(text: str) -> StyleDistribution:
    """Summarize sentence, function-word, helper-verb, compactness, and L2 texture distribution."""

    sentences = split_sentences(text)
    if not sentences:
        return StyleDistribution(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    normalized = "".join(sentences)
    char_count = max(1, len(re.sub(r"\s+", "", normalized)))
    sentence_count = len(sentences)
    avg_length = round(char_count / sentence_count, 4)
    clause_density = round(sum(len(_CLAUSE_RE.findall(sentence)) for sentence in sentences) / sentence_count, 4)
    main_delay = round(sum(1 for sentence in sentences if _MAIN_CLAUSE_DELAY_RE.search(sentence)) / sentence_count, 4)
    function_density = round(len(_FUNCTION_WORD_RE.findall(normalized)) / max(1, char_count / 18), 4)
    helper_usage = round(len(_HELPER_VERB_RE.findall(normalized)) / sentence_count, 4)
    explanatory = round(sum(1 for sentence in sentences if _EXPLANATORY_RE.search(sentence)) / sentence_count, 4)
    compactness = round(len(_COMPACT_NATIVE_RE.findall(normalized)) / sentence_count, 4)
    l2_texture = round(len(_L2_TEXTURE_RE.findall(normalized)) / sentence_count, 4)
    native_fluency = round(
        max(
            0.0,
            min(
                1.0,
                compactness * 0.55
                + (1.0 - min(1.0, function_density)) * 0.25
                + (1.0 - min(1.0, l2_texture)) * 0.20,
            ),
        ),
        4,
    )
    return StyleDistribution(
        sentence_count=sentence_count,
        avg_sentence_length=avg_length,
        clause_per_sentence=clause_density,
        main_clause_delay_ratio=main_delay,
        function_word_density=function_density,
        helper_verb_usage=helper_usage,
        explanatory_rewrite_ratio=explanatory,
        compactness=compactness,
        native_fluency=native_fluency,
        l2_texture=l2_texture,
    )


def describe_style_distribution_by_class(text: str) -> dict[str, dict[str, object]]:
    """Summarize style distributions separately for each paragraph class."""

    by_class: dict[str, list[str]] = defaultdict(list)
    for paragraph in _extract_alignment_paragraphs(text):
        by_class[paragraph.paragraph_class].append(paragraph.text)
    distributions: dict[str, dict[str, object]] = {}
    for paragraph_class, paragraphs in by_class.items():
        distributions[paragraph_class] = {
            **describe_style_distribution("\n\n".join(paragraphs)).to_dict(),
            "paragraph_count": len(paragraphs),
        }
    return distributions


def class_aware_style_match_ratio(model_text: str, target_text: str) -> float:
    """Compare style distributions by paragraph class instead of one global average."""

    model_by_class = describe_style_distribution_by_class(model_text)
    target_by_class = describe_style_distribution_by_class(target_text)
    weighted_total = 0.0
    weight_sum = 0.0
    for paragraph_class, target_distribution in target_by_class.items():
        model_distribution = model_by_class.get(paragraph_class)
        if model_distribution is None:
            continue
        paragraph_count = max(1, int(target_distribution.get("paragraph_count", 1)))
        weight = _ALIGNMENT_CLASS_WEIGHTS.get(paragraph_class, 1.0) * paragraph_count
        ratio = _style_match_ratio(
            _style_distribution_from_dict(model_distribution),
            _style_distribution_from_dict(target_distribution),
        )
        weighted_total += ratio * weight
        weight_sum += weight
    if weight_sum == 0:
        return 1.0
    return round(weighted_total / weight_sum, 4)


def schedule_class_aware_repairs(
    *,
    source_text: str,
    model_output: str,
    target_text: str,
) -> tuple[str, list[str]]:
    """Repair only low-match paragraph classes with class-specific guarded budgets."""

    target_by_class = {
        name: _style_distribution_from_dict(payload)
        for name, payload in describe_style_distribution_by_class(target_text).items()
    }
    current_text = model_output
    actions: list[str] = []
    current_report = analyze_target_style_alignment(
        model_output=current_text,
        target_text=target_text,
        source_text=source_text,
    )
    current_class_match = float(current_report.get("class_aware_style_match_ratio", 1.0))
    current_global_match = float(current_report.get("style_distribution_match_ratio", 1.0))
    current_grammar = float(current_report.get("grammar_error_rate", 0.0))
    current_term_drift = int(current_report.get("terminology_drift", 0))
    current_evidence_drift = int(current_report.get("evidence_drift", 0))
    breakdown = current_report.get("class_alignment_breakdown", {})

    low_match_classes = [
        class_name
        for class_name, payload in breakdown.items()
        if float(payload.get("class_style_match_ratio", 1.0)) < 0.70
    ]

    for class_name in low_match_classes:
        budget = _CLASS_REPAIR_BUDGETS.get(class_name, 0)
        if budget <= 0:
            continue
        for _ in range(budget):
            paragraphs = [
                paragraph
                for paragraph in _extract_alignment_paragraphs(current_text)
                if paragraph.included_in_body_prose and paragraph.paragraph_class == class_name
            ]
            if not paragraphs:
                break
            paragraph_target = target_by_class.get(class_name)
            if paragraph_target is None:
                break
            paragraph = max(
                paragraphs,
                key=lambda item: _paragraph_gap_score(describe_style_distribution(item.text), paragraph_target, class_name),
            )
            candidate_text, candidate_actions = _repair_paragraph_for_class(
                source_text=source_text,
                current_text=current_text,
                paragraph=paragraph,
                target_distribution=paragraph_target,
                class_name=class_name,
                current_term_drift=current_term_drift,
                current_evidence_drift=current_evidence_drift,
                current_grammar=current_grammar,
                current_global_match=current_global_match,
                current_class_match=current_class_match,
                target_text=target_text,
            )
            if candidate_text == current_text:
                break
            current_text = candidate_text
            actions.extend(candidate_actions)
            current_report = analyze_target_style_alignment(
                model_output=current_text,
                target_text=target_text,
                source_text=source_text,
            )
            current_class_match = float(current_report.get("class_aware_style_match_ratio", 1.0))
            current_global_match = float(current_report.get("style_distribution_match_ratio", 1.0))
            current_grammar = float(current_report.get("grammar_error_rate", 0.0))
            current_term_drift = int(current_report.get("terminology_drift", 0))
            current_evidence_drift = int(current_report.get("evidence_drift", 0))

    return current_text, _deduplicate(actions)


def align_text_to_target_style(
    *,
    source_text: str,
    model_output: str,
    target_text: str | None,
) -> tuple[str, list[str]]:
    """Apply conservative target-style fitting repairs without copying target wording."""

    if not target_text:
        return model_output, []

    current_text, actions = schedule_class_aware_repairs(
        source_text=source_text,
        model_output=model_output,
        target_text=target_text,
    )
    current_text = _keep_evidence_scope(source_text, model_output, current_text)
    return current_text, _deduplicate(actions)


def split_sentences(text: str) -> list[str]:
    """Split text into simple sentence-like units for style statistics."""

    sentences = [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]
    return [sentence for sentence in sentences if re.search(r"[\u4e00-\u9fffA-Za-z0-9]", sentence)]


def _expand_compact_clauses(text: str) -> str:
    replacements = (
        ("兼顾", "能够同时兼顾"),
        ("适用于", "比较适合用来"),
        ("提取特征", "进行特征提取"),
        ("实现融合", "来实现融合"),
    )
    return _replace_limited(text, replacements, limit=4)


def _add_function_word_support(text: str) -> str:
    replacements = (
        ("用于", "是用来"),
        ("分析", "进行分析"),
        ("验证", "进行验证"),
    )
    return _replace_limited(text, replacements, limit=3)


def _inject_explanatory_phrases(text: str) -> str:
    replacements = (
        ("模块用于", "模块的主要作用是用来"),
        ("策略用于", "策略的目的，是通过"),
        ("模型用于", "模型是用来"),
        ("系统用于", "系统是用来"),
    )
    return _replace_limited(text, replacements, limit=2)


def _expand_explanatory_support(text: str) -> str:
    replacements = (
        ("说明", "能够说明"),
        ("表明", "能够表明"),
        ("采用", "会采用"),
        ("缓解", "来缓解"),
        ("降低", "来降低"),
        ("提高", "来提高"),
        ("保持稳定", "保持比较稳定"),
        ("支持", "提供支持"),
    )
    return _replace_limited(text, replacements, limit=2)


def _reduce_native_like_fluency(text: str) -> str:
    replacements = (
        ("构成", "形成了"),
        ("负责", "负责相关"),
        ("支撑", "提供支撑"),
        ("控制", "进行控制"),
    )
    return _replace_limited(text, replacements, limit=2)


def _introduce_mild_l2_texture(text: str) -> str:
    replacements = (
        ("该模型", "这个模型"),
        ("该系统", "这个系统"),
        ("该分支", "这个分支"),
        ("该模块", "这个模块"),
    )
    return _replace_limited(text, replacements, limit=2)


def _keep_evidence_scope(source_text: str, original_output: str, repaired_output: str) -> str:
    if source_backed_evidence_drift(source_text, repaired_output) > source_backed_evidence_drift(source_text, original_output):
        return original_output
    if _terminology_drift(source_text, repaired_output) > _terminology_drift(source_text, original_output):
        return original_output
    if _grammar_error_rate(split_sentences(extract_target_style_body_prose(repaired_output))) > 0.02:
        return original_output
    return repaired_output


def _extract_alignment_paragraphs(text: str) -> list[AlignmentParagraph]:
    normalized = _CODE_FENCE_RE.sub("\n", text.replace("\r\n", "\n"))
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", normalized) if block.strip()]
    paragraphs: list[AlignmentParagraph] = []
    current_heading = ""
    front_matter = True
    for block in blocks:
        if _HEADING_RE.match(block):
            current_heading = block
            front_matter = False
            continue
        if front_matter and len(block.splitlines()) == 1 and len(re.sub(r"\s+", "", block)) <= 40:
            continue
        if _SECTION_LABEL_RE.match(block):
            continue
        paragraph_class = classify_alignment_paragraph(block, current_heading)
        paragraphs.append(
            AlignmentParagraph(
                text=block,
                heading=current_heading,
                paragraph_class=paragraph_class,
                included_in_body_prose=not _exclude_alignment_block(block, current_heading),
            )
        )
    return paragraphs


def _exclude_alignment_block(block: str, heading_context: str) -> bool:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return True
    if _SECTION_LABEL_RE.match(block):
        return True
    if block.startswith("目 录") or _TOC_LINK_RE.search(block):
        return True
    if len(lines) == 1:
        line = lines[0]
        if (
            _IMAGE_LINE_RE.match(line)
            or _LIST_ONLY_RE.match(line)
            or _TABLE_RE.match(line)
            or _CITATION_LINE_RE.match(line)
            or _FIGURE_CAPTION_RE.match(line)
            or _looks_like_path_or_checkpoint(line)
        ):
            return True
    if all(_TABLE_RE.match(line) for line in lines):
        return True
    if all(_IMAGE_LINE_RE.match(line) for line in lines):
        return True
    if any(_looks_like_formula_line(line) for line in lines) and _technical_density_score(block) >= 0.28:
        return True
    if _FIGURE_CAPTION_RE.match(lines[0]):
        return True
    if _contains_any(heading_context, ("参考文献", "附录")):
        return True
    if _looks_like_path_or_checkpoint(block):
        return True
    if _INLINE_CITATION_RE.findall(block) and len(re.sub(r"[^\u4e00-\u9fffA-Za-z]", "", block)) < 18:
        return True
    if classify_alignment_paragraph(block, heading_context) == "technical_dense":
        return True
    return False


def _replace_limited(text: str, replacements: tuple[tuple[str, str], ...], *, limit: int) -> str:
    repaired = text
    count = 0
    for old, new in replacements:
        if old == new:
            continue
        while old in repaired and count < limit:
            repaired = repaired.replace(old, new, 1)
            count += 1
        if count >= limit:
            break
    return repaired


def _apply_guarded_paragraph_action(
    source_text: str,
    current_text: str,
    paragraph_text: str,
    transform: callable,
    baseline_term_drift: int,
    baseline_evidence_drift: int,
) -> tuple[str, bool]:
    if paragraph_text not in current_text:
        return current_text, False
    repaired_paragraph = transform(paragraph_text)
    if repaired_paragraph == paragraph_text:
        return current_text, False
    candidate_text = current_text.replace(paragraph_text, repaired_paragraph, 1)
    if _terminology_drift(source_text, candidate_text) > baseline_term_drift:
        return current_text, False
    if source_backed_evidence_drift(source_text, candidate_text) > baseline_evidence_drift:
        return current_text, False
    if _grammar_error_rate(split_sentences(extract_target_style_body_prose(candidate_text))) > 0.02:
        return current_text, False
    return candidate_text, True


def _repair_paragraph_for_class(
    *,
    source_text: str,
    current_text: str,
    paragraph: AlignmentParagraph,
    target_distribution: StyleDistribution,
    class_name: str,
    current_term_drift: int,
    current_evidence_drift: int,
    current_grammar: float,
    current_global_match: float,
    current_class_match: float,
    target_text: str,
) -> tuple[str, list[str]]:
    paragraph_distribution = describe_style_distribution(paragraph.text)
    class_actions: list[str] = []
    candidate_text = current_text

    if paragraph_distribution.avg_sentence_length + 4 < target_distribution.avg_sentence_length:
        candidate_text, changed = _apply_guarded_paragraph_action(
            source_text,
            candidate_text,
            paragraph.text,
            _expand_compact_clauses,
            current_term_drift,
            current_evidence_drift,
        )
        if changed:
            class_actions.append(f"{class_name}:expand_compact_clause")
            paragraph = _refresh_paragraph(candidate_text, paragraph)
    if class_name in {"abstract_prose", "significance_prose", "summary_conclusion", "background_prose"}:
        if paragraph and describe_style_distribution(paragraph.text).function_word_density + 0.04 < target_distribution.function_word_density:
            candidate_text, changed = _apply_guarded_paragraph_action(
                source_text,
                candidate_text,
                paragraph.text,
                _add_function_word_support,
                current_term_drift,
                current_evidence_drift,
            )
            if changed:
                class_actions.append(f"{class_name}:add_function_word_support")
                paragraph = _refresh_paragraph(candidate_text, paragraph)
    if class_name in {"abstract_prose", "significance_prose", "summary_conclusion", "result_analysis"}:
        if paragraph and describe_style_distribution(paragraph.text).explanatory_rewrite_ratio + 0.08 < target_distribution.explanatory_rewrite_ratio:
            candidate_text, changed = _apply_guarded_paragraph_action(
                source_text,
                candidate_text,
                paragraph.text,
                _inject_explanatory_phrases,
                current_term_drift,
                current_evidence_drift,
            )
            if changed:
                class_actions.append(f"{class_name}:inject_explanatory_phrase")
                paragraph = _refresh_paragraph(candidate_text, paragraph)
    if class_name in {"significance_prose", "summary_conclusion", "training_strategy", "result_analysis"}:
        if paragraph:
            candidate_text, changed = _apply_guarded_paragraph_action(
                source_text,
                candidate_text,
                paragraph.text,
                _expand_explanatory_support,
                current_term_drift,
                current_evidence_drift,
            )
            if changed:
                class_actions.append(f"{class_name}:expand_explanatory_support")
                paragraph = _refresh_paragraph(candidate_text, paragraph)
    if class_name in _L2_ALLOWED_CLASSES:
        if paragraph and describe_style_distribution(paragraph.text).native_fluency > target_distribution.native_fluency + 0.08:
            candidate_text, changed = _apply_guarded_paragraph_action(
                source_text,
                candidate_text,
                paragraph.text,
                _reduce_native_like_fluency,
                current_term_drift,
                current_evidence_drift,
            )
            if changed:
                class_actions.append(f"{class_name}:reduce_native_like_fluency")
                paragraph = _refresh_paragraph(candidate_text, paragraph)
        if paragraph and target_distribution.l2_texture > describe_style_distribution(paragraph.text).l2_texture + 0.08:
            candidate_text, changed = _apply_guarded_paragraph_action(
                source_text,
                candidate_text,
                paragraph.text,
                _introduce_mild_l2_texture,
                current_term_drift,
                current_evidence_drift,
            )
            if changed:
                class_actions.append(f"{class_name}:introduce_mild_l2_texture")

    if candidate_text == current_text:
        return current_text, []

    candidate_report = analyze_target_style_alignment(
        model_output=candidate_text,
        target_text=target_text,
        source_text=source_text,
    )
    if int(candidate_report.get("terminology_drift", 0)) > current_term_drift:
        return current_text, []
    if int(candidate_report.get("evidence_drift", 0)) > current_evidence_drift:
        return current_text, []
    if float(candidate_report.get("grammar_error_rate", 0.0)) > current_grammar:
        return current_text, []
    if float(candidate_report.get("style_distribution_match_ratio", 0.0)) < current_global_match:
        return current_text, []
    if float(candidate_report.get("class_aware_style_match_ratio", 0.0)) < current_class_match:
        return current_text, []
    return candidate_text, class_actions


def _refresh_paragraph(text: str, paragraph: AlignmentParagraph) -> AlignmentParagraph | None:
    for candidate in _extract_alignment_paragraphs(text):
        if candidate.heading == paragraph.heading and candidate.paragraph_class == paragraph.paragraph_class:
            if candidate.text[:32] == paragraph.text[:32] or paragraph.text[:32] in candidate.text:
                return candidate
    return None


def _paragraph_gap_score(
    paragraph_distribution: StyleDistribution,
    target_distribution: StyleDistribution,
    class_name: str,
) -> float:
    score = abs(paragraph_distribution.avg_sentence_length - target_distribution.avg_sentence_length) / 80
    score += abs(paragraph_distribution.function_word_density - target_distribution.function_word_density)
    score += abs(paragraph_distribution.explanatory_rewrite_ratio - target_distribution.explanatory_rewrite_ratio)
    if class_name in _L2_ALLOWED_CLASSES:
        score += abs(paragraph_distribution.native_fluency - target_distribution.native_fluency)
    return score


def _deviation_sentence_ids(
    sentences: list[str],
    target_distribution: StyleDistribution,
) -> tuple[list[int], list[int], list[int]]:
    over_native: list[int] = []
    under_explained: list[int] = []
    compact: list[int] = []
    for index, sentence in enumerate(sentences, start=1):
        if _COMPACT_NATIVE_RE.search(sentence) and target_distribution.native_fluency < 0.72:
            over_native.append(index)
        if not _EXPLANATORY_RE.search(sentence) and target_distribution.explanatory_rewrite_ratio >= 0.20:
            under_explained.append(index)
        if len(re.sub(r"\s+", "", sentence)) < max(24, target_distribution.avg_sentence_length * 0.65):
            compact.append(index)
    return over_native[:20], under_explained[:20], compact[:20]


def _style_deviation_examples(
    *,
    model_sentences: list[str],
    target_distribution: StyleDistribution,
    over_native_ids: list[int],
    under_explained_ids: list[int],
    compact_ids: list[int],
) -> list[dict[str, object]]:
    ids = _deduplicate([*over_native_ids[:5], *under_explained_ids[:5], *compact_ids[:5]])
    examples: list[dict[str, object]] = []
    for sentence_id in ids[:12]:
        sentence = model_sentences[sentence_id - 1]
        reasons: list[str] = []
        if sentence_id in over_native_ids:
            reasons.append("over_native_fluency")
        if sentence_id in under_explained_ids:
            reasons.append("not_explanatory_enough")
        if sentence_id in compact_ids:
            reasons.append("more_compact_than_target")
        examples.append(
            {
                "sentence_id": sentence_id,
                "sentence": sentence[:160],
                "reasons": reasons,
                "target_avg_sentence_length": target_distribution.avg_sentence_length,
            }
        )
    return examples


def _build_class_alignment_breakdown(
    model_by_class: dict[str, dict[str, object]],
    target_by_class: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    breakdown: dict[str, dict[str, object]] = {}
    for class_name, target_payload in target_by_class.items():
        model_payload = model_by_class.get(class_name)
        if model_payload is None:
            continue
        model_distribution = _style_distribution_from_dict(model_payload)
        target_distribution = _style_distribution_from_dict(target_payload)
        breakdown[class_name] = {
            "class_style_match_ratio": _style_match_ratio(model_distribution, target_distribution),
            "sentence_length_gap": _absdiff(model_distribution.avg_sentence_length, target_distribution.avg_sentence_length),
            "function_word_gap": _absdiff(model_distribution.function_word_density, target_distribution.function_word_density),
            "explanatory_gap": round(max(0.0, target_distribution.explanatory_rewrite_ratio - model_distribution.explanatory_rewrite_ratio), 4),
            "native_fluency_gap": round(max(0.0, model_distribution.native_fluency - target_distribution.native_fluency), 4),
            "paragraph_count": int(target_payload.get("paragraph_count", 0)),
        }
    return breakdown


def _worst_alignment_classes(class_breakdown: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(
        (
            {"class_name": class_name, **payload}
            for class_name, payload in class_breakdown.items()
        ),
        key=lambda item: (item.get("class_style_match_ratio", 1.0), -item.get("paragraph_count", 0)),
    )
    return ordered[:3]


def _strategy_adjustments(
    *,
    avg_length_diff: float,
    function_diff: float,
    explanatory_gap: float,
    compactness_gap: float,
    native_gap: float,
    l2_gap: float,
) -> list[str]:
    actions: list[str] = []
    if avg_length_diff > 6:
        actions.append("expand_compact_clause")
    if function_diff > 0.05:
        actions.append("add_function_word_support")
    if explanatory_gap > 0.06:
        actions.append("inject_explanatory_phrase")
        actions.append("add_action_purpose_structure")
    if compactness_gap > 0.06:
        actions.append("expand_high_level_statement")
    if native_gap > 0.06:
        actions.append("reduce_native_like_fluency")
        actions.append("slightly_redundant_rephrase")
    if l2_gap > 0.06:
        actions.append("introduce_mild_l2_texture")
    return _deduplicate(actions)


def _style_match_ratio(model_distribution: StyleDistribution, target_distribution: StyleDistribution) -> float:
    penalties = [
        min(1.0, abs(model_distribution.avg_sentence_length - target_distribution.avg_sentence_length) / 60),
        min(1.0, abs(model_distribution.clause_per_sentence - target_distribution.clause_per_sentence) / 2),
        min(1.0, abs(model_distribution.function_word_density - target_distribution.function_word_density) / 0.6),
        min(1.0, abs(model_distribution.helper_verb_usage - target_distribution.helper_verb_usage) / 1.2),
        min(1.0, abs(model_distribution.explanatory_rewrite_ratio - target_distribution.explanatory_rewrite_ratio) / 0.5),
        min(1.0, abs(model_distribution.native_fluency - target_distribution.native_fluency) / 0.5),
        min(1.0, abs(model_distribution.l2_texture - target_distribution.l2_texture) / 0.5),
    ]
    return round(max(0.0, 1.0 - sum(penalties) / len(penalties)), 4)


def _grammar_error_rate(sentences: list[str]) -> float:
    if not sentences:
        return 0.0
    errors = sum(1 for sentence in sentences if _GRAMMAR_RISK_RE.search(sentence))
    return round(errors / len(sentences), 4)


def _terminology_drift(source_text: str, model_output: str) -> int:
    """Count only true protected-term disappearance or protected token changes."""

    if not source_text:
        return 0
    source_terms = normalized_protected_term_counter(source_text)
    output_terms = normalized_protected_term_counter(model_output)
    missing = source_terms - output_terms
    return sum(missing.values())


def _looks_like_protected_term(term: str) -> bool:
    chinese_term_ok = (
        term.endswith(("模型", "数据集", "算法", "模块", "分支", "损失", "指标", "路径", "策略", "机制", "接口", "框架"))
        and len(term) <= 6
        and not _contains_any(term, _CHINESE_TERM_STOPWORDS)
    )
    return bool(
        _PATH_RE.fullmatch(term)
        or _CHECKPOINT_RE.search(term)
        or re.search(r"[A-Z]", term)
        or "_" in term
        or "/" in term
        or chinese_term_ok
    )


def _looks_like_ignorable_marker(term: str) -> bool:
    return bool(
        _FIGURE_CAPTION_RE.match(term)
        or _INLINE_CITATION_RE.fullmatch(term)
        or re.fullmatch(r"(?:图|表)\d+(?:[-－.]\d+)*", term)
    )


def _looks_like_formula_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(_FORMULA_HINT_RE.search(stripped) and len(re.findall(r"[\u4e00-\u9fff]", stripped)) < 12)


def _looks_like_path_or_checkpoint(text: str) -> bool:
    return bool(_PATH_RE.search(text) or _CHECKPOINT_RE.search(text))


def _is_technical_dense_paragraph(text: str, heading_context: str = "") -> bool:
    heading = heading_context or ""
    density = _technical_density_score(text)
    if _contains_any(heading + text, ("损失函数", "loss", "checkpoint", "版本", "路径", "阈值设置", "指标定义")):
        return True
    return density >= 0.32


def _technical_density_score(text: str) -> float:
    token_count = max(1, len(text))
    english_like = len(re.findall(r"[A-Za-z][A-Za-z0-9_./\\-]*", text))
    numbers = len(_NUMBER_RE.findall(text))
    citations = len(_INLINE_CITATION_RE.findall(text))
    symbols = len(re.findall(r"[=<>$^_/\\]", text))
    return (english_like * 4 + numbers * 3 + citations * 4 + symbols * 2) / token_count


def _normalize_for_alignment(text: str) -> str:
    return text.replace("\u3000", " ").replace("\xa0", " ")


def _normalize_protected_term(term: str) -> str:
    normalized = re.sub(r"\s+", "", term).strip("`*_[](){}<>")
    return normalized.lower() if re.search(r"[A-Za-z]", normalized) else normalized


def _style_distribution_from_dict(payload: dict[str, object]) -> StyleDistribution:
    return StyleDistribution(
        sentence_count=int(payload.get("sentence_count", 0)),
        avg_sentence_length=float(payload.get("avg_sentence_length", 0.0)),
        clause_per_sentence=float(payload.get("clause_per_sentence", 0.0)),
        main_clause_delay_ratio=float(payload.get("main_clause_delay_ratio", 0.0)),
        function_word_density=float(payload.get("function_word_density", 0.0)),
        helper_verb_usage=float(payload.get("helper_verb_usage", 0.0)),
        explanatory_rewrite_ratio=float(payload.get("explanatory_rewrite_ratio", 0.0)),
        compactness=float(payload.get("compactness", 0.0)),
        native_fluency=float(payload.get("native_fluency", 0.0)),
        l2_texture=float(payload.get("l2_texture", 0.0)),
    )


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _absdiff(left: float, right: float) -> float:
    return round(abs(left - right), 4)


def _deduplicate(values: list[int] | list[str]) -> list:
    seen: set[object] = set()
    ordered: list[object] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
