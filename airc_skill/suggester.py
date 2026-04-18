from __future__ import annotations

import re
from dataclasses import dataclass

from .chunker import chunk_text
from .markdown_guard import protect, restore

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMBER_RE = re.compile(r"\d")
_CITATION_RE = re.compile(r"\[[0-9,\-\s]+\]|\((?:19|20)\d{2}\)|et al\.", re.IGNORECASE)

_VAGUE_MARKERS = (
    "一些研究",
    "很多研究",
    "相关研究",
    "许多学者",
    "在很多方面",
    "具有重要意义",
    "起到了重要作用",
    "值得进一步研究",
    "many studies",
    "some studies",
    "important role",
    "significant impact",
)
_RESEARCH_HINTS = ("研究", "文献", "实验", "调查", "样本", "study", "survey", "experiment")
_CASE_HINTS = ("案例", "实践", "应用", "场景", "对象", "现象", "问题", "case", "scenario")


@dataclass(frozen=True)
class Suggestion:
    paragraph_index: int
    excerpt: str
    suggestions: list[str]


def generate_suggestions(text: str, suffix: str) -> list[Suggestion]:
    working_text = text
    placeholders: dict[str, str] = {}
    if suffix.lower() == ".md":
        working_text, placeholders = protect(text)

    chunks = chunk_text(working_text, suffix=suffix)
    suggestions: list[Suggestion] = []
    paragraph_index = 0

    for chunk in chunks:
        if not chunk.rewritable:
            continue
        paragraph_index += 1
        paragraph = restore(chunk.text, placeholders).strip()
        if len(paragraph) < 50:
            continue

        suggestion_items = _build_suggestions(paragraph)
        if suggestion_items:
            suggestions.append(
                Suggestion(
                    paragraph_index=paragraph_index,
                    excerpt=_build_excerpt(paragraph),
                    suggestions=suggestion_items,
                )
            )

    return suggestions


def _build_suggestions(paragraph: str) -> list[str]:
    lowered = paragraph.lower()
    has_vague_marker = any(marker.lower() in lowered for marker in _VAGUE_MARKERS)
    has_year = bool(_YEAR_RE.search(paragraph))
    has_number = bool(_NUMBER_RE.search(paragraph))
    has_citation = bool(_CITATION_RE.search(paragraph))

    items: list[str] = []

    if any(token.lower() in lowered for token in _RESEARCH_HINTS) and not has_citation:
        items.append("可补充真实文献来源、引用位置或已有研究脉络。")
    if not has_year and ("近年来" in paragraph or "目前" in paragraph or "当前" in paragraph or has_vague_marker):
        items.append("可补充真实年份、时间范围或研究阶段信息。")
    if any(token.lower() in lowered for token in _CASE_HINTS) and not has_number:
        items.append("可补充具体案例、场景或研究对象细节。")
    if any(token.lower() in lowered for token in ("样本", "调查", "实验", "问卷", "访谈", "survey", "experiment")) and not has_number:
        items.append("可补充真实样本量、方法设置或对象说明。")
    if has_vague_marker and not items:
        items.append("可补充可核验的事实依据，以减少空泛表述。")

    deduplicated: list[str] = []
    for item in items:
        if item not in deduplicated:
            deduplicated.append(item)
    return deduplicated


def _build_excerpt(paragraph: str, limit: int = 70) -> str:
    compact = re.sub(r"\s+", " ", paragraph).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."
