from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Pattern

from .markdown_guard import PLACEHOLDER_RE

_HEADING_LINE_RE = re.compile(
    r"(?m)^(?:\s{0,3}#{1,6}\s+.*|第[一二三四五六七八九十百0-9]+章.*|摘\s*要\s*$|关\s*键\s*词[:：]?.*$|Abstract\s*$|Keywords[:：]?.*)$"
)
_REFERENCE_LINE_RE = re.compile(r"(?m)^\[\d+\].*$")
_FORMULA_BLOCK_RE = re.compile(r"(?ms)^\$\$.*?^\$\$[ \t]*$", re.MULTILINE)
_FORMULA_INLINE_RE = re.compile(r"\\\(.+?\\\)|\\\[.+?\\\]")
_FORMULA_LINE_RE = re.compile(
    r"(?m)^\s*(?:\*+\s*)?(?:[\(（][^)）]+[\)）]\s*)?(?:[A-Za-z][A-Za-z0-9_]*|[α-ωΑ-Ωτλσ]+).{0,40}?=\s*.+$"
)
_CITATION_RE = re.compile(
    r"\[\d+\](?:\[\d+\])*|\([A-Z][A-Za-z\-]+,\s*\d{4}[a-z]?\)|（[A-Z][A-Za-z\-]+,\s*\d{4}[a-z]?）"
)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]+\)")
_LINK_RE = re.compile(r"(?<!!)\[[^\]]+]\([^)]+\)")
_LINK_DEF_RE = re.compile(r"(?m)^\[[^\]]+\]:\s+\S+.*$")
_AUTOLINK_RE = re.compile(r"<https?://[^>\n]+>")
_BARE_URL_RE = re.compile(r"https?://[^\s<>()]+(?:\([^\s<>()]*\)[^\s<>()]*)*")
_CAPTION_FRAGMENT_RE = re.compile(
    r"(?:如图|如表|图\s*\d+(?:-\d+)?|表\s*\d+(?:-\d+)?|Figure\s*\d+|Table\s*\d+)[^。\n]*?(?:所示|如下|展示|流程)[：:]?"
)
_CAPTION_LINE_RE = re.compile(
    r"(?m)^(?:\s*(?:图|表)\s*\d+(?:-\d+)?[^\n]*|.*(?:如图|如表|Figure|Table)[^\n]*?(?:所示|如下|展示|流程)[：:]?)$"
)
_PATH_RE = re.compile(
    r"(?:[A-Za-z]:\\[^\s<>()]+|(?:[\w.-]+/)+[\w./-]+|[\w.-]+\.(?:pth|pt|ckpt|png|jpg|jpeg|pdf))"
)
_CHECKPOINT_RE = re.compile(r"\b(?:checkpoint[s]?/[^\s<>()]+|[\w.-]+\.(?:pth|pt|ckpt))\b")
_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_TRAILING_SPACE_RE = re.compile(r"(?m)[ \t]+$")
_TECH_TERM_PATTERNS = (
    re.compile(r"\b[A-Z]{2,}[A-Za-z0-9_-]*\b"),
    re.compile(r"\b[A-Za-z]+_[A-Za-z0-9_]+\b"),
    re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b"),
    re.compile(r"\b[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9_-]*\b"),
    re.compile(r"\b[A-Za-z]+[0-9]+[A-Za-z0-9_-]*\b"),
)


@dataclass(frozen=True)
class CoreSnapshot:
    headings: list[str]
    formulas: list[str]
    citations: list[str]
    technical_terms: list[str]
    numbers: list[str]
    paths: list[str]
    heading_formats: list[str]
    english_blocks: list[str]
    placeholders: list[str]
    captions: list[str]
    caption_has_colon_period: bool
    markdown_symbol_counts: dict[str, int]
    blank_line_signature: list[int]
    trailing_space_count: int


@dataclass(frozen=True)
class ProtectionStats:
    frozen_heading_blocks: int
    frozen_formula_blocks: int
    frozen_english_blocks: int
    frozen_placeholder_blocks: int
    frozen_caption_blocks: int


def protect_core_content(text: str, suffix: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}
    protected = text
    counter = 0

    protected, counter = _protect_paragraph_blocks(
        protected,
        label="CORE_ENGLISH",
        placeholders=placeholders,
        counter=counter,
        predicate=_is_english_block,
    )

    for label, pattern in (
        ("CORE_HEADING", _HEADING_LINE_RE),
        ("CORE_REFERENCE", _REFERENCE_LINE_RE),
        ("CORE_FORMULA_BLOCK", _FORMULA_BLOCK_RE),
        ("CORE_FORMULA_INLINE", _FORMULA_INLINE_RE),
        ("CORE_FORMULA_LINE", _FORMULA_LINE_RE),
        ("CORE_CITATION", _CITATION_RE),
        ("CORE_CAPTION_LINE", _CAPTION_LINE_RE),
        ("CORE_CAPTION", _CAPTION_FRAGMENT_RE),
        ("CORE_CHECKPOINT", _CHECKPOINT_RE),
        ("CORE_PATH", _PATH_RE),
    ):
        protected, counter = _protect_with_pattern(
            protected,
            pattern=pattern,
            label=label,
            placeholders=placeholders,
            counter=counter,
        )

    for pattern in _TECH_TERM_PATTERNS:
        protected, counter = _protect_with_pattern(
            protected,
            pattern=pattern,
            label="CORE_TERM",
            placeholders=placeholders,
            counter=counter,
        )

    protected, counter = _protect_with_pattern(
        protected,
        pattern=_NUMBER_RE,
        label="CORE_NUMBER",
        placeholders=placeholders,
        counter=counter,
    )
    return protected, placeholders


def restore_core_content(text: str, placeholders: dict[str, str]) -> str:
    restored = text
    for token, original in placeholders.items():
        restored = restored.replace(token, original)
    return restored


def snapshot_core_content(text: str, suffix: str) -> CoreSnapshot:
    return CoreSnapshot(
        headings=_HEADING_LINE_RE.findall(text),
        formulas=_extract_formulas(text),
        citations=_CITATION_RE.findall(text),
        technical_terms=_extract_technical_terms(text),
        numbers=_NUMBER_RE.findall(text),
        paths=_extract_paths(text),
        heading_formats=_extract_heading_formats(text),
        english_blocks=_extract_english_blocks(text),
        placeholders=_extract_placeholders(text),
        captions=_extract_captions(text),
        caption_has_colon_period="：。" in text,
        markdown_symbol_counts=_extract_markdown_symbol_counts(text),
        blank_line_signature=_extract_blank_line_signature(text),
        trailing_space_count=len(_TRAILING_SPACE_RE.findall(text)),
    )


def compare_core_snapshots(original: CoreSnapshot, revised: CoreSnapshot) -> dict[str, bool]:
    return {
        "title_integrity_check": original.headings == revised.headings,
        "formula_integrity_check": original.formulas == revised.formulas,
        "citation_integrity_check": original.citations == revised.citations,
        "terminology_integrity_check": Counter(original.technical_terms) == Counter(revised.technical_terms),
        "numeric_integrity_check": original.numbers == revised.numbers,
        "path_integrity_check": original.paths == revised.paths,
        "heading_format_integrity_check": original.heading_formats == revised.heading_formats,
        "english_spacing_integrity_check": original.english_blocks == revised.english_blocks,
        "placeholder_integrity_check": original.placeholders == revised.placeholders,
        "caption_punctuation_integrity_check": (
            original.captions == revised.captions and not revised.caption_has_colon_period
        ),
        "markdown_symbol_integrity_check": original.markdown_symbol_counts == revised.markdown_symbol_counts,
        "linebreak_whitespace_integrity_check": (
            original.blank_line_signature == revised.blank_line_signature
            and revised.trailing_space_count <= original.trailing_space_count
        ),
    }


def collect_protection_stats(text: str, suffix: str) -> ProtectionStats:
    snapshot = snapshot_core_content(text, suffix)
    return ProtectionStats(
        frozen_heading_blocks=len(snapshot.heading_formats),
        frozen_formula_blocks=len(snapshot.formulas),
        frozen_english_blocks=len(snapshot.english_blocks),
        frozen_placeholder_blocks=len(snapshot.placeholders),
        frozen_caption_blocks=len(snapshot.captions),
    )


def _extract_formulas(text: str) -> list[str]:
    formulas: list[str] = []
    formulas.extend(_FORMULA_BLOCK_RE.findall(text))
    formulas.extend(_FORMULA_INLINE_RE.findall(text))
    formulas.extend(_FORMULA_LINE_RE.findall(text))
    return formulas


def _extract_technical_terms(text: str) -> list[str]:
    found: list[str] = []
    for pattern in _TECH_TERM_PATTERNS:
        found.extend(match.group(0) for match in pattern.finditer(text))
    return found


def _extract_paths(text: str) -> list[str]:
    paths = _CHECKPOINT_RE.findall(text)
    paths.extend(match.group(0) for match in _PATH_RE.finditer(text))
    return _deduplicate_preserve_order(paths)


def _extract_heading_formats(text: str) -> list[str]:
    return _HEADING_LINE_RE.findall(text)


def _extract_english_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        stripped = paragraph.strip()
        if not stripped or _CJK_RE.search(stripped):
            continue
        english_words = _ENGLISH_WORD_RE.findall(stripped)
        if len(english_words) < 4:
            continue
        blocks.append(stripped)
    return blocks


def _extract_placeholders(text: str) -> list[str]:
    placeholders: list[str] = []
    placeholders.extend(_IMAGE_RE.findall(text))
    placeholders.extend(_LINK_RE.findall(text))
    placeholders.extend(_LINK_DEF_RE.findall(text))
    placeholders.extend(_AUTOLINK_RE.findall(text))
    placeholders.extend(_BARE_URL_RE.findall(text))
    return placeholders


def _extract_captions(text: str) -> list[str]:
    captions: list[str] = []
    captions.extend(match.group(0).strip() for match in _CAPTION_LINE_RE.finditer(text))
    captions.extend(match.group(0).strip() for match in _CAPTION_FRAGMENT_RE.finditer(text))
    return _deduplicate_preserve_order(captions)


def _extract_markdown_symbol_counts(text: str) -> dict[str, int]:
    return {
        "headings": len(_HEADING_LINE_RE.findall(text)),
        "images": len(_IMAGE_RE.findall(text)),
        "links": len(_LINK_RE.findall(text)),
        "link_defs": len(_LINK_DEF_RE.findall(text)),
        "autolinks": len(_AUTOLINK_RE.findall(text)),
        "urls": len(_BARE_URL_RE.findall(text)),
        "reference_lines": len(_REFERENCE_LINE_RE.findall(text)),
    }


def _extract_blank_line_signature(text: str) -> list[int]:
    signature: list[int] = []
    streak = 0
    for line in text.splitlines():
        if line.strip():
            if streak:
                signature.append(streak)
                streak = 0
        else:
            streak += 1
    if streak:
        signature.append(streak)
    return signature


def _protect_paragraph_blocks(
    text: str,
    label: str,
    placeholders: dict[str, str],
    counter: int,
    predicate: Callable[[str], bool],
) -> tuple[str, int]:
    segments = re.split(r"(\n\s*\n)", text)
    protected_segments: list[str] = []

    for segment in segments:
        if not segment:
            continue
        if re.fullmatch(r"\n\s*\n", segment):
            protected_segments.append(segment)
            continue
        if predicate(segment):
            token = f"[[AIRC:{label}:{counter:04d}]]"
            placeholders[token] = segment
            protected_segments.append(token)
            counter += 1
        else:
            protected_segments.append(segment)

    return "".join(protected_segments), counter


def _protect_with_pattern(
    text: str,
    pattern: Pattern[str],
    label: str,
    placeholders: dict[str, str],
    counter: int,
) -> tuple[str, int]:
    parts = re.split(f"({PLACEHOLDER_RE.pattern})", text)
    protected_parts: list[str] = []

    for part in parts:
        if not part:
            continue
        if PLACEHOLDER_RE.fullmatch(part):
            protected_parts.append(part)
            continue

        def _replace(match: re.Match[str]) -> str:
            nonlocal counter
            token = f"[[AIRC:{label}:{counter:04d}]]"
            placeholders[token] = match.group(0)
            counter += 1
            return token

        protected_parts.append(pattern.sub(_replace, part))

    return "".join(protected_parts), counter


def _deduplicate_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _is_english_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped or _CJK_RE.search(stripped):
        return False
    english_words = _ENGLISH_WORD_RE.findall(stripped)
    if len(english_words) < 4:
        return False
    ascii_letters = sum(1 for char in stripped if char.isascii() and char.isalpha())
    visible = sum(1 for char in stripped if not char.isspace())
    if visible == 0:
        return False
    return ascii_letters / visible >= 0.45
