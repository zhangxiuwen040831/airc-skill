from __future__ import annotations

import re
from dataclasses import dataclass

from .markdown_guard import PLACEHOLDER_RE

_SENTENCE_BREAK_RE = re.compile(r"(?<=[。！？!?；;])")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")
_LINK_DEF_RE = re.compile(r"^\[[^\]]+\]:\s+\S+")
_PLACEHOLDER_BLOCK_RE = re.compile(
    rf"^(?:\s*(?:\*+\s*)?(?:{PLACEHOLDER_RE.pattern})(?:\s*(?:{PLACEHOLDER_RE.pattern}|\*+))*\s*)+$"
)
_CAPTION_RE = re.compile(
    r"^(?:"
    r"(?:图|表)\s*\d+(?:-\d+)?[^\n]*"
    r"|(?:Figure|Table)\s*\d+[^\n]*"
    r")$"
)
_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class Chunk:
    text: str
    rewritable: bool


def chunk_text(text: str, suffix: str, max_chars: int = 1500) -> list[Chunk]:
    normalized_suffix = suffix.lower()
    if normalized_suffix == ".txt":
        return _chunk_txt(text, max_chars=max_chars)
    if normalized_suffix == ".md":
        return _chunk_markdown(text, max_chars=max_chars)
    raise ValueError(f"Unsupported suffix for chunking: {suffix}")


def _chunk_txt(text: str, max_chars: int) -> list[Chunk]:
    lines = text.splitlines(keepends=True)
    chunks: list[Chunk] = []
    index = 0

    while index < len(lines):
        if lines[index].strip() == "":
            blanks: list[str] = []
            while index < len(lines) and lines[index].strip() == "":
                blanks.append(lines[index])
                index += 1
            chunks.append(Chunk(text="".join(blanks), rewritable=False))
            continue

        paragraph: list[str] = []
        while index < len(lines) and lines[index].strip() != "":
            paragraph.append(lines[index])
            index += 1
        chunks.extend(_split_rewritable_block("".join(paragraph), max_chars=max_chars))

    return chunks


def _chunk_markdown(text: str, max_chars: int) -> list[Chunk]:
    lines = text.splitlines(keepends=True)
    chunks: list[Chunk] = []
    index = 0

    while index < len(lines):
        current_line = lines[index]
        stripped = current_line.strip()

        if stripped == "":
            blanks: list[str] = []
            while index < len(lines) and lines[index].strip() == "":
                blanks.append(lines[index])
                index += 1
            chunks.append(Chunk(text="".join(blanks), rewritable=False))
            continue

        if _is_placeholder_only(stripped) or _is_heading_line(stripped) or _is_link_def(stripped):
            chunks.append(Chunk(text=current_line, rewritable=False))
            index += 1
            continue

        if _is_list_line(stripped):
            group: list[str] = []
            while index < len(lines) and lines[index].strip() != "":
                stripped_group = lines[index].strip()
                if _is_list_line(stripped_group) or lines[index].startswith((" ", "\t")):
                    group.append(lines[index])
                    index += 1
                    continue
                break
            chunks.append(Chunk(text="".join(group), rewritable=False))
            continue

        if _is_quote_line(stripped):
            group = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                group.append(lines[index])
                index += 1
            chunks.append(Chunk(text="".join(group), rewritable=False))
            continue

        if _is_table_line(stripped):
            group = []
            while index < len(lines) and lines[index].strip() != "" and _is_table_line(lines[index].strip()):
                group.append(lines[index])
                index += 1
            chunks.append(Chunk(text="".join(group), rewritable=False))
            continue

        paragraph: list[str] = []
        while index < len(lines):
            lookahead = lines[index].strip()
            if lookahead == "" or _starts_structural_block(lines[index]):
                break
            paragraph.append(lines[index])
            index += 1
        paragraph_text = "".join(paragraph)
        if _is_frozen_paragraph(paragraph_text):
            chunks.append(Chunk(text=paragraph_text, rewritable=False))
        else:
            chunks.extend(_split_rewritable_block(paragraph_text, max_chars=max_chars))

    return chunks


def _split_rewritable_block(text: str, max_chars: int) -> list[Chunk]:
    if not text:
        return []
    if len(text) <= max_chars:
        return [Chunk(text=text, rewritable=True)]

    sentences = [segment for segment in _SENTENCE_BREAK_RE.split(text) if segment]
    if len(sentences) <= 1:
        return _hard_split_block(text, max_chars=max_chars)

    chunks: list[Chunk] = []
    buffer = ""
    for sentence in sentences:
        if len(buffer) + len(sentence) <= max_chars or not buffer:
            buffer += sentence
        else:
            chunks.append(Chunk(text=buffer, rewritable=True))
            buffer = sentence

    if buffer:
        if len(buffer) > max_chars:
            chunks.extend(_hard_split_block(buffer, max_chars=max_chars))
        else:
            chunks.append(Chunk(text=buffer, rewritable=True))
    return chunks


def _hard_split_block(text: str, max_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    remaining = text

    while len(remaining) > max_chars:
        split_at = max_chars
        window = remaining[:max_chars]
        for marker in ("，", ",", "；", ";", " "):
            position = window.rfind(marker)
            if position >= max_chars // 2:
                split_at = position + 1
                break
        chunks.append(Chunk(text=remaining[:split_at], rewritable=True))
        remaining = remaining[split_at:]

    if remaining:
        chunks.append(Chunk(text=remaining, rewritable=True))
    return chunks


def _starts_structural_block(line: str) -> bool:
    stripped = line.strip()
    return any(
        (
            _is_placeholder_only(stripped),
            _is_heading_line(stripped),
            _is_list_line(stripped),
            _is_quote_line(stripped),
            _is_table_line(stripped),
            _is_link_def(stripped),
        )
    )


def _is_placeholder_only(stripped: str) -> bool:
    return bool(_PLACEHOLDER_BLOCK_RE.fullmatch(stripped))


def _is_heading_line(stripped: str) -> bool:
    return bool(_HEADING_RE.match(stripped))


def _is_list_line(stripped: str) -> bool:
    return bool(_LIST_RE.match(stripped))


def _is_quote_line(stripped: str) -> bool:
    return stripped.startswith(">")


def _is_table_line(stripped: str) -> bool:
    if stripped.startswith("|") and stripped.count("|") >= 2:
        return True
    if "|" in stripped and re.fullmatch(r"[:\-\|\s]+", stripped):
        return True
    return False


def _is_link_def(stripped: str) -> bool:
    return bool(_LINK_DEF_RE.match(stripped))


def _is_frozen_paragraph(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if _is_placeholder_only(stripped):
        return True
    if stripped.endswith(("：", ":")) and not re.search(r"[。！？!?；;]", stripped):
        return True
    if _is_caption_block(stripped):
        return True
    if _is_english_dominant_block(stripped):
        return True
    return False


def _is_caption_block(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    return all(_CAPTION_RE.fullmatch(line) for line in lines)


def _is_english_dominant_block(text: str) -> bool:
    if _CJK_RE.search(text):
        return False
    english_words = _ENGLISH_WORD_RE.findall(text)
    if len(english_words) < 4:
        return False
    ascii_letters = sum(1 for char in text if char.isascii() and char.isalpha())
    visible = sum(1 for char in text if not char.isspace())
    if visible == 0:
        return False
    return ascii_letters / visible >= 0.45
