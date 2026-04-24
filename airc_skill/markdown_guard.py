from __future__ import annotations

import re
from typing import Pattern


PLACEHOLDER_RE = re.compile(r"\[\[AIRC:[A-Z_]+:\d{4}\]\]")

_FENCED_CODE_RE = re.compile(
    r"(?ms)^(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^\s*(?P=fence)[ \t]*$",
    re.MULTILINE,
)
_MATH_BLOCK_RE = re.compile(r"(?ms)^\$\$.*?^\$\$[ \t]*$", re.MULTILINE)
_IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]+\)")
_LINK_RE = re.compile(r"(?<!!)\[[^\]]+]\([^)]+\)")
_LINK_DEF_RE = re.compile(r"(?m)^\[[^\]]+\]:\s+\S+.*$")
_AUTOLINK_RE = re.compile(r"<https?://[^>\n]+>")
_BARE_URL_RE = re.compile(r"https?://[^\s<>()]+(?:\([^\s<>()]*\)[^\s<>()]*)*")
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(?:\\.|[^$\n])+\$(?!\$)")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def protect(text: str) -> tuple[str, dict[str, str]]:
    """Protect Markdown fragments that should be restored verbatim."""

    placeholders: dict[str, str] = {}
    protected = text
    counter = 0

    for label, pattern in (
        ("CODE", _FENCED_CODE_RE),
        ("MATH_BLOCK", _MATH_BLOCK_RE),
        ("IMAGE", _IMAGE_RE),
        ("LINK", _LINK_RE),
        ("LINK_DEF", _LINK_DEF_RE),
        ("AUTOLINK", _AUTOLINK_RE),
        ("URL", _BARE_URL_RE),
        ("INLINE_MATH", _INLINE_MATH_RE),
        ("INLINE_CODE", _INLINE_CODE_RE),
    ):
        protected, counter = _protect_with_pattern(
            protected, pattern=pattern, label=label, placeholders=placeholders, counter=counter
        )

    return protected, placeholders


def restore(text: str, placeholders: dict[str, str]) -> str:
    restored = text
    for token, original in placeholders.items():
        restored = restored.replace(token, original)
    return restored


def _protect_with_pattern(
    text: str,
    pattern: Pattern[str],
    label: str,
    placeholders: dict[str, str],
    counter: int,
) -> tuple[str, int]:
    def _replace(match: re.Match[str]) -> str:
        nonlocal counter
        token = f"[[AIRC:{label}:{counter:04d}]]"
        placeholders[token] = match.group(0)
        counter += 1
        return token

    return pattern.sub(_replace, text), counter
