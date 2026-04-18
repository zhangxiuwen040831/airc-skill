from __future__ import annotations

from difflib import unified_diff
from pathlib import Path

from .validator import ValidatedFile


class FileOperationError(OSError):
    """Raised when reading or writing a file fails."""


def read_text_file(validated: ValidatedFile) -> str:
    try:
        return validated.path.read_text(encoding=validated.encoding)
    except OSError as exc:
        raise FileOperationError(f"Failed to read file: {validated.path}") from exc


def write_text_file(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    target = Path(path)
    try:
        target.write_text(text, encoding=encoding)
    except OSError as exc:
        raise FileOperationError(f"Failed to write file: {target}") from exc
    return target


def build_output_path(source: str | Path) -> Path:
    source_path = Path(source)
    return source_path.with_name(f"{source_path.stem}.airc{source_path.suffix}")


def generate_diff(
    original: str,
    revised: str,
    source_name: str,
    revised_name: str,
) -> str:
    diff = unified_diff(
        original.splitlines(keepends=True),
        revised.splitlines(keepends=True),
        fromfile=source_name,
        tofile=revised_name,
    )
    return "".join(diff)
