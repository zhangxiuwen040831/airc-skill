from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_CONFIG


class ValidationError(ValueError):
    """Raised when an input file does not satisfy runtime constraints."""


@dataclass(frozen=True)
class ValidatedFile:
    path: Path
    suffix: str
    encoding: str
    size_bytes: int


def validate_input_file(
    path: str | Path,
    max_size_bytes: int = DEFAULT_CONFIG.max_file_size_bytes,
) -> ValidatedFile:
    candidate = Path(path).expanduser().resolve()

    if not candidate.exists():
        raise ValidationError(f"Input file does not exist: {candidate}")
    if not candidate.is_file():
        raise ValidationError(f"Input path is not a file: {candidate}")
    if candidate.suffix.lower() not in DEFAULT_CONFIG.supported_suffixes:
        raise ValidationError(
            f"Unsupported file type: {candidate.suffix}. "
            f"Supported types are: {', '.join(DEFAULT_CONFIG.supported_suffixes)}"
        )

    size_bytes = candidate.stat().st_size
    if size_bytes > max_size_bytes:
        raise ValidationError(
            f"File is too large: {size_bytes} bytes. Limit is {max_size_bytes} bytes."
        )

    raw = candidate.read_bytes()
    encoding = _detect_encoding(raw)
    return ValidatedFile(
        path=candidate,
        suffix=candidate.suffix.lower(),
        encoding=encoding,
        size_bytes=size_bytes,
    )


def _detect_encoding(raw: bytes) -> str:
    for encoding in DEFAULT_CONFIG.candidate_encodings:
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    raise ValidationError("Unable to decode the file with the supported encodings.")
