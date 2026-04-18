from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import DEFAULT_CONFIG
from .validator import ValidationError, validate_input_file

SUPPORTED_INPUT_TYPES = (".md", ".txt", ".docx", ".doc")


@dataclass(frozen=True)
class InputNormalizationReport:
    original_path: str
    original_type: str
    normalized_path: str | None
    normalized_type: str
    converter_used: str
    pandoc_used: bool
    pandoc_available: bool | None
    normalization_success: bool
    normalization_error: str | None
    keep_intermediate: bool
    temporary_path: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class InputNormalizationError(ValueError):
    """Raised when an input file cannot be normalized to markdown."""

    def __init__(self, message: str, report: InputNormalizationReport) -> None:
        super().__init__(message)
        self.report = report


def detect_input_type(path: str | Path) -> str:
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists():
        raise ValidationError(f"Input file does not exist: {candidate}")
    if not candidate.is_file():
        raise ValidationError(f"Input path is not a file: {candidate}")
    suffix = candidate.suffix.lower()
    if suffix not in SUPPORTED_INPUT_TYPES:
        raise ValidationError(
            f"Unsupported file type for public run: {suffix}. "
            f"Supported types are: {', '.join(SUPPORTED_INPUT_TYPES)}"
        )
    size_bytes = candidate.stat().st_size
    if size_bytes > DEFAULT_CONFIG.max_file_size_bytes:
        raise ValidationError(
            f"File is too large: {size_bytes} bytes. Limit is {DEFAULT_CONFIG.max_file_size_bytes} bytes."
        )
    return suffix


def check_pandoc_available() -> str:
    executable = shutil.which("pandoc")
    if executable is None:
        raise FileNotFoundError(
            "pandoc is not installed or not on PATH. Install pandoc, or convert the file to .md/.docx first."
        )
    return executable


def convert_with_pandoc(input_path: str | Path, output_path: str | Path) -> Path:
    source = Path(input_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    executable = check_pandoc_available()
    command = [executable, str(source), "-t", "gfm", "-o", str(target)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "pandoc returned a non-zero exit code.").strip()
        raise RuntimeError(details)
    return target


def normalize_to_markdown(
    path: str | Path,
    *,
    keep_intermediate: bool = False,
) -> InputNormalizationReport:
    source = Path(path).expanduser().resolve()
    input_type = detect_input_type(source)
    if input_type == ".md":
        return InputNormalizationReport(
            original_path=str(source),
            original_type=input_type,
            normalized_path=str(source),
            normalized_type=".md",
            converter_used="none",
            pandoc_used=False,
            pandoc_available=None,
            normalization_success=True,
            normalization_error=None,
            keep_intermediate=True,
            temporary_path=False,
        )

    if input_type == ".txt":
        return _normalize_txt(source, keep_intermediate=keep_intermediate)

    return _normalize_doc_with_pandoc(source, input_type=input_type, keep_intermediate=keep_intermediate)


def normalize_text_input(path: str | Path) -> tuple[str, InputNormalizationReport]:
    report = normalize_to_markdown(path, keep_intermediate=False)
    if report.normalized_path is None:
        raise InputNormalizationError("Input normalization failed before markdown text could be read.", report)
    text = Path(report.normalized_path).read_text(encoding="utf-8")
    return text, report


def cleanup_normalized_file(report: InputNormalizationReport) -> None:
    if not report.temporary_path or not report.normalized_path:
        return
    try:
        Path(report.normalized_path).unlink(missing_ok=True)
    except OSError:
        # Cleanup is best effort; the report still contains the temporary path for debugging.
        return


def _normalize_txt(source: Path, *, keep_intermediate: bool) -> InputNormalizationReport:
    validated = validate_input_file(source, max_size_bytes=DEFAULT_CONFIG.max_file_size_bytes)
    raw_text = validated.path.read_text(encoding=validated.encoding)
    markdown_text = raw_text.strip()
    if markdown_text:
        markdown_text = f"{markdown_text}\n"
    target, temporary = _normalized_target(source, keep_intermediate=keep_intermediate)
    target.write_text(markdown_text, encoding="utf-8")
    return InputNormalizationReport(
        original_path=str(source),
        original_type=".txt",
        normalized_path=str(target),
        normalized_type=".md",
        converter_used="plain_text_wrapper",
        pandoc_used=False,
        pandoc_available=None,
        normalization_success=True,
        normalization_error=None,
        keep_intermediate=keep_intermediate,
        temporary_path=temporary,
    )


def _normalize_doc_with_pandoc(
    source: Path,
    *,
    input_type: str,
    keep_intermediate: bool,
) -> InputNormalizationReport:
    target, temporary = _normalized_target(source, keep_intermediate=keep_intermediate)
    pandoc_available: bool | None = None
    try:
        check_pandoc_available()
        pandoc_available = True
        convert_with_pandoc(source, target)
    except FileNotFoundError as exc:
        pandoc_available = False
        message = _normalization_error_message(
            input_type=input_type,
            pandoc_missing=True,
            detail=str(exc),
        )
        report = _failed_report(source, input_type, message, pandoc_available=pandoc_available)
        raise InputNormalizationError(message, report) from exc
    except Exception as exc:
        message = _normalization_error_message(
            input_type=input_type,
            pandoc_missing=False,
            detail=str(exc),
        )
        report = _failed_report(source, input_type, message, pandoc_available=pandoc_available)
        raise InputNormalizationError(message, report) from exc

    return InputNormalizationReport(
        original_path=str(source),
        original_type=input_type,
        normalized_path=str(target),
        normalized_type=".md",
        converter_used="pandoc",
        pandoc_used=True,
        pandoc_available=True,
        normalization_success=True,
        normalization_error=None,
        keep_intermediate=keep_intermediate,
        temporary_path=temporary,
    )


def _normalized_target(source: Path, *, keep_intermediate: bool) -> tuple[Path, bool]:
    if keep_intermediate:
        return source.with_name(f"{source.stem}.airc.normalized.md"), False
    handle = tempfile.NamedTemporaryFile(prefix=f"{source.stem}.airc.", suffix=".md", delete=False)
    handle.close()
    return Path(handle.name).resolve(), True


def _failed_report(
    source: Path,
    input_type: str,
    message: str,
    pandoc_available: bool | None,
) -> InputNormalizationReport:
    return InputNormalizationReport(
        original_path=str(source),
        original_type=input_type,
        normalized_path=None,
        normalized_type=".md",
        converter_used="pandoc",
        pandoc_used=False,
        pandoc_available=pandoc_available,
        normalization_success=False,
        normalization_error=message,
        keep_intermediate=False,
        temporary_path=False,
    )


def _normalization_error_message(
    *,
    input_type: str,
    pandoc_missing: bool,
    detail: str,
) -> str:
    if pandoc_missing:
        return (
            f"Input normalization failed for {input_type}: pandoc is missing. "
            "Install pandoc and ensure it is on PATH, or convert the file to .md/.docx first. "
            f"Detail: {detail}"
        )
    if input_type == ".doc":
        return (
            "Input normalization failed for .doc: doc support is experimental, please convert to docx first. "
            f"Pandoc was available but conversion failed. Detail: {detail}"
        )
    return (
        f"Input normalization failed for {input_type}: pandoc conversion failed. "
        "Install/update pandoc or convert the file to .md/.docx first. "
        f"Detail: {detail}"
    )
