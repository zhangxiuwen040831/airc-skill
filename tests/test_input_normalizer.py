from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from airc_skill.cli import main
from airc_skill.input_normalizer import InputNormalizationError, normalize_to_markdown
from airc_skill.pipeline import run_file


def _body_text() -> str:
    return (
        "近年来，数字平台正在持续进入课堂评价场景。与此同时，教师需要根据系统反馈调整任务安排，"
        "因此有必要重新评估人工校对流程。\n\n"
        "本研究的主题为课堂反馈机制的优化路径。本研究不仅包含多轮写作任务设计，还完成了反馈链条的流程梳理。"
        "因此，本研究进一步讨论系统输出与人工评价之间的协同方式。\n\n"
        "在实际教学场景中，反馈周期、评价标准和任务复杂度都会影响教师的判断压力。"
        "与此同时，平台能力的提升也改变了原有的工作分配方式，因此需要形成更稳定的人工复核机制。"
    )


def _fake_pandoc_run(command, capture_output, text, check):
    output_path = Path(command[-1])
    output_path.write_text(_body_text(), encoding="utf-8")
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_md_input_run_normalizes_without_converter(tmp_path: Path) -> None:
    source = tmp_path / "paper.md"
    source.write_text(_body_text(), encoding="utf-8")

    result = run_file(source, preset="academic_natural", dry_run=True)

    assert result.input_normalization.original_type == ".md"
    assert result.input_normalization.converter_used == "none"
    assert result.input_normalization.normalization_success is True


def test_txt_input_run_wraps_as_markdown(tmp_path: Path) -> None:
    source = tmp_path / "paper.txt"
    source.write_text(_body_text(), encoding="utf-8")

    result = run_file(source, preset="academic_natural", dry_run=True)

    assert result.input_normalization.original_type == ".txt"
    assert result.input_normalization.normalized_type == ".md"
    assert result.input_normalization.converter_used == "plain_text_wrapper"
    assert result.output_path.suffix == ".md"


def test_docx_input_converts_with_pandoc_when_available(tmp_path: Path) -> None:
    source = tmp_path / "paper.docx"
    source.write_bytes(b"fake-docx")

    with patch("airc_skill.input_normalizer.shutil.which", return_value="pandoc"):
        with patch("airc_skill.input_normalizer.subprocess.run", side_effect=_fake_pandoc_run):
            report = normalize_to_markdown(source, keep_intermediate=True)

    assert report.original_type == ".docx"
    assert report.converter_used == "pandoc"
    assert report.pandoc_used is True
    assert report.normalized_path is not None
    assert Path(report.normalized_path).read_text(encoding="utf-8") == _body_text()


def test_pandoc_missing_returns_clear_error(tmp_path: Path) -> None:
    source = tmp_path / "paper.docx"
    source.write_bytes(b"fake-docx")

    with patch("airc_skill.input_normalizer.shutil.which", return_value=None):
        with pytest.raises(InputNormalizationError) as exc_info:
            normalize_to_markdown(source)

    message = str(exc_info.value)
    assert ".docx" in message
    assert "pandoc is missing" in message
    assert "Install pandoc" in message
    assert exc_info.value.report.normalization_success is False


def test_doc_input_failure_suggests_converting_to_docx(tmp_path: Path) -> None:
    source = tmp_path / "legacy.doc"
    source.write_bytes(b"fake-doc")

    def fail_run(command, capture_output, text, check):
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="legacy format rejected")

    with patch("airc_skill.input_normalizer.shutil.which", return_value="pandoc"):
        with patch("airc_skill.input_normalizer.subprocess.run", side_effect=fail_run):
            with pytest.raises(InputNormalizationError) as exc_info:
                normalize_to_markdown(source)

    message = str(exc_info.value)
    assert "doc support is experimental" in message
    assert "please convert to docx first" in message
    assert exc_info.value.report.original_type == ".doc"


def test_cli_run_docx_report_contains_normalization(tmp_path: Path) -> None:
    source = tmp_path / "paper.docx"
    output = tmp_path / "paper.airc.md"
    report = tmp_path / "paper.airc.report.json"
    source.write_bytes(b"fake-docx")

    with patch("airc_skill.input_normalizer.shutil.which", return_value="pandoc"):
        with patch("airc_skill.input_normalizer.subprocess.run", side_effect=_fake_pandoc_run):
            exit_code = main(
                [
                    "run",
                    str(source),
                    "--preset",
                    "academic_natural",
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                ]
            )

    assert exit_code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["input_normalization"]["original_type"] == ".docx"
    assert payload["input_normalization"]["converter_used"] == "pandoc"
    assert payload["input_normalization"]["normalization_success"] is True
    assert payload["output_schema"]["input_normalization"]["pandoc_used"] is True
