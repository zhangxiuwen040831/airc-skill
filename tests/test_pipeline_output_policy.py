from pathlib import Path

from airc_skill.cli import main
from airc_skill.config import RewriteMode
from airc_skill.pipeline import rewrite_file


def test_no_write_when_rejected(tmp_path: Path) -> None:
    source = tmp_path / "minimal.txt"
    source.write_text("AIGC检测系统。", encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.BALANCED)

    assert result.output_written is False
    assert not result.output_path.exists()
    assert result.review.decision == "reject"


def test_cli_reports_reject_clearly(tmp_path: Path, capsys) -> None:
    source = tmp_path / "cli.txt"
    source.write_text("AIGC检测系统。", encoding="utf-8")

    exit_code = main(["rewrite", str(source), "--mode", "balanced"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Reviewer decision: reject" in output
    assert "Output written: no" in output
    assert "Output file was not written." in output
