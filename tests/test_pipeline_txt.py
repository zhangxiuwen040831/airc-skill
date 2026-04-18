from pathlib import Path

from airc_skill.config import RewriteMode
from airc_skill.pipeline import rewrite_file


def test_txt_pipeline_rewrite_runs(tmp_path: Path) -> None:
    source = tmp_path / "report.txt"
    source.write_text(
        "总的来说，本研究在很多方面都具有重要意义。总的来说，本研究在很多方面都具有重要意义。\n\n"
        "此外，研究进行了分析，并且进行了分析。",
        encoding="utf-8",
    )

    result = rewrite_file(source, mode=RewriteMode.BALANCED)

    assert result.review.decision in {"pass", "pass_with_minor_risk", "reject"}
    assert result.output_path.name == "report.airc.txt"
    assert result.text.strip()


def test_dry_run_does_not_write_file(tmp_path: Path) -> None:
    source = tmp_path / "report.txt"
    source.write_text("总而言之，这一部分需要进一步完善。", encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.CONSERVATIVE, dry_run=True)

    assert result.output_written is False
    assert not result.output_path.exists()
    assert isinstance(result.diff, str)
