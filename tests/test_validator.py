from pathlib import Path

import pytest

from airc_skill.validator import ValidationError, validate_input_file


def test_validator_rejects_unsupported_extension(tmp_path: Path) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_text("not supported", encoding="utf-8")

    with pytest.raises(ValidationError):
        validate_input_file(sample)


def test_validator_does_not_add_utf8_bom_to_plain_utf8_files(tmp_path: Path) -> None:
    sample = tmp_path / "sample.md"
    sample.write_text("## 标题\n\n正文内容。", encoding="utf-8")

    validated = validate_input_file(sample)

    assert validated.encoding == "utf-8"
