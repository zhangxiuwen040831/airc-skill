from pathlib import Path

import pytest

from airc_skill.validator import ValidationError, validate_input_file


def test_validator_rejects_unsupported_extension(tmp_path: Path) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_text("not supported", encoding="utf-8")

    with pytest.raises(ValidationError):
        validate_input_file(sample)
