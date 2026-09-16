import pytest
from pydantic import ValidationError

from src.models import MinimalSource


def test_minimal_source_uses_assignment_field_names() -> None:
    source = MinimalSource(
        file_path="data/raw/vllm-0.10.1/example.py",
        first_character_index=3,
        last_character_index=12,
    )

    assert source.model_dump() == {
        "file_path": "data/raw/vllm-0.10.1/example.py",
        "first_character_index": 3,
        "last_character_index": 12,
    }


def test_minimal_source_rejects_legacy_span_field_names() -> None:
    with pytest.raises(ValidationError):
        MinimalSource.model_validate(
            {
                "file_path": "data/raw/vllm-0.10.1/example.py",
                "first_char_idx": 3,
                "last_char_idx": 12,
            }
        )
