# pyrefly: ignore [missing-import]
from src.chunking.python_chunker import PythonChunker
# pyrefly: ignore [missing-import]
from src.models import Chunk
import pytest


@pytest.fixture
def chunked_dummy() -> tuple[str, list[Chunk]]:
    file_name = "tests/data_tests/test_dummy.py"
    with open(file_name, "r") as f:
        text = f.read()

    chunker = PythonChunker()
    chunks = chunker.chunk(text, file_name, max_chunk_size=200)
    return text, chunks


def test_chunk_size_limit(chunked_dummy: tuple[str, list[Chunk]]) -> None:
    _, chunks = chunked_dummy
    for chunk in chunks:
        assert len(chunk.content) <= 200


def test_chunk_indices_mapping(chunked_dummy: tuple[str, list[Chunk]]) -> None:
    text, chunks = chunked_dummy
    for chunk in chunks:
        assert text[chunk.first_char_idx:chunk.last_char_idx] == chunk.content


def test_chunk_expected_count(chunked_dummy: tuple[str, list[Chunk]]) -> None:
    _, chunks = chunked_dummy
    assert len(chunks) >= 3


def test_recursive_class_body_processing(
    chunked_dummy: tuple[str, list[Chunk]]
) -> None:
    _, chunks = chunked_dummy
    found_method = any("first_method" in chunk.content for chunk in chunks)
    assert found_method, \
        "Chunker failed to recursively process the class body!"
