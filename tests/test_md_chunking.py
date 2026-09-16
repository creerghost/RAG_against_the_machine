# pyrefly: ignore [missing-import]
from src.chunking.markdown_chunker import MarkdownChunker
# pyrefly: ignore [missing-import]
from src.models import Chunk
import pytest


@pytest.fixture
def chunked_readme() -> tuple[str, list[Chunk]]:
    file_name = "tests/data_tests/test_readme.md"
    with open(file_name, "r") as f:
        text = f.read()
    chunker = MarkdownChunker()
    chunks = chunker.chunk(text, file_name, 2000)
    return text, chunks


def test_chunk_size_limit(chunked_readme: tuple[str, list[Chunk]]) -> None:
    _, chunks = chunked_readme
    for chunk in chunks:
        assert len(chunk.content) <= 2000


def test_chunk_indices_mapping(
    chunked_readme: tuple[str, list[Chunk]]
) -> None:
    text, chunks = chunked_readme
    for chunk in chunks:
        assert text[
            chunk.first_character_index:chunk.last_character_index
        ] == chunk.content


def test_chunk_expected_count(chunked_readme: tuple[str, list[Chunk]]) -> None:
    _, chunks = chunked_readme
    assert len(chunks) == 4


def test_chunk_content_extraction(
    chunked_readme: tuple[str, list[Chunk]]
) -> None:
    _, chunks = chunked_readme
    assert chunks[0].content.startswith("# Project Title")
