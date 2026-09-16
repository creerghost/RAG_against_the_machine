import pytest

from src.chunking.factory import ChunkerFactory
from src.chunking.markdown_chunker import MarkdownChunker
from src.chunking.python_chunker import PythonChunker


def test_markdown_without_heading_is_indexed_and_bounded() -> None:
    text = "plain text " * 20

    chunks = MarkdownChunker().chunk(text, "notes.txt", 10)

    assert chunks
    assert all(0 < len(chunk.content) <= 10 for chunk in chunks)
    assert "".join(chunk.content for chunk in chunks) == text


def test_python_large_literal_is_hard_split() -> None:
    text = 'value = "' + ("x" * 100) + '"\n'

    chunks = PythonChunker().chunk(text, "large.py", 10)

    assert chunks
    assert all(0 < len(chunk.content) <= 10 for chunk in chunks)
    assert "".join(chunk.content for chunk in chunks) == text


def test_text_extension_is_routed() -> None:
    assert ChunkerFactory.route("guide.txt") is not None


def test_chunkers_reject_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        MarkdownChunker().chunk("# heading", "guide.md", 0)

    with pytest.raises(ValueError, match="positive"):
        PythonChunker().chunk("value = 1", "guide.py", -1)
