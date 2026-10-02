"""Tests for the chunking package."""

from pathlib import Path
from typing import Optional

import pytest

from src.chunking import MarkdownChunker, TextChunker, chunker_for
from src.config import RagConfig
from src.models import Chunk

LORA = Path("data/raw/vllm-0.10.1/docs/features/lora.md")
FENCE = "`" * 3
SEPS = RagConfig().separators


def md(size: int = 2000) -> MarkdownChunker:
    """Return a MarkdownChunker with the default separators."""
    return MarkdownChunker(size, SEPS)


def txt(size: int = 2000) -> TextChunker:
    """Return a TextChunker with the default separators."""
    return TextChunker(size, SEPS)


def spans(chunks: list[Chunk]) -> list[tuple[int, int, Optional[str]]]:
    """Return ``(start, end, title)`` tuples for easy comparison."""
    return [(c.first_character_index, c.last_character_index, c.title)
            for c in chunks]


def assert_valid(chunks: list[Chunk], text: str, max_size: int) -> None:
    """Chunks cover ``text`` contiguously, none empty or oversized."""
    s = spans(chunks)
    assert s[0][0] == 0 and s[-1][1] == len(text)
    assert all(a[1] == b[0] for a, b in zip(s, s[1:]))
    assert all(0 < end - start <= max_size for start, end, _ in s)


class TestMarkdownHeadings:
    """Section detection."""
    def test_no_headings(self) -> None:
        """A file without headings is one untitled section."""
        assert md()._sections("plain text") == [(0, 10, "")]

    def test_intro_before_first_heading(self) -> None:
        """Text before the first heading is kept, untitled."""
        assert md()._sections("intro\n# A\nbody") == [(0, 6, ""), (6, 14, "A")]

    def test_hash_without_space_is_not_heading(self) -> None:
        """``#comment`` and shebangs are not headings."""
        text = "#comment\n#!/bin/bash\n"
        assert md().find_headings(text) == []

    def test_heading_inside_fence_ignored(self) -> None:
        """``#`` lines inside code fences are not headings."""
        text = f"# A\n{FENCE}py\n# not\n{FENCE}\n# B\nx"
        assert md().find_headings(text) == [(0, "A"), (20, "B")]

    def test_indented_fence(self) -> None:
        """Indented fences toggle code mode too."""
        text = f"    {FENCE}\n# inside\n    {FENCE}\n# Real"
        assert md().find_headings(text) == [(25, "Real")]

    def test_crlf_offsets(self) -> None:
        """Offsets count ``\\r\\n`` correctly."""
        assert md().find_headings("a\r\n# W\r\nb") == [(3, "W")]


class TestSplit:
    """Size enforcement shared by all chunkers."""
    @pytest.mark.parametrize("max_size", [1, 7, 50, 300, 2000])
    def test_limits(self, max_size: int) -> None:
        """Any limit gives valid contiguous chunks."""
        text = "para one\n\n" * 40 + "x" * 900 + "\nline\n" * 30
        assert_valid(txt(max_size).chunk("f", text), text, max_size)

    def test_hard_cut(self) -> None:
        """A single long line is cut into fixed windows."""
        chunks = txt().chunk("f", "x" * 4500)
        assert spans(chunks) == [(0, 2000, None), (2000, 4000, None),
                                 (4000, 4500, None)]

    def test_trailing_separator(self) -> None:
        """Text ending with a separator yields no empty chunk."""
        text = "a" * 8 + "\n\n" + "b" * 8 + "\n\n"
        assert_valid(txt(10).chunk("f", text), text, 10)

    def test_empty_file(self) -> None:
        """An empty file has no chunks."""
        assert txt().chunk("f", "") == []

    def test_invalid_size(self) -> None:
        """max_chunk_size must be positive."""
        with pytest.raises(ValueError):
            txt(0)


def test_chunker_for() -> None:
    """Extension picks the strategy; unknown extensions are skipped."""
    config = RagConfig()
    assert isinstance(chunker_for("a/b.md", config), MarkdownChunker)
    assert isinstance(chunker_for("a/b.PY", config), TextChunker)
    assert chunker_for("a/b.png", config) is None


@pytest.mark.skipif(not LORA.is_file(), reason="corpus not present")
class TestLora:
    """End-to-end on the real file used during design."""
    @pytest.mark.parametrize("max_size", [1, 7, 50, 300, 2000])
    def test_valid(self, max_size: int) -> None:
        """Valid chunking at any size."""
        text = LORA.read_text(encoding="utf-8")
        chunks = md(max_size).chunk(str(LORA), text)
        assert_valid(chunks, text, max_size)

    def test_reference_section_is_one_chunk(self) -> None:
        """The section answering the LoRA endpoint question stays whole."""
        text = LORA.read_text(encoding="utf-8")
        chunks = md().chunk(str(LORA), text)
        assert (4695, 6100, "Using API Endpoints") in spans(chunks)
        assert all(c.file_path == str(LORA) for c in chunks)
        assert all(c.kind == "markdown" for c in chunks)
