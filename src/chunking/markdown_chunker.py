import re

from ..interfaces import BaseChunker
from ..models import Chunk


class MarkdownChunker(BaseChunker):
    """Split Markdown and plain-text files while preserving source offsets."""

    def chunk(
        self, text: str, file_path: str, max_chunk_size: int
    ) -> list[Chunk]:
        """Split a document at headings, then at character boundaries."""
        self._validate_chunk_size(max_chunk_size)
        if not text:
            return []

        headings = list(re.finditer(r"(?m)^#+[ \t]+", text))
        sections: list[tuple[int, int]] = []
        if not headings:
            sections.append((0, len(text)))
        else:
            first_heading = headings[0].start()
            if first_heading > 0:
                sections.append((0, first_heading))

            for index, heading in enumerate(headings):
                end = (
                    headings[index + 1].start()
                    if index + 1 < len(headings)
                    else len(text)
                )
                sections.append((heading.start(), end))

        chunks: list[Chunk] = []
        for start, end in sections:
            chunks.extend(
                self._split_range(
                    text, start, end, file_path, max_chunk_size
                )
            )
        return chunks

    @staticmethod
    def _validate_chunk_size(max_chunk_size: int) -> None:
        """Reject a chunk size that cannot produce valid chunks."""
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")

    @staticmethod
    def _split_range(
        text: str,
        start: int,
        end: int,
        file_path: str,
        max_chunk_size: int,
    ) -> list[Chunk]:
        """Return contiguous non-empty chunks for a source range."""
        chunks: list[Chunk] = []
        current = start
        while current < end:
            chunk_end = min(current + max_chunk_size, end)
            chunks.append(
                Chunk(
                    file_path=file_path,
                    content=text[current:chunk_end],
                    first_character_index=current,
                    last_character_index=chunk_end,
                )
            )
            current = chunk_end
        return chunks
