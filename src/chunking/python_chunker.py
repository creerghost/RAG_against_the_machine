import ast

from ..interfaces import BaseChunker
from ..models import Chunk


class PythonChunker(BaseChunker):
    """Split Python files at AST nodes with a bounded character fallback."""

    def chunk(
        self, text: str, file_path: str, max_chunk_size: int
    ) -> list[Chunk]:
        """Split top-level Python nodes and preserve their source spans."""
        self._validate_chunk_size(max_chunk_size)
        parsed_text = ast.parse(text)
        chunks: list[Chunk] = []
        search_start = 0

        for node in parsed_text.body:
            search_start = self._process_node(
                node, text, file_path, max_chunk_size, search_start, chunks
            )

        if search_start < len(text):
            chunks.extend(
                self._split_segment(
                    text[search_start:], search_start, file_path, max_chunk_size
                )
            )

        return chunks

    @staticmethod
    def _validate_chunk_size(max_chunk_size: int) -> None:
        """Reject a chunk size that cannot produce valid chunks."""
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")

    def _process_node(
        self, node: ast.AST, text: str, file_path: str,
        max_size: int, search_start: int, chunks: list[Chunk]
    ) -> int:
        """Chunk one top-level AST node and return its source end offset."""
        segment = ast.get_source_segment(text, node)
        if not segment:
            return search_start

        start_idx = text.find(segment, search_start)
        if start_idx == -1:
            return search_start

        if start_idx > search_start:
            chunks.extend(
                self._split_segment(
                    text[search_start:start_idx],
                    search_start,
                    file_path,
                    max_size,
                )
            )

        last_character_index = start_idx + len(segment)
        chunks.extend(
            self._split_segment(
                segment, start_idx, file_path, max_size
            )
        )
        return last_character_index

    @staticmethod
    def _split_segment(
        segment: str,
        start: int,
        file_path: str,
        max_size: int,
    ) -> list[Chunk]:
        """Split one AST source segment without exceeding ``max_size``."""
        chunks: list[Chunk] = []
        current = 0
        while current < len(segment):
            chunk_end = min(current + max_size, len(segment))
            chunks.append(
                Chunk(
                    file_path=file_path,
                    content=segment[current:chunk_end],
                    first_character_index=start + current,
                    last_character_index=start + chunk_end,
                )
            )
            current = chunk_end
        return chunks
