"""Chunker base class: shared size enforcement and ``Chunk`` building."""

from abc import ABC, abstractmethod

from ..models import Chunk

Span = tuple[int, int, str]
"""``(start, end, title)``: a character range ``text[start:end]``."""


class Chunker(ABC):
    """Split files into chunks of at most ``max_chunk_size`` characters.

    Subclasses only implement ``_sections``; size enforcement is shared.
    """
    kind: str = "text"

    def __init__(self, max_chunk_size: int,
                 separators: tuple[str, ...]) -> None:
        """Store the size limit and ``separators`` (best first) for splitting.

        Raises ``ValueError`` if ``max_chunk_size`` is below 1.
        """
        if max_chunk_size < 1:
            raise ValueError("max_chunk_size must be >= 1")
        self.max_chunk_size = max_chunk_size
        self.separators = separators

    def chunk(self, file_path: str, text: str) -> list[Chunk]:
        """Return contiguous chunks of ``text`` tagged with ``file_path``.

        Template method: ``_sections``, then ``_split``; empty text -> ``[]``.
        """
        if not text:
            return []
        spans: list[Span] = []
        for start, end, title in self._sections(text):
            spans.extend(self._split(text, start, end, title,
                                     self.separators, self.max_chunk_size))
        return [Chunk(file_path=file_path, first_character_index=start,
                      last_character_index=end, kind=self.kind,
                      title=title or None) for start, end, title in spans]

    @abstractmethod
    def _sections(self, text: str) -> list[Span]:
        """Return contiguous sections covering ``text``, of any size."""

    def _split(self, text: str, start: int, end: int, title: str,
               seps: tuple[str, ...], max_s: int) -> list[Span]:
        """Split ``text[start:end]`` into pieces of at most ``max_s`` chars.

        Packs parts by ``seps[0]``, recurses with the rest, hard-cuts last.
        """
        if end - start <= max_s:
            return [(start, end, title)]
        if not seps:
            return [(x, min(x + max_s, end), title)
                    for x in range(start, end, max_s)]
        sep, rest = seps[0], seps[1:]
        pieces: list[Span] = []
        parts = text[start:end].split(sep)
        last_pos = start
        pos = start
        for j, part in enumerate(parts):
            is_last = j == len(parts) - 1
            # split() removes separators, so add their length back
            part_end = pos + len(part) + (0 if is_last else len(sep))
            if part_end - last_pos > max_s and pos > last_pos:
                pieces.append((last_pos, pos, title))
                last_pos = pos
            pos = part_end
        if last_pos < end:  # text ending with sep leaves an empty last part
            pieces.append((last_pos, end, title))
        result: list[Span] = []
        for p in pieces:
            if p[1] - p[0] <= max_s:
                result.append(p)
            else:
                result.extend(self._split(text, p[0], p[1], p[2], rest,
                                          max_s))
        return result


class TextChunker(Chunker):
    """Whole file as one section, split only by size."""
    def _sections(self, text: str) -> list[Span]:
        """Return the whole file as a single untitled section."""
        return [(0, len(text), "")]
