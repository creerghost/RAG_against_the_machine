"""Markdown chunking by headings, ignoring ``#`` lines inside code fences."""

import re

from .base import Chunker, Span


class MarkdownChunker(Chunker):
    """One section per Markdown heading."""
    kind = "markdown"

    def _sections(self, text: str) -> list[Span]:
        """Return heading-to-heading sections covering ``text``."""
        return self.headings_to_chunks(text, self.find_headings(text))

    def find_headings(self, text: str) -> list[tuple[int, str]]:
        """Return ``(offset, title)`` for each heading outside code fences."""
        result: list[tuple[int, str]] = []
        in_code = False
        pos = 0
        for line in text.splitlines(keepends=True):
            if line.lstrip().startswith("```"):
                in_code = not in_code
            elif not in_code:
                match = re.match(r"(#+) (.*)", line)
                if match:
                    result.append((pos, match.group(2).strip()))
            pos += len(line)
        return result

    def headings_to_chunks(self, text: str,
                           headings: list[tuple[int, str]]) -> list[Span]:
        """Turn ``headings`` into contiguous sections covering ``text``.

        Text before the first heading (or a file without any) is untitled.
        """
        if not headings:
            return [(0, len(text), "")]
        result: list[Span] = []
        if headings[0][0] != 0:
            result.append((0, headings[0][0], ""))
        for (c_start, c_str), (n_start, _) in zip(headings, headings[1:]):
            result.append((c_start, n_start, c_str))
        result.append((headings[-1][0], len(text), headings[-1][1]))
        return result
