"""Chunking strategies and per-extension selection."""

from pathlib import PurePath
from typing import Optional

from ..config import RagConfig
from .base import Chunker, TextChunker
from .markdown_chunker import MarkdownChunker


def chunker_for(path: str, config: RagConfig) -> Optional[Chunker]:
    """Return the chunker ``config`` maps ``path``'s extension to, or None."""
    kind = config.chunker_by_extension.get(PurePath(path).suffix.lower())
    by_kind = {c.kind: c for c in (MarkdownChunker, TextChunker)}
    cls = by_kind.get(kind) if kind else None
    return cls(config.max_chunk_size, config.separators) if cls else None


__all__ = ["Chunker", "MarkdownChunker", "TextChunker", "chunker_for"]
