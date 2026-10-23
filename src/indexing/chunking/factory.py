"""Factory: one shared chunker per indexed file extension."""

from src.config import ChunkingConfig
from .base import Chunker, TextChunker
from .markdown_chunker import MarkdownChunker
from .python_chunker import PythonChunker


def chunkers_by_extension(config: ChunkingConfig) -> dict[str, Chunker]:
    """Return the chunker for each extension in ``config``, built once and
    shared; ``ValueError`` if a ``Chunker.kind`` there does not exist."""
    chunkers = [cls(config.max_chunk_size, config.separators)
                for cls in (MarkdownChunker, PythonChunker, TextChunker)]
    by_kind = {c.kind: c for c in chunkers}
    unknown = set(config.chunker_by_extension.values()) - by_kind.keys()
    if unknown:
        raise ValueError(f"unknown chunker kind(s) {sorted(unknown)}; "
                         f"known: {sorted(by_kind)}")
    return {ext.lower(): by_kind[kind]
            for ext, kind in config.chunker_by_extension.items()}
