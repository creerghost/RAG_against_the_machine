"""Offline part, run once by ``index``: load the corpus, chunk it, build the
BM25 index (and the embeddings)."""

from .chunking import chunker_for
from .index import build_index
from .loader import CorpusLoader

__all__ = ["CorpusLoader", "build_index", "chunker_for"]
