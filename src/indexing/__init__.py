"""Offline part, run once by ``index``: load the corpus, chunk it, build the
BM25 index (and the embeddings)."""

from .chunking import chunkers_by_extension
from .index import build_index
from .loader import CorpusLoader

__all__ = ["CorpusLoader", "build_index", "chunkers_by_extension"]
