from ..interfaces import BaseRetriever
from ..models import MinimalSource, Chunk
from .tokenization import tokenize
from rank_bm25 import BM25
from typing import Optional
import pickle
import numpy


class BM25Retriever(BaseRetriever):
    """Load a BM25 index and return ranked minimal source locations."""

    def __init__(self) -> None:
        """Initialize an unloaded retriever."""
        self.chunks: list[Chunk] = []
        self.bm25: Optional[BM25] = None

    def load(self, index_path: str) -> None:
        """Load serialized chunks and BM25 scores from ``index_path``."""
        with open(index_path, "rb") as f:
            content = pickle.load(f)
            self.chunks = content["chunks"]
            self.bm25 = content["bm25"]

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return the top ``k`` source spans ranked for ``query``."""
        if self.bm25 is None:
            raise ValueError("An index must be loaded before searching")
        if k <= 0 or not query.strip():
            return []

        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        # argsort returns indices in ascending order, so we reverse it [::-1]
        # to get descending order (highest scores first) and slice the top k
        top_k_idxs = numpy.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k_idxs:
            chunk = self.chunks[idx]
            source = MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index,
            )
            results.append(source)
        return results
