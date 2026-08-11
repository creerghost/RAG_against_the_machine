from ..interfaces import BaseRetriever
from ..models import MinimalSource, Chunk
from rank_bm25 import BM25
from typing import Optional
import pickle
import numpy


class BM25Retriever(BaseRetriever):
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.bm25: Optional[BM25] = None

    def load(self, index_path: str) -> None:
        with open(index_path, "rb") as f:
            content = pickle.load(f)
            self.chunks = content["chunks"]
            self.bm25 = content["bm25"]

    def search(self, query: str, k: int) -> list[MinimalSource]:
        tokenized_query: list[str] = query.split()
        scores: list[float] = self.bm25.get_scores(tokenized_query)
        # argsort returns indices in ascending order, so we reverse it [::-1]
        # to get descending order (highest scores first) and slice the top k
        top_k_idxs = numpy.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k_idxs:
            chunk = self.chunks[idx]
            source = MinimalSource(
                file_path=chunk.file_path,
                first_char_idx=chunk.first_char_idx,
                last_char_idx=chunk.last_char_idx,
            )
            results.append(source)
        return results
