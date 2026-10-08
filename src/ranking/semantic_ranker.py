"""Embedding (cosine similarity) ranking."""

from .base import Ranker
from ..embedder import Embedder
from ..top_k import top_k_ids, FloatArray


class SemanticRanker(Ranker):
    """Ranks chunks by cosine similarity between query and chunk vectors."""
    def __init__(self, embedder: Embedder, vectors: FloatArray) -> None:
        """Embed queries with ``embedder``; ``vectors`` row i is chunk i."""
        self.embedder = embedder
        self.vectors = vectors

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return the ``k`` chunks closest to ``query``, best first; rows
        are unit length, so the dot product is the cosine."""
        q = self.embedder.embed([query])[0]
        scores = self.vectors @ q
        best = top_k_ids(scores, k)
        return [(int(i), float(scores[i])) for i in best]
