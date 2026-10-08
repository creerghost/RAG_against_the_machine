from .base import Ranker
from ..bm25 import FloatArray
from ..embedder import Embedder


class SemanticRanker(Ranker):
    def __init__(self, embedder: Embedder, vectors: FloatArray):
        self.embedder = embedder
        self.vectors = vectors

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        pass