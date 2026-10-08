from .base import Ranker
from .lexical_ranker import LexicalRanker


class HybridRanker(Ranker):
    def __init__(self, rankers: list[Ranker], rrf_k: int) -> None:
        self.rankers = rankers

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        chunks: list[tuple[int, float]] = []
        tokens = self.tokenizer.tokenize(query)
