from .base import Ranker
from ..bm25 import BM25Index
from ..tokenizer import Tokenizer


class LexicalRanker(Ranker):
    def __init__(self, bm25: BM25Index, tokenizer: Tokenizer) -> None:
        self.bm25 = bm25
        self.tokenizer = tokenizer

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        tokens = self.tokenizer.tokenize(query)
        return self.bm25.top_k(tokens, k)
