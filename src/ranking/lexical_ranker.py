"""BM25 keyword ranking."""

from .base import Ranker
from ..bm25 import BM25Index
from ..tokenizer import Tokenizer


class LexicalRanker(Ranker):
    """Ranks chunks by BM25 over the query's tokens."""
    def __init__(self, bm25: BM25Index, tokenizer: Tokenizer) -> None:
        """Rank with ``bm25``, splitting queries with ``tokenizer``."""
        self.bm25 = bm25
        self.tokenizer = tokenizer

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` chunks with a positive BM25 score, best first."""
        tokens = self.tokenizer.tokenize(query)
        return self.bm25.top_k(tokens, k)
