"""Turn a ranker's chunk ids into sources."""

from src.models import Chunk, MinimalSource
from .ranking import Ranker


class Retriever:
    """Answers top-k queries with a ranker and the chunk table it ranks."""
    def __init__(self, ranker: Ranker, chunks: list[Chunk]) -> None:
        """Store ``ranker`` and ``chunks``; list index = chunk id."""
        self.ranker = ranker
        self.chunks = chunks

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return up to ``k`` sources for ``query``, best first."""
        return [self.chunks[i].to_source()
                for i, _ in self.ranker.rank(query, k)]
