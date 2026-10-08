"""The Ranker interface every retrieval strategy implements."""

from abc import ABC, abstractmethod


class Ranker(ABC):
    """Scores chunks for a query; chunk ids are rows of chunks.json."""
    @abstractmethod
    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(chunk_id, score)`` pairs for ``query``, best
        first."""
