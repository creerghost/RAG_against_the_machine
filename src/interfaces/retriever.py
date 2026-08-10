from ..models import MinimalSource
from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    @abstractmethod
    def load(self, index_path: str) -> None:
        """Load a persisted index from disk."""
        pass

    @abstractmethod
    def search(
        self, query: str, k: int
    ) -> list[MinimalSource]:
        """Return top-k source locations for a query."""
        pass
