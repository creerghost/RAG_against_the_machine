"""Query the persisted index."""

from pathlib import Path

from .models import MinimalSource


class Retriever:
    """Loads the index once and answers top-k queries."""
    def __init__(self, processed_dir: Path) -> None:
        """Load the index from ``processed_dir``; FileNotFoundError if none."""
        self.processed_dir = processed_dir
        raise NotImplementedError("Retriever.__init__")

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return up to ``k`` sources for ``query``, best first."""
        raise NotImplementedError("Retriever.search")
