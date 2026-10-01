"""Query the persisted index."""

from pathlib import Path
from typing import List

from .models import MinimalSource


class Retriever:
    """Loads the index once and answers top-k queries."""

    def __init__(self, processed_dir: Path) -> None:
        """Load the index from ``processed_dir``.

        Args:
            processed_dir: Directory written by ``build_index``.

        Raises:
            FileNotFoundError: If the index has not been built.
        """
        self.processed_dir = processed_dir
        raise NotImplementedError("Retriever.__init__")

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """Return the ``k`` best sources for ``query``, best first.

        Args:
            query: Non-empty question text.
            k: Number of results, >= 1.

        Returns:
            List[MinimalSource]: At most ``k`` sources; empty if no query
            term is in the vocabulary.
        """
        raise NotImplementedError("Retriever.search")
