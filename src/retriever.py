"""Query the persisted index."""

from pathlib import Path

from .config import RagConfig
from .models import MinimalSource, ChunkTable
from .ranking import ranker_for
from .io_utils import load_model


class Retriever:
    """Loads the ranker and chunk table once and answers top-k queries."""
    def __init__(self, config: RagConfig) -> None:
        """Load the index from ``processed_dir``; FileNotFoundError if none."""
        processed_dir = Path(config.processed_dir)
        self.ranker = ranker_for(config.mode, config)
        table = load_model(processed_dir / "chunks.json", ChunkTable)
        # list index = chunk id in every ranker
        self.chunks = table.chunks

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return up to ``k`` sources for ``query``, best first."""
        return [self.chunks[i].to_source()
                for i, _ in self.ranker.rank(query, k)]
