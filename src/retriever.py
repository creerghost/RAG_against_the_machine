"""Query the persisted index."""

from pathlib import Path

from .config import RagConfig
from .models import MinimalSource, ChunkTable
from .bm25 import BM25Index
from .io_utils import load_model
from .tokenizer import Tokenizer


class Retriever:
    """Loads the index once and answers top-k queries."""
    def __init__(self, config: RagConfig) -> None:
        """Load the index from ``processed_dir``; FileNotFoundError if none."""
        self.processed_dir = Path(config.processed_dir)
        self.bm25 = BM25Index.load(self.processed_dir / "bm25.npz")
        table = load_model(self.processed_dir / "chunks.json", ChunkTable)
        self.chunks = table.chunks  # index = BM25 chunk id
        self.tokenizer = Tokenizer(config.stopwords, config.min_token_length)

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return up to ``k`` sources for ``query``, best first."""
        sources: list[MinimalSource] = []
        tokens = self.tokenizer.tokenize(query)
        for chunk_id, _ in self.bm25.top_k(tokens, k):
            sources.append(self.chunks[chunk_id].to_source())
        return sources
