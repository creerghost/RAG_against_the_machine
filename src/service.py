"""One entry point for indexing, search and answers, shared by the CLI and
the HTTP server; builds the retriever and the answer model on first use."""

from pathlib import Path
from typing import Any, Optional

from .config import RagConfig
from .indexing import build_index
from .io_utils import load_model
from .models import ChunkTable, MinimalSource
from .retrieval import Retriever, ranker_for


class RagService:
    """Wires the pipeline from one ``RagConfig``; heavy parts load lazily."""
    def __init__(self, config: RagConfig) -> None:
        """Keep ``config``; nothing is loaded until a method needs it."""
        self.config = config
        self._retriever: Optional[Retriever] = None
        self._generator: Any = None  # Generator, imported on first use

    def index(self, semantic: bool,
              max_chunk_size: Optional[int] = None) -> int:
        """Build the index (with embeddings if ``semantic``) and return the
        chunk count; ``max_chunk_size`` overrides the configured width."""
        cfg = self.config
        if max_chunk_size is not None:  # frozen: copy the group and the root
            cfg = cfg.model_copy(update={"chunking": cfg.chunking.model_copy(
                update={"max_chunk_size": max_chunk_size})})
        self._retriever = None  # an old retriever would serve stale ids
        return build_index(cfg.paths, cfg.chunking, cfg.tokenizer, cfg.bm25,
                           cfg.embedding, semantic)

    def load_index(self) -> Retriever:
        """Return the retriever, loading ranker and chunk table once;
        FileNotFoundError if the index is missing."""
        if self._retriever is None:
            cfg = self.config
            ranker = ranker_for(cfg.paths, cfg.tokenizer, cfg.embedding,
                                cfg.retrieval)
            path = Path(cfg.paths.processed_dir) / "chunks.json"
            chunks = load_model(path, ChunkTable).chunks
            self._retriever = Retriever(ranker, chunks)
        return self._retriever

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return up to ``k`` sources for ``query``, best first."""
        return self.load_index().search(query, k)

    def answer(self, question: str, sources: list[MinimalSource]) -> str:
        """Return an answer grounded in ``sources``; with none, the fixed
        reply, without loading the model."""
        if not sources:
            return self.config.generation.no_answer
        if self._generator is None:
            from .generator import Generator  # heavy: torch takes ~2 s
            self._generator = Generator(self.config.generation)
        return str(self._generator.answer(question, sources))
