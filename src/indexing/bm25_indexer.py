from ..chunking import ChunkerFactory
from ..interfaces import BaseIndexer
from ..models import Chunk
from .tokenization import tokenize
from rank_bm25 import BM25, BM25Okapi
from typing import Optional
from pathlib import Path
from tqdm import tqdm
import pickle


class BM25Indexer(BaseIndexer):
    """Build a persisted BM25 index from supported source files."""

    def __init__(self) -> None:
        """Initialize empty chunk and BM25 collections."""
        self.chunks: list[Chunk] = []
        self.bm25: Optional[BM25] = None

    def index(self, corpus_path: str, max_chunk_size: int) -> None:
        """Chunk the corpus and build a normalized BM25Okapi index."""
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")

        self.chunks = []
        for file in tqdm(Path(corpus_path).rglob("*")):
            if not file.is_file():
                continue
            if not file.name.lower().endswith((".py", ".md", ".txt", ".text")):
                continue
            chunker = ChunkerFactory.route(str(file))
            if chunker:
                with open(file, "r", encoding="utf-8") as f:
                    text = f.read()
                self.chunks.extend(
                    chunker.chunk(text, str(file), max_chunk_size)
                )

        tokenized_corpus = [
            tokenize(f"{chunk.content} {chunk.file_path}")
            for chunk in self.chunks
        ]

        if not tokenized_corpus:
            raise ValueError(f"No chunks found in {corpus_path}. "
                             f"Does the directory exist and contain "
                             f".py/.md files?")

        self.bm25 = BM25Okapi(tokenized_corpus, k1=0.75, b=0.75)

    def save(self, output_path: str) -> None:
        """Serialize chunks and the BM25 index to ``output_path``."""
        with open(output_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, f)
