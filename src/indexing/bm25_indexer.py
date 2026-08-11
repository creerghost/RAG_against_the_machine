from ..chunking import ChunkerFactory
from ..interfaces import BaseIndexer
from ..models import Chunk
from rank_bm25 import BM25, BM25Okapi
from typing import Optional
from pathlib import Path
from tqdm import tqdm
import pickle


class BM25Indexer(BaseIndexer):
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.bm25: Optional[BM25] = None

    def index(self, corpus_path: str, max_chunk_size: int) -> None:
        for file in tqdm(Path(corpus_path).rglob("*")):
            if not (file.name.endswith(".py") or file.name.endswith(".md")):
                continue
            chunker = ChunkerFactory.route(str(file))
            if chunker:
                self.chunks.extend(
                    chunker.chunk(str(file), max_chunk_size)
                )

        tokenized_corpus = [chunk.content.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save(self, output_path: str) -> None:
        with open(output_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, f)
