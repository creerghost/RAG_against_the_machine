"""Build the on-disk index: load files, chunk them, build BM25 (and,
optionally, chunk embeddings)."""

import sys
from collections import Counter
from pathlib import Path, PurePath

import numpy as np

from src.config import (BM25Config, ChunkingConfig, EmbeddingConfig,
                        PathsConfig, TokenizerConfig)
from .loader import CorpusLoader
from src.retrieval import BM25Index, Tokenizer
from .chunking import chunkers_by_extension
from src.models import Chunk, ChunkTable
from src.io_utils import save_model


def build_index(paths_cfg: PathsConfig, chunking_cfg: ChunkingConfig,
                tokenizer_cfg: TokenizerConfig, bm25_cfg: BM25Config,
                embedding_cfg: EmbeddingConfig, semantic: bool = False) -> int:
    """Chunk every file under ``paths_cfg.raw_dir``, persist the index.

    With ``semantic`` also embed every chunk into ``embeddings.npy``. Returns
    the number of chunks; stored paths keep the ``raw_dir`` prefix.
    """
    loader = CorpusLoader(paths_cfg.raw_dir,
                          chunking_cfg.chunker_by_extension.keys())
    chunks: list[Chunk] = []
    counters: list[Counter[str]] = []
    texts: list[str] = []  # chunk texts, kept only for the embedder
    tokenizer = Tokenizer(tokenizer_cfg.stopwords,
                          tokenizer_cfg.min_token_length)
    chunkers = chunkers_by_extension(chunking_cfg)
    for path, text in loader.load():
        chunker = chunkers[PurePath(path).suffix.lower()]
        for chunk in chunker.chunk(path, text):
            chunk_text = text[
                chunk.first_character_index:chunk.last_character_index]
            tokens = tokenizer.tokenize(chunk_text)
            tokens += tokenizer.tokenize(chunk.title or "")
            rel_path = chunk.file_path.removeprefix(paths_cfg.raw_dir)
            tokens += tokenizer.tokenize(rel_path)
            chunks.append(chunk)
            counters.append(Counter(sys.intern(t) for t in tokens))
            if semantic:
                title = chunk.title or ""
                texts.append(f"{rel_path} {title}\n{chunk_text}")

    bm25 = BM25Index.build(counters, bm25_cfg.k1, bm25_cfg.b)
    out_path = Path(paths_cfg.processed_dir)
    bm25.save(out_path / "bm25.npz")
    save_model(ChunkTable(chunks=chunks), out_path / "chunks.json")
    # old vectors would not match the new chunk ids: drop them first, so an
    # interrupted embedding run cannot leave a stale file behind.
    embeddings_path = out_path / "embeddings.npy"
    embeddings_path.unlink(missing_ok=True)
    if semantic:
        # heavy: torch, only with --semantic
        from src.retrieval.embedder import Embedder
        embedder = Embedder(embedding_cfg.model, embedding_cfg.max_length,
                            embedding_cfg.batch_size, embedding_cfg.dtype)
        np.save(embeddings_path, embedder.embed(texts, desc="Embedding"))
    return len(chunks)
