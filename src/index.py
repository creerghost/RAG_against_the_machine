"""Build the on-disk index: load files, chunk them, build BM25 (and,
optionally, chunk embeddings)."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .config import RagConfig
from .loader import CorpusLoader
from .bm25 import BM25Index
from .tokenizer import Tokenizer
from .chunking import chunker_for
from .models import Chunk, ChunkTable
from .io_utils import save_model


def build_index(config: RagConfig, semantic: bool = False) -> int:
    """Chunk every indexed file under ``config.raw_dir``, persist the index.

    With ``semantic`` also embed every chunk into ``embeddings.npy``. Returns
    the number of chunks; stored paths keep the ``raw_dir`` prefix.
    """
    loader = CorpusLoader(config.raw_dir, config.chunker_by_extension.keys())
    chunks: list[Chunk] = []
    counters: list[Counter[str]] = []
    texts: list[str] = []  # chunk texts, kept only for the embedder
    tokenizer = Tokenizer(config.stopwords, config.min_token_length)
    for path, text in loader.load():
        # this will create around 1800 tiny objects,
        # which are negligible to cache
        chunker = chunker_for(path, config)
        if chunker is None:
            continue  # never happens, just for mypy

        for chunk in chunker.chunk(path, text):
            chunk_text = text[
                chunk.first_character_index:chunk.last_character_index]
            tokens = tokenizer.tokenize(chunk_text)
            tokens += tokenizer.tokenize(chunk.title or "")
            rel_path = chunk.file_path.removeprefix(config.raw_dir)
            tokens += tokenizer.tokenize(rel_path)
            chunks.append(chunk)
            counters.append(Counter(sys.intern(t) for t in tokens))
            if semantic:
                texts.append(chunk_text)

    bm25 = BM25Index.build(counters, config.bm25_k1, config.bm25_b)
    out_path = Path(config.processed_dir)
    bm25.save(out_path / "bm25.npz")
    save_model(ChunkTable(chunks=chunks), out_path / "chunks.json")
    # old vectors would not match the new chunk ids: drop them first, so an
    # interrupted embedding run cannot leave a stale file behind.
    embeddings_path = out_path / "embeddings.npy"
    embeddings_path.unlink(missing_ok=True)
    if semantic:
        from .embedder import Embedder  # heavy: torch, only with --semantic
        embedder = Embedder(config.embedding_model,
                            config.embedding_max_length,
                            config.embedding_batch_size,
                            config.embedding_dtype)
        np.save(embeddings_path, embedder.embed(texts, desc="Embedding"))
    return len(chunks)
