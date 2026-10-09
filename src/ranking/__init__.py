"""Rankers that score chunks for a query, and the factory picking one."""

from pathlib import Path

import numpy as np

from .base import Ranker
from .hybrid_ranker import HybridRanker
from .lexical_ranker import LexicalRanker
from ..bm25 import BM25Index
from ..config import RagConfig, RankMode
from ..tokenizer import Tokenizer


def ranker_for(mode: RankMode, config: RagConfig) -> Ranker:
    """Build the ranker for ``mode`` from the index files in processed_dir."""
    if mode not in ("lexical", "semantic", "hybrid"):
        raise ValueError(f"unknown mode: {mode}")
    processed_dir = Path(config.processed_dir)
    tokenizer = Tokenizer(config.stopwords, config.min_token_length)
    bm25 = BM25Index.load(processed_dir / "bm25.npz")
    lexical = LexicalRanker(bm25, tokenizer)
    if mode == "lexical":
        return lexical
    # imported here: they pull in torch, which lexical mode never needs.
    from .semantic_ranker import SemanticRanker
    from ..embedder import Embedder
    # loaded before the model so a missing file fails without the wait
    vectors = np.load(processed_dir / "embeddings.npy")
    # row i must be chunk i; a mismatch would return ids past the chunks
    if len(vectors) != bm25.n_chunks:
        raise ValueError(f"embeddings.npy has {len(vectors)} rows but the "
                         f"index has {bm25.n_chunks} chunks")
    embedder = Embedder(config.embedding_model, config.embedding_max_length,
                        config.embedding_batch_size, config.embedding_dtype)
    semantic = SemanticRanker(embedder, vectors)
    if mode == "semantic":
        return semantic
    return HybridRanker([lexical, semantic], list(config.rrf_weights),
                        config.rrf_k, config.rrf_candidates)


__all__ = ["HybridRanker", "LexicalRanker", "Ranker", "ranker_for"]
