"""Rankers that score chunks for a query, and the factory picking one."""

from pathlib import Path

import numpy as np

from .base import Ranker
from .hybrid_ranker import HybridRanker
from .lexical_ranker import LexicalRanker
from src.retrieval.bm25 import BM25Index
from src.config import (EmbeddingConfig, PathsConfig, RetrievalConfig,
                        TokenizerConfig)
from src.retrieval.tokenizer import Tokenizer


def ranker_for(paths_cfg: PathsConfig, tokenizer_cfg: TokenizerConfig,
               embedding_cfg: EmbeddingConfig,
               retrieval_cfg: RetrievalConfig) -> Ranker:
    """Build the ``retrieval_cfg.mode`` ranker from the index files."""
    mode = retrieval_cfg.mode
    if mode not in ("lexical", "semantic", "hybrid"):
        raise ValueError(f"unknown mode: {mode}")
    processed_dir = Path(paths_cfg.processed_dir)
    tokenizer = Tokenizer(tokenizer_cfg.stopwords,
                          tokenizer_cfg.min_token_length)
    bm25 = BM25Index.load(processed_dir / "bm25.npz")
    lexical = LexicalRanker(bm25, tokenizer)
    if mode == "lexical":
        return lexical
    # imported here: they pull in torch, which lexical mode never needs.
    from .semantic_ranker import SemanticRanker
    from src.retrieval.embedder import Embedder
    # loaded before the model so a missing file fails without the wait
    vectors = np.load(processed_dir / "embeddings.npy")
    # row i must be chunk i; a mismatch would return ids past the chunks
    if len(vectors) != bm25.n_chunks:
        raise ValueError(f"embeddings.npy has {len(vectors)} rows but the "
                         f"index has {bm25.n_chunks} chunks")
    embedder = Embedder(embedding_cfg.model, embedding_cfg.max_length,
                        embedding_cfg.batch_size, embedding_cfg.dtype)
    semantic = SemanticRanker(embedder, vectors)
    if mode == "semantic":
        return semantic
    return HybridRanker([lexical, semantic], list(retrieval_cfg.rrf_weights),
                        retrieval_cfg.rrf_k, retrieval_cfg.rrf_candidates)


__all__ = ["HybridRanker", "LexicalRanker", "Ranker", "ranker_for"]
