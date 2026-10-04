"""Tests for the BM25 index."""

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.bm25 import BM25Index

DOCS = [["lora", "adapter", "load", "lora"], ["load", "model"],
        ["lora", "config"]]


def index() -> BM25Index:
    """Return the three-chunk example index."""
    return BM25Index.build([Counter(d) for d in DOCS], k1=1.5, b=0.75)


def reference(query: list[str]) -> list[float]:
    """Score ``DOCS`` with a plain-Python BM25 for comparison."""
    n, avgdl = len(DOCS), sum(map(len, DOCS)) / len(DOCS)
    out = []
    for doc in DOCS:
        score = 0.0
        for t in dict.fromkeys(query):
            df = sum(t in d for d in DOCS)
            if df:
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                tf = doc.count(t)
                score += idf * tf * 2.5 / (
                    tf + 1.5 * (0.25 + 0.75 * len(doc) / avgdl))
        out.append(score)
    return out


def test_csr_layout() -> None:
    """Postings match the worked example."""
    idx = index()
    assert idx.vocab == {"adapter": 0, "config": 1, "load": 2, "lora": 3,
                         "model": 4}
    assert idx.indptr.tolist() == [0, 1, 2, 4, 6, 7]
    assert idx.chunk_ids.tolist() == [0, 2, 0, 1, 0, 2, 1]
    assert idx.tfs.tolist() == [1, 1, 1, 1, 2, 1, 1]


def test_scores_match_formula() -> None:
    """Vectorised scores equal the plain formula; idf stays positive."""
    query = ["lora", "config", "load", "load"]
    assert np.allclose(index().scores(query), reference(query))
    assert (index().idf > 0).all()


def test_top_k() -> None:
    """Best first, only positive scores, k clamped, bad k gives []."""
    idx = index()
    assert [c for c, _ in idx.top_k(["lora", "config"], 10)] == [2, 0]
    assert len(idx.top_k(["lora", "config"], 1)) == 1
    assert idx.top_k(["unknown"], 5) == []
    assert idx.top_k([], 5) == []
    assert idx.top_k(["load"], 0) == []
    chunk_id, score = idx.top_k(["lora"], 1)[0]
    assert type(chunk_id) is int and type(score) is float


def test_empty_corpus() -> None:
    """An empty corpus builds and answers nothing."""
    assert BM25Index.build([], 1.5, 0.75).top_k(["a"], 3) == []


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """A loaded index is identical, also from a nested new directory."""
    idx = index()
    path = tmp_path / "a" / "b" / "bm25.npz"
    idx.save(path)
    loaded = BM25Index.load(path)
    assert loaded.vocab == idx.vocab
    assert all(type(t) is str for t in loaded.vocab)
    assert (loaded.k1, loaded.b) == (1.5, 0.75)
    query = ["lora", "config"]
    assert loaded.top_k(query, 3) == idx.top_k(query, 3)


def test_load_missing(tmp_path: Path) -> None:
    """Loading a missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        BM25Index.load(tmp_path / "missing.npz")
