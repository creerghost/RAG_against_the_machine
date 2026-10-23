"""BM25 over an inverted index stored as CSR arrays (numpy only)."""

from collections import Counter
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .top_k import top_k_ids, FloatArray, IntArray64

IntArray32 = npt.NDArray[np.int32]


class BM25Index:
    """Raw term counts per chunk; idf and avgdl are derived, never stored.

    Postings of term ``t`` are ``chunk_ids[indptr[t]:indptr[t + 1]]`` with
    counts ``tfs[...]`` at the same positions.
    """
    def __init__(self, vocab: dict[str, int], indptr: IntArray64,
                 chunk_ids: IntArray32, tfs: IntArray32, doc_len: IntArray32,
                 k1: float, b: float) -> None:
        """Store the arrays and BM25 parameters; derive avgdl and idf."""
        self.vocab = vocab
        self.indptr = indptr
        self.chunk_ids = chunk_ids
        self.tfs = tfs
        self.doc_len = doc_len
        self.k1 = k1
        self.b = b
        self.n_chunks = len(doc_len)
        self.avgdl = float(doc_len.mean()) if self.n_chunks else 0.0
        df = np.diff(indptr)  # chunks containing each term
        self.idf: FloatArray = np.log(
            1 + (self.n_chunks - df + 0.5) / (df + 0.5))

    @classmethod
    def build(cls, counters: list[Counter[str]], k1: float,
              b: float) -> "BM25Index":
        """Index ``counters``, one term count per chunk; id = list index."""
        unique = {word for c in counters for word in c}
        vocab = {word: idx for idx, word in enumerate(sorted(unique))}
        total = sum(len(c) for c in counters)
        term_ids = np.empty(total, dtype=np.int64)
        chunk_ids = np.empty(total, dtype=np.int32)
        tfs = np.empty(total, dtype=np.int32)

        pos = 0
        for chunk_id, counter in enumerate(counters):
            n = len(counter)
            term_ids[pos:pos+n] = [vocab[word] for word in counter]
            tfs[pos:pos+n] = list(counter.values())
            chunk_ids[pos:pos+n] = chunk_id
            pos += n

        order = np.argsort(term_ids, kind="stable")
        tfs = tfs[order]
        chunk_ids = chunk_ids[order]
        lengths = np.bincount(term_ids, minlength=len(vocab))
        indptr = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        doc_len = np.array([sum(c.values()) for c in counters],
                           dtype=np.int32)
        return cls(vocab, indptr, chunk_ids, tfs, doc_len, k1, b)

    def scores(self, query_tokens: list[str]) -> FloatArray:
        """Return one BM25 score per chunk; repeated query terms count once."""
        scores = np.zeros(self.n_chunks)
        for t in dict.fromkeys(query_tokens):
            if t not in self.vocab:
                continue
            t_id = self.vocab[t]
            start, end = self.indptr[t_id], self.indptr[t_id + 1]
            ids = self.chunk_ids[start:end]
            tf = self.tfs[start:end]
            d_len = self.doc_len[ids]
            w = self.idf[t_id] * tf * (self.k1 + 1) / (
                tf + self.k1 * (1 - self.b + self.b * d_len / self.avgdl))
            scores[ids] += w
        return scores

    def top_k(self, query_tokens: list[str],
              k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` best ``(chunk_id, score)`` with score > 0."""
        scores = self.scores(query_tokens)
        ids = np.flatnonzero(scores > 0)
        if len(ids) == 0 or k <= 0:
            return []
        best = ids[top_k_ids(scores[ids], k)]
        return [(int(c), float(scores[c])) for c in best]

    def save(self, path: str | Path) -> None:
        """Write the raw arrays, terms (in id order), k1 and b to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        terms = sorted(self.vocab, key=self.vocab.__getitem__)
        # UTF-8 bytes: a numpy str array would take 4 bytes per character
        encoded_terms = "\n".join(terms).encode("utf-8")
        np.savez(path, indptr=self.indptr, chunk_ids=self.chunk_ids,
                 tfs=self.tfs, doc_len=self.doc_len,
                 terms=np.frombuffer(encoded_terms, dtype=np.uint8),
                 params=np.array([self.k1, self.b]))

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        """Read an index written by ``save``; FileNotFoundError if absent."""
        with np.load(path) as data:
            indptr = data["indptr"]
            chunk_ids = data["chunk_ids"]
            tfs = data["tfs"]
            doc_len = data["doc_len"]
            joined_terms = data["terms"].tobytes().decode("utf-8")
            terms = joined_terms.split("\n") if joined_terms else []
            vocab = {t: i for i, t in enumerate(terms)}
            k1 = float(data["params"][0])
            b = float(data["params"][1])
        return cls(vocab, indptr, chunk_ids, tfs, doc_len, k1, b)
