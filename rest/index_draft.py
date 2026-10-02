import numpy as np
from collections import Counter
import itertools


class BM25Index:
    def __init__(self, vocab: dict[str, int], indptr: np.ndarray,
                 chunk_ids: np.ndarray, tfs: np.ndarray, doc_len: np.ndarray,
                 k1: float, b: float) -> None:
        self.vocab = vocab
        self.indptr = indptr
        self.chunk_ids = chunk_ids
        self.tfs = tfs
        self.doc_len = doc_len
        self.k1 = k1
        self.b = b
        # derived, never saved: always consistent with the counts
        n_chunks = len(doc_len)
        self.avgdl = float(doc_len.mean()) if n_chunks else 0.0
        df = np.diff(indptr)  # chunks containing each term
        self.idf = np.log(1 + (n_chunks - df + 0.5) / (df + 0.5))

    @classmethod
    def build(cls, docs: list[list[str]], k1: float, b: float) -> "BM25Index":
        chunks_counter = [Counter(chunk) for chunk in docs]
        unique = set(itertools.chain.from_iterable(docs))
        vocab = {word: idx for idx, word in enumerate(sorted(unique))}

        postings: list[list[tuple[int, int]]] = [[] for _ in vocab]
        for chunk_id, counter in enumerate(chunks_counter):
            for word, tf in counter.items():
                postings[vocab[word]].append((chunk_id, tf))

        lengths = [len(p) for p in postings]
        indptr = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        chunk_ids = np.array([c for p in postings for c, _ in p],
                             dtype=np.int64)
        tfs = np.array([tf for p in postings for _, tf in p], dtype=np.int64)
        doc_len = np.array([len(chunk) for chunk in docs], dtype=np.int64)
        return cls(vocab, indptr, chunk_ids, tfs, doc_len, k1, b)

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        # one score per chunk
        pass

    def top_k(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


if __name__ == "__main__":
    chunks = [
        ["lora", "adapter", "load", "lora"],
        ["load", "model"],
        ["lora", "config"]
    ]
    index = BM25Index.build(chunks, k1=1.5, b=0.75)
    print("vocab    ", index.vocab)
    print("indptr   ", index.indptr)
    print("chunk_ids", index.chunk_ids)
    print("tfs      ", index.tfs)
    print("doc_len  ", index.doc_len, "avgdl", round(index.avgdl, 2))
    print("idf      ", index.idf.round(3))
    t = index.vocab["lora"]
    print("lora ->", index.chunk_ids[index.indptr[t]:index.indptr[t + 1]],
          index.tfs[index.indptr[t]:index.indptr[t + 1]])
