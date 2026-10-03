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
        self.n_chunks = len(doc_len)
        self.avgdl = float(doc_len.mean()) if self.n_chunks else 0.0
        df = np.diff(indptr)  # chunks containing each term
        self.idf = np.log(1 + (self.n_chunks - df + 0.5) / (df + 0.5))

    @classmethod
    def build(cls, docs: list[list[str]], k1: float, b: float) -> "BM25Index":
        tf = [Counter(chunk) for chunk in docs]
        unique = set(itertools.chain.from_iterable(docs))
        vocab = {word: idx for idx, word in enumerate(sorted(unique))}

        postings: list[list[tuple[int, int]]] = [[] for _ in vocab]
        for chunk_id, counter in enumerate(tf):
            for word, tf in counter.items():
                postings[vocab[word]].append((chunk_id, tf))

        # these 4 variables represents postings without nested lists
        # lengths, indptr, chunk_ids, tfs

        # lengths = nbr of chunks containing that word
        lengths = [len(p) for p in postings]
        # we are about to flatten nested list -> we need to remember
        # where each word's imformation starts and ends -> indptr
        # basically word boundaries indptr = f(lengths)
        indptr = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        chunk_ids = np.array([c for p in postings for c, _ in p],
                             dtype=np.int64)
        tfs = np.array([tf for p in postings for _, tf in p], dtype=np.int64)
        doc_len = np.array([len(chunk) for chunk in docs], dtype=np.int64)
        return cls(vocab, indptr, chunk_ids, tfs, doc_len, k1, b)

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_chunks)
        # dict.fromkeys() will remove duplicate tokens
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

    def top_k(self, query_tokens: list[str], k: int
              ) -> list[tuple[int, float]]:
        scores = self.scores(query_tokens)
        ids = np.flatnonzero(scores > 0)
        if len(ids) == 0 or k <= 0:
            return []
        if k > len(ids):
            k = len(ids)
        valid_scores = scores[ids]
        top_k_idx = np.argpartition(valid_scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-valid_scores[top_k_idx])]
        best = ids[top_k_idx_sorted]
        return [(int(c), float(scores[c])) for c in best]

        

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
    print(index.top_k(["hello", "config", "lora"], 5))