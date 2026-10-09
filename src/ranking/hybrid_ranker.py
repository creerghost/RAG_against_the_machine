from .base import Ranker


class HybridRanker(Ranker):
    def __init__(self, rankers: list[Ranker], rrf_k: int,
                 rrf_candidates: int) -> None:
        self.rankers = rankers
        self.rrf_candidates = rrf_candidates
        self.rrf_k = rrf_k

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        depth = max(k, self.rrf_candidates)
        chunks: dict[int, float] = {}
        for ranker in self.rankers:
            scores = ranker.rank(query, depth)
            for rank, (chunk_id, _) in enumerate(scores, start=1):
                score = 1 / (rank + self.rrf_k)
                chunks[chunk_id] = score + chunks.get(chunk_id, 0.0)
        sorted_chunk_scores = sorted(
            chunks.items(), key=lambda x: x[1], reverse=True)[:k]
        return sorted_chunk_scores
