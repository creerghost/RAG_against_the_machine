"""Hybrid ranker: fuses several rankers by weighted reciprocal rank."""

from .base import Ranker


class HybridRanker(Ranker):
    """Fuses ``rankers`` by rank, so their score scales never mix."""
    def __init__(self, rankers: list[Ranker], weights: list[float],
                 rrf_k: int, rrf_candidates: int) -> None:
        """Store ``rankers`` with one vote weight each in ``weights``."""
        if len(weights) != len(rankers):
            raise ValueError(f"{len(rankers)} rankers need as many weights, "
                             f"got {len(weights)}")
        self.rankers = rankers
        self.weights = weights
        self.rrf_k = rrf_k
        self.rrf_candidates = rrf_candidates

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return the top ``k`` (chunk id, fused score) pairs, best first.

        Each ranker votes weight / (rrf_k + rank) for its top candidates.
        """
        depth = max(k, self.rrf_candidates)
        chunks: dict[int, float] = {}
        for ranker, weight in zip(self.rankers, self.weights):
            scores = ranker.rank(query, depth)
            for rank, (chunk_id, _) in enumerate(scores, start=1):
                score = weight / (self.rrf_k + rank)
                chunks[chunk_id] = score + chunks.get(chunk_id, 0.0)
        return sorted(chunks.items(), key=lambda x: x[1], reverse=True)[:k]
