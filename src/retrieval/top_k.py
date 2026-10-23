"""Top-k selection over a score array, shared by every ranker."""

from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating[Any]]
IntArray64 = npt.NDArray[np.int64]


def top_k_ids(scores: FloatArray, k: int) -> IntArray64:
    """Return the positions of the ``k`` highest ``scores``, best first;
    argpartition is O(n), only the ``k`` winners get sorted."""
    k = min(k, len(scores))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    top_k_idx = np.argpartition(scores, -k)[-k:]
    return top_k_idx[np.argsort(-scores[top_k_idx])]
