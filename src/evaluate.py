"""Own recall@k, mirroring the moulinette rule (same file, IoU >= 0.05)."""

from typing import Sequence

from .models import RagDataset, StudentSearchResults


def recall_at_k(results: StudentSearchResults, dataset: RagDataset,
                ks: Sequence[int]) -> dict[int, float]:
    """Return mean recall in [0, 1] for each k in ``ks``.

    Questions of ``dataset`` without sources are skipped.
    """
    raise NotImplementedError("recall_at_k")
