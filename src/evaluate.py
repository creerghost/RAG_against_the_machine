"""Own recall@k, mirroring the moulinette rule (same file, IoU >= 0.05)."""

from typing import Dict, Sequence

from .models import RagDataset, StudentSearchResults


def recall_at_k(
    results: StudentSearchResults,
    dataset: RagDataset,
    ks: Sequence[int],
) -> Dict[int, float]:
    """Compute mean recall@k over the questions of ``dataset``.

    Args:
        results: Student search results.
        dataset: Ground truth; questions without sources are skipped.
        ks: Cut-offs to report, e.g. ``(1, 3, 5, 10)``.

    Returns:
        Dict[int, float]: Recall in [0, 1] for each k.
    """
    raise NotImplementedError("recall_at_k")
