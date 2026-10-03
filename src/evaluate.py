"""Own recall@k, mirroring the moulinette rule (same file, IoU >= min_iou)."""

from typing import Sequence

from .models import (AnsweredQuestion, MinimalSource, RagDataset,
                     StudentSearchResults)


def _iou(a: MinimalSource, b: MinimalSource) -> float:
    """Return intersection over union of the spans ``a`` and ``b``."""
    a1, a2 = a.first_character_index, a.last_character_index
    b1, b2 = b.first_character_index, b.last_character_index
    i = max(0, min(a2, b2) - max(a1, b1))
    u = a2 - a1 + b2 - b1 - i
    return i / u if u else 0.0


def recall_at_k(results: StudentSearchResults, dataset: RagDataset,
                ks: Sequence[int], min_iou: float) -> dict[int, float]:
    """Return mean recall in [0, 1] for each k in ``ks``.

    A reference counts as found if a top-k result in the same file has
    IoU >= ``min_iou``; questions without sources are skipped.
    """
    answers = {r.question_id: r.retrieved_sources
               for r in results.search_results}
    recall: dict[int, float] = {}
    for k in ks:
        per_question: list[float] = []
        for q in dataset.rag_questions:
            if not isinstance(q, AnsweredQuestion) or not q.sources:
                continue
            top = answers.get(q.question_id, [])[:k]
            found = 0
            for ref in q.sources:
                if any(r.file_path == ref.file_path
                       and _iou(ref, r) >= min_iou for r in top):
                    found += 1
            per_question.append(found / len(q.sources))
        recall[k] = (sum(per_question) / len(per_question)
                     if per_question else 0.0)
    return recall
