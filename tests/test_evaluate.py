"""Tests for recall@k."""

import pytest

from src.evaluate import _iou, recall_at_k
from src.models import (AnsweredQuestion, MinimalSearchResults,
                        MinimalSource, RagDataset, StudentSearchResults,
                        UnansweredQuestion)


def src(path: str, a: int, b: int) -> MinimalSource:
    """Return a source span ``[a, b)`` in ``path``."""
    return MinimalSource(file_path=path, first_character_index=a,
                         last_character_index=b)


def results(*retrieved: list[MinimalSource]) -> StudentSearchResults:
    """Return results for questions q0, q1, ... with ``retrieved`` each."""
    return StudentSearchResults(k=10, search_results=[
        MinimalSearchResults(question_id=f"q{i}", question="?",
                             retrieved_sources=r)
        for i, r in enumerate(retrieved)])


def dataset(*refs: list[MinimalSource]) -> RagDataset:
    """Return answered questions q0, q1, ... with reference ``refs``."""
    return RagDataset(rag_questions=[
        AnsweredQuestion(question_id=f"q{i}", question="?", sources=r,
                         answer="") for i, r in enumerate(refs)])


def test_iou() -> None:
    """Identical, partial, disjoint and empty spans."""
    assert _iou(src("f", 0, 10), src("f", 0, 10)) == 1.0
    assert _iou(src("f", 100, 200), src("f", 150, 250)) == pytest.approx(
        50 / 150)
    assert _iou(src("f", 0, 10), src("f", 20, 30)) == 0.0
    assert _iou(src("f", 5, 5), src("f", 5, 5)) == 0.0


def test_hit_rules() -> None:
    """Same file and IoU >= 0.05 count; other file or tiny IoU do not."""
    ref = [src("a.md", 100, 160)]
    cases = {
        "exact": [src("a.md", 100, 160)],
        "other file": [src("b.md", 100, 160)],
        "too big": [src("a.md", 0, 2000)],  # IoU 0.03
    }
    for name, retrieved in cases.items():
        expected = 1.0 if name == "exact" else 0.0
        got = recall_at_k(results(retrieved), dataset(ref), [1], 0.05)
        assert got == {1: expected}, name


def test_rank_cutoff() -> None:
    """A hit at rank 7 counts at @10 but not at @5."""
    retrieved = [src("x", 0, 1)] * 6 + [src("a", 0, 10)]
    got = recall_at_k(results(retrieved), dataset([src("a", 0, 10)]),
                      [5, 10], 0.05)
    assert got == {5: 0.0, 10: 1.0}


def test_multiple_references() -> None:
    """Each reference counts once: 2 hits on A, none on B -> 0.5."""
    refs = [src("a", 0, 10), src("b", 0, 10)]
    retrieved = [src("a", 0, 10), src("a", 0, 9)]
    assert recall_at_k(results(retrieved), dataset(refs), [5], 0.05) == {
        5: 0.5}


def test_missing_and_unanswered() -> None:
    """Unanswered questions are skipped; missing results score 0."""
    data = RagDataset(rag_questions=[
        UnansweredQuestion(question_id="u", question="?"),
        AnsweredQuestion(question_id="q0", question="?",
                         sources=[src("a", 0, 10)], answer=""),
        AnsweredQuestion(question_id="q9", question="?",
                         sources=[src("a", 0, 10)], answer="")])
    got = recall_at_k(results([src("a", 0, 10)]), data, [1], 0.05)
    assert got == {1: 0.5}
    assert recall_at_k(results(), RagDataset(rag_questions=[]), [1],
                       0.05) == {1: 0.0}
