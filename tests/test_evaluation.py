from src.evaluation import recall_at_k, source_overlaps
from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


def test_source_overlaps_when_file_and_iou_match() -> None:
    expected = MinimalSource(
        file_path="data/raw/x.py",
        first_character_index=100,
        last_character_index=200,
    )
    retrieved = MinimalSource(
        file_path="data/raw/x.py",
        first_character_index=95,
        last_character_index=205,
    )

    assert source_overlaps(retrieved, expected)


def test_source_overlaps_rejects_different_file() -> None:
    expected = MinimalSource(
        file_path="data/raw/x.py",
        first_character_index=100,
        last_character_index=200,
    )
    retrieved = MinimalSource(
        file_path="data/raw/y.py",
        first_character_index=95,
        last_character_index=205,
    )

    assert not source_overlaps(retrieved, expected)


def test_recall_at_k_counts_each_ground_truth_source_once() -> None:
    expected = MinimalSource(
        file_path="data/raw/x.py",
        first_character_index=100,
        last_character_index=200,
    )
    retrieved = MinimalSource(
        file_path="data/raw/x.py",
        first_character_index=95,
        last_character_index=205,
    )
    student_results = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question="Where is x?",
                retrieved_sources=[retrieved],
            )
        ],
        k=1,
    )
    ground_truth = RagDataset(
        rag_questions=[
            AnsweredQuestion(
                question_id="q1",
                question="Where is x?",
                sources=[expected],
                answer="It is in x.py.",
            )
        ]
    )

    assert recall_at_k(student_results, ground_truth, k=1) == 1.0
