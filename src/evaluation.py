from .models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


IOU_THRESHOLD = 0.05


def source_overlaps(
    retrieved: MinimalSource,
    expected: MinimalSource,
    iou_threshold: float = IOU_THRESHOLD,
) -> bool:
    """Return whether two same-file spans meet the assignment IoU threshold."""
    if retrieved.file_path != expected.file_path:
        return False

    intersection = max(
        0,
        min(
            retrieved.last_character_index,
            expected.last_character_index,
        )
        - max(
            retrieved.first_character_index,
            expected.first_character_index,
        ),
    )
    union = max(
        retrieved.last_character_index,
        expected.last_character_index,
    ) - min(
        retrieved.first_character_index,
        expected.first_character_index,
    )
    return union > 0 and intersection / union >= iou_threshold


def recall_at_k(
    student_results: StudentSearchResults,
    ground_truth: RagDataset,
    k: int,
) -> float:
    """Calculate source recall using question IDs and file/span overlap."""
    if k <= 0:
        return 0.0

    student_by_id = {
        result.question_id: result
        for result in student_results.search_results
    }
    total_sources = 0
    found_sources = 0
    for question in ground_truth.rag_questions:
        if not isinstance(question, AnsweredQuestion):
            continue
        total_sources += len(question.sources)
        student_result = student_by_id.get(question.question_id)
        if student_result is None:
            continue
        retrieved_sources = student_result.retrieved_sources[:k]
        found_sources += sum(
            any(
                source_overlaps(retrieved, expected)
                for retrieved in retrieved_sources
            )
            for expected in question.sources
        )

    if total_sources == 0:
        return 0.0
    return found_sources / total_sources
