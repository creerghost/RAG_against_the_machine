import json
from pathlib import Path

from src.app.pipeline import Pipeline
from src.models import StudentSearchResults


def test_index_uses_configured_index_path(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "guide.md").write_text("# Guide\nUse BM25 retrieval.")
    index_path = tmp_path / "processed" / "index.pkl"

    Pipeline().index(
        corpus_path=str(corpus),
        max_chunk_size=100,
        index_path=str(index_path),
    )

    assert index_path.exists()


def test_search_dataset_writes_assignment_schema(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "guide.md").write_text("# Guide\nUse BM25 retrieval.")
    index_path = tmp_path / "index.pkl"
    dataset_path = tmp_path / "questions.json"
    dataset_path.write_text(
        json.dumps(
            {
                "rag_questions": [
                    {"question_id": "q1", "question": "How do I retrieve?"}
                ]
            }
        )
    )
    output_dir = tmp_path / "output"

    pipeline = Pipeline()
    pipeline.index(str(corpus), 100, str(index_path))
    pipeline.search_dataset(
        str(dataset_path), 1, str(output_dir), str(index_path)
    )

    output_file = output_dir / dataset_path.name
    results = StudentSearchResults.model_validate_json(
        output_file.read_text()
    )
    source = results.search_results[0].retrieved_sources[0]
    assert source.first_character_index == 0
    assert source.last_character_index > 0
