from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel
from tqdm import tqdm

from ..evaluation import recall_at_k
from ..generator import QwenGenerator
from ..indexing import BM25Indexer, BM25Retriever
from ..models import (
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from .catch import catch


DEFAULT_CORPUS_PATH = "data/raw"
DEFAULT_INDEX_PATH = "data/processed/index.pkl"
DEFAULT_SEARCH_OUTPUT_DIR = "data/output/search_results"
DEFAULT_ANSWER_OUTPUT_DIR = "data/output/search_results_and_answer"


class Pipeline:
    """Expose the RAG stages as commands for Python Fire."""

    @catch
    def _save_json(
        self,
        data: BaseModel,
        original_path: str,
        save_dir: str,
        count: int,
        item_name: str,
    ) -> None:
        """Save a Pydantic model using the input dataset's filename."""
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        output_file = save_dir_path / Path(original_path).name
        output_file.write_text(data.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved {count} {item_name} to {output_file}")

    @catch
    def index(
        self,
        corpus_path: str = DEFAULT_CORPUS_PATH,
        max_chunk_size: int = 2000,
        index_path: str = DEFAULT_INDEX_PATH,
    ) -> None:
        """Index Python, Markdown, and text files from the target corpus."""
        indexer = BM25Indexer()
        indexer.index(corpus_path, max_chunk_size)
        output_path = Path(index_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        indexer.save(str(output_path))
        print(f"Ingestion complete! Index saved to {output_path}")

    @catch
    def search(
        self,
        question: str,
        k: int = 10,
        index_path: str = DEFAULT_INDEX_PATH,
    ) -> None:
        """Print ranked sources for one question as StudentSearchResults JSON."""
        retriever = BM25Retriever()
        retriever.load(index_path)
        result = MinimalSearchResults(
            question_id=str(uuid4()),
            question=question,
            retrieved_sources=retriever.search(question, k),
        )
        print(
            StudentSearchResults(
                search_results=[result],
                k=max(k, 0),
            ).model_dump_json(indent=2)
        )

    @catch
    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = DEFAULT_SEARCH_OUTPUT_DIR,
        index_path: str = DEFAULT_INDEX_PATH,
    ) -> None:
        """Search every question in a JSON dataset and save the results."""
        dataset = RagDataset.model_validate_json(
            Path(dataset_path).read_text(encoding="utf-8")
        )
        retriever = BM25Retriever()
        retriever.load(index_path)
        all_search_results: list[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            all_search_results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=retriever.search(question.question, k),
                )
            )

        self._save_json(
            StudentSearchResults(
                search_results=all_search_results,
                k=max(k, 0),
            ),
            dataset_path,
            save_directory,
            len(all_search_results),
            "search results",
        )

    @catch
    def answer(
        self,
        question: str,
        k: int = 10,
        index_path: str = DEFAULT_INDEX_PATH,
    ) -> None:
        """Answer one question from the top-k retrieved source locations."""
        retriever = BM25Retriever()
        retriever.load(index_path)
        sources = retriever.search(question, k)
        answer = "No relevant sources were retrieved."
        if sources:
            answer = QwenGenerator().generate(question, sources)
        output = StudentSearchResultsAndAnswer(
            search_results=[
                MinimalAnswer(
                    question_id=str(uuid4()),
                    question=question,
                    retrieved_sources=sources,
                    answer=answer,
                )
            ],
            k=max(k, 0),
        )
        print(output.model_dump_json(indent=2))

    @catch
    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = DEFAULT_ANSWER_OUTPUT_DIR,
    ) -> None:
        """Generate grounded answers for a saved search-results dataset."""
        student_results = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_text(encoding="utf-8")
        )
        generator: QwenGenerator | None = None
        all_answers: list[MinimalAnswer] = []
        for result in tqdm(student_results.search_results, desc="Answering"):
            if result.retrieved_sources:
                if generator is None:
                    generator = QwenGenerator()
                answer = generator.generate(
                    result.question,
                    result.retrieved_sources,
                )
            else:
                answer = "No relevant sources were retrieved."
            all_answers.append(
                MinimalAnswer(
                    **result.model_dump(),
                    answer=answer,
                )
            )

        self._save_json(
            StudentSearchResultsAndAnswer(
                search_results=all_answers,
                k=student_results.k,
            ),
            student_search_results_path,
            save_directory,
            len(all_answers),
            "answers",
        )

    @catch
    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 10,
    ) -> None:
        """Print recall@1, @3, @5, and @10 for a ground-truth dataset."""
        student_results = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_text(encoding="utf-8")
        )
        ground_truth = RagDataset.model_validate_json(
            Path(dataset_path).read_text(encoding="utf-8")
        )
        for metric_k in (1, 3, 5, 10):
            score = recall_at_k(student_results, ground_truth, metric_k)
            print(f"Recall@{metric_k}: {score:.3f}")
