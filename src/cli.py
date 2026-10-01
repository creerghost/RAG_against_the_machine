"""Fire command-line interface: argument checks, I/O and progress bars.

Every public method of ``RagCli`` is a command. Commands return ``None`` so
Fire prints nothing extra; expected failures raise ``CliError`` and are
printed by ``main`` without a traceback.
"""

import sys
from pathlib import Path
from typing import Any, List

import fire
from tqdm import tqdm

from .evaluate import recall_at_k
from .index import build_index
from .io_utils import CliError, load_model, save_model
from .models import (MinimalAnswer, MinimalSearchResults, MinimalSource,
                     RagDataset, StudentSearchResults,
                     StudentSearchResultsAndAnswer)
from .retriever import Retriever

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
MAX_CONTEXT_LENGTH = 2000
EVAL_KS = (1, 3, 5, 10)


def _as_text(value: Any, name: str) -> str:
    """Coerce a Fire argument to a non-empty string.

    Fire parses ``search 42`` as an int and ``search True`` as a bool, so
    every free-text argument goes through here.
    """
    if value is None:
        raise CliError(f"--{name} is required")
    text = str(value).strip()
    if not text:
        raise CliError(f"--{name} must not be empty")
    return text


def _as_positive_int(value: Any, name: str, maximum: int = 0) -> int:
    """Validate a Fire argument as an int in ``[1, maximum]``.

    ``maximum`` of 0 means no upper bound. Bools are rejected even though
    ``bool`` subclasses ``int``.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise CliError(f"--{name} must be an integer, got {value!r}")
    if value < 1:
        raise CliError(f"--{name} must be >= 1, got {value}")
    if maximum and value > maximum:
        raise CliError(f"--{name} must be <= {maximum}, got {value}")
    return value


def _as_path(value: Any, name: str) -> Path:
    """Coerce a Fire argument to a ``Path``."""
    return Path(_as_text(value, name))


def _print_sources(sources: List[MinimalSource]) -> None:
    """Print one ``path [start:end]`` line per source."""
    if not sources:
        print("No relevant source found.")
    for s in sources:
        print(
            f"{s.file_path} "
            f"[{s.first_character_index}:{s.last_character_index}]"
        )


class RagCli:
    """RAG over the vLLM codebase: index, search, answer, evaluate."""

    def __init__(self, raw_dir: str = "data/raw",
                 processed_dir: str = "data/processed",
                 model_name: str = DEFAULT_MODEL) -> None:
        """Store options shared by all commands.

        Args:
            raw_dir: Corpus root to index.
            processed_dir: Where the index is written and read.
            model_name: Hugging Face id of the generation model.
        """
        self._raw_dir = raw_dir
        self._processed_dir = processed_dir
        self._model_name = model_name

    def _retriever(self) -> Retriever:
        """Load the index, turning a missing index into a ``CliError``."""
        processed = _as_path(self._processed_dir, "processed_dir")
        try:
            return Retriever(processed)
        except FileNotFoundError as e:
            raise CliError(
                f"No index in {processed} ({e}). Run the 'index' command "
                "first."
            ) from e

    def index(self, max_chunk_size: int = MAX_CONTEXT_LENGTH) -> None:
        """Ingest the corpus and build the index.

        Args:
            max_chunk_size: Maximum chunk width in characters (1-2000).
        """
        size = _as_positive_int(
            max_chunk_size, "max_chunk_size", MAX_CONTEXT_LENGTH
        )
        raw = _as_path(self._raw_dir, "raw_dir")
        processed = _as_path(self._processed_dir, "processed_dir")
        if not raw.is_dir():
            raise CliError(f"Corpus directory not found: {raw}")
        n_chunks = build_index(raw, processed, size)
        print(
            f"Ingestion complete! Indexed {n_chunks} chunks under "
            f"{processed}/"
        )

    def search(self, query: Any = None, k: int = 10) -> None:
        """Print the top-k sources for a single query.

        Args:
            query: Question text.
            k: Number of sources to return.
        """
        text = _as_text(query, "query")
        k = _as_positive_int(k, "k")
        _print_sources(self._retriever().search(text, k))

    def search_dataset(self, dataset_path: Any = None, k: int = 10,
                       save_directory: Any = None) -> None:
        """Search every question of a dataset; save StudentSearchResults.

        Args:
            dataset_path: RagDataset JSON file.
            k: Number of sources per question.
            save_directory: Output directory; the file keeps the dataset's
                name.
        """
        src = _as_path(dataset_path, "dataset_path")
        out_dir = _as_path(save_directory, "save_directory")
        k = _as_positive_int(k, "k")
        dataset = load_model(src, RagDataset)
        retriever = self._retriever()

        results: List[MinimalSearchResults] = []
        for q in tqdm(dataset.rag_questions, desc="Searching", unit="q"):
            text = q.question.strip()
            sources = retriever.search(text, k) if text else []
            results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=sources,
                )
            )
        out = out_dir / src.name
        save_model(StudentSearchResults(search_results=results, k=k), out)
        print(f"Saved student_search_results to {out}")

    def answer(self, query: Any = None, k: int = 10) -> None:
        """Retrieve context for one query and print a generated answer.

        Args:
            query: Question text.
            k: Number of sources given to the model.
        """
        from .generator import Generator  # heavy: torch, transformers

        text = _as_text(query, "query")
        k = _as_positive_int(k, "k")
        sources = self._retriever().search(text, k)
        print(Generator(self._model_name).answer(text, sources))
        print("\nSources:")
        _print_sources(sources)

    def answer_dataset(self, student_search_results_path: Any = None,
                       save_directory: Any = None) -> None:
        """Answer every question of a StudentSearchResults file.

        Args:
            student_search_results_path: Output of ``search_dataset``.
            save_directory: Output directory; the file keeps the input's
                name.
        """
        from .generator import Generator  # heavy: torch, transformers

        src = _as_path(
            student_search_results_path, "student_search_results_path"
        )
        out_dir = _as_path(save_directory, "save_directory")
        searched = load_model(src, StudentSearchResults)
        total = len(searched.search_results)
        print(f"Loaded {total} questions from {src}")
        generator = Generator(self._model_name)

        answers: List[MinimalAnswer] = []
        for r in tqdm(searched.search_results, desc="Answering", unit="q"):
            answers.append(
                MinimalAnswer(
                    question_id=r.question_id,
                    question=r.question,
                    retrieved_sources=r.retrieved_sources,
                    answer=generator.answer(r.question, r.retrieved_sources),
                )
            )
        out = out_dir / src.name
        save_model(
            StudentSearchResultsAndAnswer(
                search_results=answers, k=searched.k
            ),
            out,
        )
        print(f"Processed {len(answers)} of {total} questions")
        print(f"Saved student_search_results_and_answer to {out}")

    def evaluate(self, student_search_results_path: Any = None,
                 dataset_path: Any = None) -> None:
        """Print recall@k of search results against a ground-truth dataset.

        Args:
            student_search_results_path: Output of ``search_dataset``.
            dataset_path: AnsweredQuestions RagDataset JSON file.
        """
        results = load_model(
            _as_path(
                student_search_results_path, "student_search_results_path"
            ),
            StudentSearchResults,
        )
        dataset = load_model(
            _as_path(dataset_path, "dataset_path"), RagDataset
        )
        recalls = recall_at_k(results, dataset, EVAL_KS)
        print(" ".join(f"Recall@{k}: {r:.3f}" for k, r in recalls.items()))


def main() -> None:
    """Run the Fire CLI and print expected errors without a traceback."""
    try:
        fire.Fire(RagCli, name="src")
    except CliError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as e:
        print(f"Error: not implemented yet: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:  # last resort: the CLI must never show a trace
        print(f"Unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
