"""Fire CLI: argument checks, I/O and progress bars around the pipeline.

Commands return ``None``; expected failures raise ``CliError``, which
``main`` prints without a traceback.
"""

import sys
from pathlib import Path
from typing import Any, Optional

import fire
from tqdm import tqdm

from .config import RagConfig
from .evaluate import recall_at_k
from .index import build_index
from .io_utils import CliError, load_model, save_model
from .models import (MinimalAnswer, MinimalSearchResults, MinimalSource,
                     RagDataset, StudentSearchResults,
                     StudentSearchResultsAndAnswer)
from .retriever import Retriever


def _as_text(value: Any, name: str) -> str:
    """Return ``value`` as a non-empty stripped string (Fire may pass ints)."""
    if value is None:
        raise CliError(f"--{name} is required")
    text = str(value).strip()
    if not text:
        raise CliError(f"--{name} must not be empty")
    return text


def _as_positive_int(value: Any, name: str, maximum: int = 0) -> int:
    """Return ``value`` if it is an int in ``[1, maximum]`` (0: no maximum)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CliError(f"--{name} must be an integer, got {value!r}")
    if value < 1:
        raise CliError(f"--{name} must be >= 1, got {value}")
    if maximum and value > maximum:
        raise CliError(f"--{name} must be <= {maximum}, got {value}")
    return value


def _as_path(value: Any, name: str) -> Path:
    """Return ``value`` as a ``Path``; it must be a non-empty string."""
    return Path(_as_text(value, name))


def _print_sources(sources: list[MinimalSource]) -> None:
    """Print one ``path [start:end]`` line per source."""
    if not sources:
        print("No relevant source found.")
    for s in sources:
        print(f"{s.file_path} "
              f"[{s.first_character_index}:{s.last_character_index}]")


class RagCli:
    """RAG over the vLLM codebase: index, search, answer, evaluate."""
    def __init__(self, raw_dir: Optional[str] = None,
                 processed_dir: Optional[str] = None,
                 model_name: Optional[str] = None) -> None:
        """Build the config; given flags override ``RagConfig`` defaults."""
        flags = {"raw_dir": raw_dir, "processed_dir": processed_dir,
                 "model_name": model_name}
        overrides = {k: str(v) for k, v in flags.items() if v is not None}
        self.config = RagConfig.model_validate(overrides)

    def _retriever(self) -> Retriever:
        """Load the index, turning a missing index into a ``CliError``."""
        processed = self.config.processed_dir
        try:
            return Retriever(self.config)
        except FileNotFoundError as e:
            raise CliError(f"No index in {processed} ({e}). "
                           "Run the 'index' command first.") from e

    def _k(self, k: Any) -> int:
        """Return a validated ``k``, or the config default when omitted."""
        return self.config.default_k if k is None else _as_positive_int(k, "k")

    def index(self, max_chunk_size: Any = None) -> None:
        """Chunk the corpus and build the index with ``max_chunk_size``."""
        cfg = self.config
        if max_chunk_size is not None:
            size = _as_positive_int(max_chunk_size, "max_chunk_size",
                                    cfg.max_context_length)
            cfg = cfg.model_copy(update={"max_chunk_size": size})
        if not Path(cfg.raw_dir).is_dir():
            raise CliError(f"Corpus directory not found: {cfg.raw_dir}")
        n_chunks = build_index(cfg)
        print(f"Ingestion complete! Indexed {n_chunks} chunks under "
              f"{cfg.processed_dir}/")

    def search(self, query: Any = None, k: Any = None) -> None:
        """Print the top ``k`` sources for ``query``."""
        text, k = _as_text(query, "query"), self._k(k)
        _print_sources(self._retriever().search(text, k))

    def search_dataset(self, dataset_path: Any = None, k: Any = None,
                       save_directory: Any = None) -> None:
        """Search every question of ``dataset_path``; save to the directory."""
        src = _as_path(dataset_path, "dataset_path")
        out_dir = _as_path(save_directory, "save_directory")
        k = self._k(k)
        dataset = load_model(src, RagDataset)
        retriever = self._retriever()
        results: list[MinimalSearchResults] = []
        for q in tqdm(dataset.rag_questions, desc="Searching", unit="q"):
            text = q.question.strip()
            sources = retriever.search(text, k) if text else []
            results.append(MinimalSearchResults(
                question_id=q.question_id, question=q.question,
                retrieved_sources=sources))
        out = out_dir / src.name
        save_model(StudentSearchResults(search_results=results, k=k), out)
        print(f"Saved student_search_results to {out}")

    def answer(self, query: Any = None, k: Any = None) -> None:
        """Print a generated answer to ``query`` and its ``k`` sources."""
        from .generator import Generator  # heavy: torch, transformers

        text, k = _as_text(query, "query"), self._k(k)
        sources = self._retriever().search(text, k)
        if sources:
            print(Generator(self.config).answer(text, sources))
        else:  # nothing to ground on: skip loading the model
            print(self.config.no_answer)
        print("\nSources:")
        _print_sources(sources)

    def answer_dataset(self, student_search_results_path: Any = None,
                       save_directory: Any = None) -> None:
        """Answer every question of a search results file; save the output."""
        from .generator import Generator  # heavy: torch, transformers

        name = "student_search_results_path"
        src = _as_path(student_search_results_path, name)
        out_dir = _as_path(save_directory, "save_directory")
        searched = load_model(src, StudentSearchResults)
        total = len(searched.search_results)
        print(f"Loaded {total} questions from {src}")
        generator = Generator(self.config)
        answers: list[MinimalAnswer] = []
        for r in tqdm(searched.search_results, desc="Answering", unit="q"):
            answer = generator.answer(r.question, r.retrieved_sources)
            answers.append(MinimalAnswer(
                question_id=r.question_id, question=r.question,
                retrieved_sources=r.retrieved_sources, answer=answer))
        out = out_dir / src.name
        result = StudentSearchResultsAndAnswer(search_results=answers,
                                               k=searched.k)
        save_model(result, out)
        print(f"Processed {len(answers)} of {total} questions")
        print(f"Saved student_search_results_and_answer to {out}")

    def evaluate(self, student_search_results_path: Any = None,
                 dataset_path: Any = None) -> None:
        """Print recall@k of the search results against ``dataset_path``."""
        name = "student_search_results_path"
        results_path = _as_path(student_search_results_path, name)
        results = load_model(results_path, StudentSearchResults)
        dataset = load_model(_as_path(dataset_path, "dataset_path"),
                             RagDataset)
        recalls = recall_at_k(results, dataset, self.config.eval_ks,
                              self.config.min_iou)
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
