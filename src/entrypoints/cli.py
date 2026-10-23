"""Fire CLI: argument checks, I/O and progress bars around the pipeline.

Settings come from ``RagConfig`` (``.env`` and ``RAG_*`` environment
variables); commands only take their own arguments. Commands return
``None``; expected failures raise ``CliError``, which ``main`` in
``src/__main__.py`` prints without a traceback.
"""

from pathlib import Path
from typing import Any

from pydantic import ValidationError
from tqdm import tqdm

from src.config import RagConfig
from src.evaluate import recall_at_k
from src.io_utils import CliError, load_model, save_model
from src.models import (AnsweredQuestion, MinimalAnswer, MinimalSearchResults,
                        RagDataset, StudentSearchResults,
                        StudentSearchResultsAndAnswer)
from src.service import RagService
from .cli_utils import (as_path, as_positive_int, as_text, env_name,
                        print_sources)


class RagCli:
    """RAG over the vLLM codebase: index, search, answer, evaluate."""
    def __init__(self) -> None:
        """Read the settings (defaults, ``.env``, ``RAG_*`` env vars)."""
        try:
            self.config = RagConfig()
        except ValidationError as e:
            errors = "; ".join(f"{env_name(err['loc'])}: {err['msg']}"
                               for err in e.errors())
            raise CliError(errors) from e
        self._service = RagService(self.config)

    def _load_index(self) -> None:
        """Load the index now, turning a missing one into a ``CliError``."""
        processed = self.config.paths.processed_dir
        mode = self.config.retrieval.mode
        # semantic and hybrid also need embeddings.npy
        command = ("'index'" if mode == "lexical"
                   else "'index --semantic'")
        try:
            self._service.load_index()
        except FileNotFoundError as e:
            if Path(str(e.filename)).name == "embeddings.npy":
                raise CliError(f"RAG_RETRIEVAL__MODE={mode} needs embeddings "
                               f"in {processed}. Run {command} first.") from e
            raise CliError(f"No index in {processed} ({e}). "
                           f"Run {command} first.") from e
        except (OSError, ValueError, KeyError, AttributeError) as e:
            # corrupt, old-format or mismatched index files
            raise CliError(f"Index in {processed} is unreadable ({e}). "
                           f"Run {command} again.") from e

    def _k(self, k: Any) -> int:
        """Return a validated ``k``, or the config default when omitted."""
        default = self.config.retrieval.default_k
        return default if k is None else as_positive_int(k, "k")

    def index(self, max_chunk_size: Any = None, semantic: Any = False) -> None:
        """Chunk the corpus and build the index with ``max_chunk_size``.

        With ``--semantic True`` also embed every chunk (``embeddings.npy``).
        """
        if not isinstance(semantic, bool):
            raise CliError(f"--semantic must be True or False, got "
                           f"{semantic!r}")
        cfg = self.config
        size = None
        if max_chunk_size is not None:
            size = as_positive_int(max_chunk_size, "max_chunk_size",
                                   cfg.chunking.max_context_length)
        raw_dir = cfg.paths.raw_dir
        if not Path(raw_dir).is_dir():
            raise CliError(f"Corpus directory not found: {raw_dir}")
        try:
            n_chunks = self._service.index(semantic, size)
        except PermissionError as e:
            raise CliError(f"Cannot write the index: {e}") from e
        except OSError as e:  # embedding model missing and not downloadable
            raise CliError(f"Cannot load {cfg.embedding.model}: {e}") from e
        extra = " with embeddings" if semantic else ""
        print(f"Ingestion complete! Indexed {n_chunks} chunks{extra} under "
              f"{cfg.paths.processed_dir}/")

    def search(self, query: Any = None, k: Any = None) -> None:
        """Print the top ``k`` sources for ``query``."""
        text, k = as_text(query, "query"), self._k(k)
        self._load_index()
        print_sources(self._service.search(text, k))

    def search_dataset(self, dataset_path: Any = None, k: Any = None,
                       save_directory: Any = None) -> None:
        """Search every question of ``dataset_path``; save to the directory."""
        src = as_path(dataset_path, "dataset_path")
        out_dir = as_path(save_directory, "save_directory")
        k = self._k(k)
        if (out_dir / src.name).resolve() == src.resolve():
            raise CliError(f"--save_directory {out_dir} would overwrite the "
                           "dataset itself; choose another directory")
        dataset = load_model(src, RagDataset)
        self._load_index()
        results: list[MinimalSearchResults] = []
        for q in tqdm(dataset.rag_questions, desc="Searching", unit="q"):
            text = q.question.strip()
            sources = self._service.search(text, k) if text else []
            results.append(MinimalSearchResults(
                question_id=q.question_id, question=q.question,
                retrieved_sources=sources))
        out = out_dir / src.name
        save_model(StudentSearchResults(search_results=results, k=k), out)
        print(f"Saved student_search_results to {out}")

    def answer(self, query: Any = None, k: Any = None) -> None:
        """Print a generated answer to ``query`` and its ``k`` sources."""
        text, k = as_text(query, "query"), self._k(k)
        self._load_index()
        sources = self._service.search(text, k)
        # no sources: the fixed reply, without loading the model
        print(self._service.answer(text, sources))
        print("\nSources:")
        print_sources(sources)

    def answer_dataset(self, student_search_results_path: Any = None,
                       save_directory: Any = None) -> None:
        """Answer every question of a search results file; save the output."""
        name = "student_search_results_path"
        src = as_path(student_search_results_path, name)
        out_dir = as_path(save_directory, "save_directory")
        searched = load_model(src, StudentSearchResults)
        total = len(searched.search_results)
        print(f"Loaded {total} questions from {src}")
        answers: list[MinimalAnswer] = []
        for r in tqdm(searched.search_results, desc="Answering", unit="q"):
            answer = self._service.answer(r.question, r.retrieved_sources)
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
        results_path = as_path(student_search_results_path, name)
        results = load_model(results_path, StudentSearchResults)
        dataset = load_model(as_path(dataset_path, "dataset_path"),
                             RagDataset)
        if not any(isinstance(q, AnsweredQuestion) and q.sources
                   for q in dataset.rag_questions):
            raise CliError(f"{dataset_path} has no reference sources; use "
                           "an AnsweredQuestions dataset as ground truth")
        recalls = recall_at_k(results, dataset, self.config.evaluation.ks,
                              self.config.evaluation.min_iou)
        print(" ".join(f"Recall@{k}: {r:.3f}" for k, r in recalls.items()))
