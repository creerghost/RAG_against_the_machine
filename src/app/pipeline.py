from ..indexing import BM25Indexer, BM25Retriever
from ..generator import QwenGenerator
from ..models import (RagDataset, MinimalSearchResults, StudentSearchResults,
                      MinimalAnswer, StudentSearchResultsAndAnswer)
from .catch import catch
from pathlib import Path
from pydantic import BaseModel
from tqdm import tqdm


class Pipeline:
    @catch
    def _save_json(
        self, data: BaseModel, original_path: str,
        save_dir: str, count: int, item_name: str
    ) -> None:
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        output_file = save_dir_path / Path(original_path).name
        with open(output_file, "w") as f:
            f.write(data.model_dump_json(indent=2))
        print(f"Successfully saved {count} {item_name} to {output_file}!")

    @catch
    def index(
        self, corpus_path: str, max_chunk_size: int = 2000
    ) -> None:
        print(f"Indexing {corpus_path} with chunk size {max_chunk_size}...")
        indexer = BM25Indexer()
        indexer.index(corpus_path, max_chunk_size)
        indexer.save("index.pkl")
        print("Done!")

    @catch
    def search(self, question: str, k: int = 10) -> None:
        print(f"Searching for '{question}'...")
        retriever = BM25Retriever()
        retriever.load("index.pkl")
        results = retriever.search(question, k)
        for r in results:
            print(f"- Found in {r.file_path}")

    @catch
    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> None:
        print(f"Reading {dataset_path}...")
        with open(dataset_path, "r") as f:
            json_string = f.read()
        print("Validating the questions...")
        dataset = RagDataset.model_validate_json(json_string)
        retriever = BM25Retriever()
        retriever.load("index.pkl")
        all_search_results: list[MinimalSearchResults] = []
        print("Searching...")
        for q in dataset.rag_questions:
            sources = retriever.search(q.question, k)
            all_search_results.append(MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=sources
            ))
        print("Searching is done!")
        student_results = StudentSearchResults(
            search_results=all_search_results,
            k=k
        )
        print(f"Saving to {dataset_path}...")
        self._save_json(
            student_results,
            dataset_path,
            save_directory,
            len(all_search_results),
            "results"
        )

    @catch
    def answer(self, question: str, k: int = 10) -> None:
        print(f"Question: {question}")
        print("Searching top-k source locations for a query...")
        retriever = BM25Retriever()
        try:
            retriever.load("index.pkl")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{e}. Try to do search first.")
        sources = retriever.search(question, k)
        print(QwenGenerator().generate(question, sources))

    @catch
    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        print(f"Reading {student_search_results_path}...")
        with open(student_search_results_path, "r") as f:
            json_string = f.read()
        print("Validating the student search results...")
        student_results = StudentSearchResults.model_validate_json(json_string)
        generator = QwenGenerator()
        all_answers = []
        print("Generating the answers...")
        for result in tqdm(student_results.search_results, desc="Answering"):
            answer = generator.generate(
                result.question,
                result.retrieved_sources
            )
            minimal_answer = MinimalAnswer(
                **result.model_dump(),
                answer=answer
            )
            all_answers.append(minimal_answer)
        student_results_and_answer = StudentSearchResultsAndAnswer(
            search_results=all_answers,
            k=student_results.k
        )
        print(f"Done! Saving to {save_directory}...")

        self._save_json(
            student_results_and_answer,
            student_search_results_path,
            save_directory,
            len(all_answers),
            "answers"
        )
