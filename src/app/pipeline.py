from ..indexing import BM25Indexer, BM25Retriever
from ..generator import QwenGenerator
from ..models import (RagDataset, MinimalSearchResults, StudentSearchResults)
from .catch import catch
from pathlib import Path


class Pipeline:
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
    def search_dataset(self, dataset_path: str, k: int, save_dir: str) -> None:
        print(f"Reading {dataset_path}...")
        with open(dataset_path, "r") as f:
            json_string = f.read()
        print(f"Validating the questions...")
        dataset = RagDataset.model_validate_json(json_string)
        retriever = BM25Retriever()
        retriever.load("index.pkl")
        all_search_results: list[MinimalSearchResults] = []
        for q in dataset.rag_questions:
            sources = retriever.search(q.question, k)
            all_search_results.append(MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=sources
            ))

        student_results = StudentSearchResults(
            search_results=all_search_results,
            k=k
        )
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)

        output_file = save_dir_path / Path(dataset_path).name
        with open(output_file, "w") as f:
            f.write(student_results.model_dump_json(indent=2))

        print(f"Successfully saved {len(all_search_results)} results "
              f"to {output_file}!")
