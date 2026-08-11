from ..indexing import BM25Indexer, BM25Retriever
from ..generator import QwenGenerator
from .catch import catch


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
