from abc import ABC, abstractmethod


class BaseIndexer(ABC):
    @abstractmethod
    def index(self, corpus_path: str, max_chunk_size: int) -> None:
        """Walk corpus_path, chunk files, build and persist index."""
        pass

    @abstractmethod
    def save(self, output_path: str) -> None:
        """Persist the index to disk."""
        pass
