from abc import ABC, abstractmethod
from ..models import Chunk


class BaseChunker(ABC):
    """Interface implemented by source-specific chunking strategies."""

    @abstractmethod
    def chunk(
        self, text: str, file_path: str, max_chunk_size: int
    ) -> list[Chunk]:
        """
        Takes raw text and splits it into chunks.
        Must return objects containing the chunk text, file_path,
        first_character_index, and last_character_index.
        """
        pass
