from ..interfaces import BaseChunker
from ..models import Chunk


class MarkdownChunker(BaseChunker):
    def chunk(
        self, text: str, file_path: str,
        max_chunk_size: int
    ) -> list[Chunk]:
        pass
