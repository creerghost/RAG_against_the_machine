from .markdown_chunker import MarkdownChunker
from .python_chunker import PythonChunker
from ..interfaces import BaseChunker


class ChunkerFactory:
    """Select a chunking strategy from a source file extension."""

    @staticmethod
    def route(file_path: str) -> BaseChunker | None:
        """Return the chunker appropriate for ``file_path``."""
        normalized_path = file_path.lower()
        if normalized_path.endswith(('.md', '.txt', '.text')):
            return MarkdownChunker()
        elif normalized_path.endswith('.py'):
            return PythonChunker()
        else:
            raise ValueError(f"Unknown file type: {file_path}")
