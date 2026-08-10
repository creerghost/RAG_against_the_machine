from .markdown_chunker import MarkdownChunker
from .python_chunker import PythonChunker
from ..interfaces import BaseChunker


class ChunkerFactory:
    @staticmethod
    def route(file_path: str) -> BaseChunker | None:
        if file_path.endswith('.md'):
            return MarkdownChunker()
        elif file_path.endswith('.py'):
            return PythonChunker()
        else:
            raise ValueError(f"Unknown file type: {file_path}")
