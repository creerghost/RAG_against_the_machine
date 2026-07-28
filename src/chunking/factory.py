from .markdown_chunker import MarkdownChunker
from .python_chunker import PythonChunker


class ChunkerFactory:
    @staticmethod
    def route(file_path: str) -> None:
        if file_path.endswith('.md'):
            return MarkdownChunker()
        elif file_path.endswith('.py'):
            return PythonChunker()
