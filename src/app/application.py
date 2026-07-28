import sys


class Application:
    def __init__(self):
        pass

    def run(self):
        try:
            from ..chunking import MarkdownChunker
            file_name = "README.md"
            with open(file_name, "r") as f:
                text = f.read()
            chunker = MarkdownChunker()
            chunker.chunk(text, file_name, 2000)
            print(chunker)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
