# pyrefly: ignore [missing-import]
from src import MarkdownChunker


def test_chunking() -> None:
    file_name = "tests/data/test_readme.md"
    with open(file_name, "r") as f:
        text = f.read()
    chunker = MarkdownChunker()
    chunker.chunk(text, file_name, 2000)
    # 1. Assert no chunk exceeds the 2000 max_chunk_size limit
    for chunk in chunker.chunks:
        assert len(chunk.content) <= 2000

    # 2. Assert the character indices accurately map to the original text
    for chunk in chunker.chunks:
        assert text[chunk.first_char_idx:
                    chunk.last_char_idx] == chunk.content

    # 3. Assert the expected chunk count (test_readme.md has 4 headers)
    assert len(chunker.chunks) == 4

    # 4. Assert content extraction is working
    assert chunker.chunks[0].content.startswith("# Project Title")
