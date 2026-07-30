# pyrefly: ignore [missing-import]
from src.chunking.python_chunker import PythonChunker


def test_python_chunking() -> None:
    file_name = "tests/data_tests/test_dummy.py"
    with open(file_name, "r") as f:
        text = f.read()

    chunker = PythonChunker()

    # we use a very small max_chunk_size (200 chars) to guarantee that the
    # MassiveClassToTestFallback exceeds it, forcing our chunker to dive
    # into it
    chunker.chunk(text, file_name, max_chunk_size=200)

    # 1. Assert no chunk exceeds the 200 limit
    for chunk in chunker.chunks:
        assert len(chunk.content) <= 200

    # 2. Assert the character indices accurately map to the original text
    for chunk in chunker.chunks:
        assert text[chunk.first_char_idx:chunk.last_char_idx] \
            == chunk.content

    # 3. Assert it found multiple chunks
    # (3 imports, 1 function, 3 class methods)
    assert len(chunker.chunks) >= 3

    # 4. Assert that it successfully dug into the class body and extracted
    # methods
    found_method = any("first_method" in chunk.content
                       for chunk in chunker.chunks)
    assert found_method, \
        "Chunker failed to recursively process the class body!"
