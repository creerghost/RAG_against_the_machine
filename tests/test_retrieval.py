import pytest
from pathlib import Path
import pickle
# pyrefly: ignore [missing-import]
from src.indexing.bm25_indexer import BM25Indexer
# pyrefly: ignore [missing-import]
from src.indexing.bm25_retriever import BM25Retriever
# pyrefly: ignore [missing-import]
from src.models import MinimalSource


@pytest.fixture
def mock_corpus(tmp_path: Path) -> Path:
    # Create a temporary corpus
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    # Create a Python file
    py_file = corpus_dir / "test.py"
    py_file.write_text("def hello_world():\n    print('hello RAG')\n")

    # Create a Markdown file
    md_file = corpus_dir / "README.md"
    md_file.write_text("# Documentation\nThis is a RAG test"
                       " documentation file.")

    # Create an invalid file
    txt_file = corpus_dir / "test.txt"
    txt_file.write_text("Should be ignored.")

    return corpus_dir


@pytest.fixture
def mock_index_file(mock_corpus: Path, tmp_path: Path) -> Path:
    indexer = BM25Indexer()
    indexer.index(str(mock_corpus), max_chunk_size=100)
    index_path = tmp_path / "index.pkl"
    indexer.save(str(index_path))
    return index_path


def test_indexer_processes_only_valid_files(mock_corpus: Path):
    indexer = BM25Indexer()
    indexer.index(str(mock_corpus), max_chunk_size=100)
    assert len(indexer.chunks) > 0
    assert indexer.bm25 is not None


def test_indexer_saves_pickle_correctly(mock_corpus: Path, tmp_path: Path):
    indexer = BM25Indexer()
    indexer.index(str(mock_corpus), max_chunk_size=100)
    index_path = tmp_path / "index.pkl"
    indexer.save(str(index_path))
    assert index_path.exists()
    with open(index_path, "rb") as f:
        content = pickle.load(f)
    assert "chunks" in content
    assert "bm25" in content


def test_retriever_loads_pickle(mock_index_file: Path):
    retriever = BM25Retriever()
    retriever.load(str(mock_index_file))
    assert len(retriever.chunks) > 0
    assert retriever.bm25 is not None


def test_retriever_returns_minimal_sources(mock_index_file: Path):
    retriever = BM25Retriever()
    retriever.load(str(mock_index_file))
    query = "Documentation"
    results = retriever.search(query, k=1)
    assert len(results) == 1
    assert isinstance(results[0], MinimalSource)
    assert results[0].file_path.endswith("README.md")
