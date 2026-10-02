"""Tests for the corpus loader."""

from pathlib import Path

import pytest

from src.config import RagConfig
from src.loader import CorpusLoader

RAW = Path("data/raw")


def make_loader(root: Path) -> CorpusLoader:
    """Return a loader over ``root`` with the configured extensions."""
    return CorpusLoader(str(root), RagConfig().chunker_by_extension)


def test_filters_sorts_and_skips(tmp_path: Path) -> None:
    """Only wanted, readable, non-blank files come back, sorted as strings."""
    (tmp_path / "a-b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "a-b" / "x.md").write_text("doc")
    (tmp_path / "a" / "y.PY").write_text("code")
    (tmp_path / "a" / "img.png").write_bytes(b"\x89PNG")
    (tmp_path / "a" / "empty.py").write_text("")
    (tmp_path / "a" / "blank.md").write_text("\n \n")
    (tmp_path / "a" / "bad.txt").write_bytes(b"\xff\xfe bad")
    pairs = make_loader(tmp_path).load()
    root = str(tmp_path)
    assert pairs == [(f"{root}/a-b/x.md", "doc"), (f"{root}/a/y.PY", "code")]


def test_missing_root(tmp_path: Path) -> None:
    """A missing raw_dir gives no files instead of an error."""
    assert make_loader(tmp_path / "nope").load() == []


def test_read_failures(tmp_path: Path) -> None:
    """Missing files and directories read as None."""
    loader = make_loader(tmp_path)
    assert loader.read(str(tmp_path / "missing.md")) is None
    assert loader.read(str(tmp_path)) is None


@pytest.mark.skipif(not RAW.is_dir(), reason="corpus not present")
def test_real_corpus() -> None:
    """The vLLM corpus loads with grader-style relative paths."""
    loader = make_loader(RAW)
    paths = loader.paths()
    assert len(paths) == 1969
    assert all(p.startswith("data/raw/vllm-0.10.1/") for p in paths)
    assert "data/raw/vllm-0.10.1/CMakeLists.txt" in paths
    lora = "data/raw/vllm-0.10.1/docs/features/lora.md"
    assert len(loader.read(lora) or "") == 15148
