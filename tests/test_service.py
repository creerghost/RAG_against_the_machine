"""Tests for RagService: lazy loading and index wiring."""

from pathlib import Path

import pytest

from src.config import PathsConfig, RagConfig
from src.models import MinimalSource
from src.service import RagService


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> RagConfig:
    """Return a config over a two-file corpus in ``tmp_path`` (no .env)."""
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "lora.md").write_text("# LoRA\n\nLoad a LoRA adapter at runtime.\n")
    (raw / "sched.py").write_text("def schedule():\n    return 'batch'\n")
    paths = PathsConfig(raw_dir=str(raw),
                        processed_dir=str(tmp_path / "processed"))
    return RagConfig(paths=paths)


def test_answer_without_sources_skips_model(config: RagConfig) -> None:
    """No sources: the fixed reply, and no model is loaded."""
    service = RagService(config)
    assert service.answer("anything?", []) == config.generation.no_answer
    assert service._generator is None


def test_search_before_index_fails(config: RagConfig) -> None:
    """Searching without an index raises FileNotFoundError (CLI maps it)."""
    with pytest.raises(FileNotFoundError):
        RagService(config).search("lora", 3)


def test_index_then_search(config: RagConfig) -> None:
    """index builds the files; search finds the matching chunk."""
    service = RagService(config)
    assert service.index(semantic=False) == 2
    sources = service.search("lora adapter", 3)
    assert sources and sources[0].file_path.endswith("lora.md")
    assert all(isinstance(s, MinimalSource) for s in sources)


def test_index_resets_loaded_retriever(config: RagConfig) -> None:
    """A new index drops the loaded retriever, so ids are never stale."""
    service = RagService(config)
    service.index(semantic=False)
    first = service.load_index()
    service.index(semantic=False, max_chunk_size=10)
    assert service.load_index() is not first
    assert len(service.load_index().chunks) > len(first.chunks)
