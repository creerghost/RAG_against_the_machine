"""Tests for RagConfig: groups, precedence of overrides, immutability."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.config import RagConfig


def config_with(**groups: Any) -> RagConfig:
    """Return a RagConfig built from group dicts, as the CLI does."""
    return RagConfig(**groups)


def test_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Without env vars or .env, every group has its defaults."""
    monkeypatch.chdir(tmp_path)  # no .env here
    config = RagConfig()
    assert config.paths.raw_dir == "data/raw"
    assert config.bm25.k1 == 1.5
    assert config.retrieval.mode == "lexical"


def test_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Init args beat env vars, env vars beat .env; groups merge per field."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("RAG_BM25__K1=1.1\nRAG_BM25__B=0.5\n"
                                   "OTHER_TOOL=1\n")
    monkeypatch.setenv("RAG_BM25__K1", "1.2")
    config = config_with(bm25={"b": 0.3})
    assert config.bm25.k1 == 1.2  # env over .env
    assert config.bm25.b == 0.3   # init over .env


def test_invalid_and_unknown(monkeypatch: pytest.MonkeyPatch,
                             tmp_path: Path) -> None:
    """Out-of-range values and typos inside a group are rejected."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError):
        config_with(bm25={"k1": -1})
    with pytest.raises(ValidationError):
        config_with(bm25={"k2": 1})
    monkeypatch.setenv("RAG_GENERATION__DTYPE", "fp16")
    with pytest.raises(ValidationError):
        RagConfig()


def test_frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Fields cannot be reassigned; model_copy makes a changed copy."""
    monkeypatch.chdir(tmp_path)
    config = RagConfig()
    with pytest.raises(ValidationError):
        config.bm25.k1 = 2.0
    chunking = config.chunking.model_copy(update={"max_chunk_size": 500})
    copy = config.model_copy(update={"chunking": chunking})
    assert copy.chunking.max_chunk_size == 500
    assert config.chunking.max_chunk_size == 2000
