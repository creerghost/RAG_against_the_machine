"""Tests for the shared helpers in entrypoints/utils."""

import pytest

from src.entrypoints.utils import index_error, refuse_global_flags
from src.io_utils import CliError


@pytest.mark.parametrize("args", [["--mode", "hybrid", "search", "x"],
                                  ["--model_dtype=bfloat16", "answer", "x"]])
def test_flags_before_command_refused(args: list[str]) -> None:
    """Settings given as options before the command are an error."""
    with pytest.raises(CliError, match="before the command"):
        refuse_global_flags(args)


@pytest.mark.parametrize("args", [["search", "x", "--k", "3"], ["--help"],
                                  []])
def test_command_options_allowed(args: list[str]) -> None:
    """Options after the command, help and no arguments pass through."""
    refuse_global_flags(args)


def test_index_error_missing_index() -> None:
    """A missing index file names the directory and the index command."""
    error = FileNotFoundError(2, "No such file", "out/bm25.npz")
    message = index_error(error, "out", "lexical")
    assert message.startswith("No index in out") and "'index' first" in message


def test_index_error_missing_embeddings() -> None:
    """Hybrid without embeddings asks for ``index --semantic``."""
    error = FileNotFoundError(2, "No such file", "out/embeddings.npy")
    message = index_error(error, "out", "hybrid")
    assert "RAG_RETRIEVAL__MODE=hybrid needs embeddings" in message
    assert "'index --semantic' first" in message


def test_index_error_unreadable() -> None:
    """Any other load error is reported as an unreadable index."""
    message = index_error(ValueError("bad rows"), "out", "lexical")
    assert "unreadable (bad rows)" in message and "again" in message
