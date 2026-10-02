"""Tests for the tokenizer."""

import pytest

from src.config import RagConfig
from src.tokenizer import Tokenizer


def tok(text: str) -> list[str]:
    """Tokenize with the configured stopwords and minimum length."""
    config = RagConfig()
    return Tokenizer(config.stopwords, config.min_token_length).tokenize(text)


@pytest.mark.parametrize("text, expected", [
    ("QKVParallelLinear", ["qkv", "parallel", "linear", "qkvparallellinear"]),
    ("CUDAGraphMode", ["cuda", "graph", "mode", "cudagraphmode"]),
    ("Qwen3ForCausalLM", ["qwen3", "causal", "lm", "qwen3forcausallm"]),
    ("FP8_MAX", ["fp8", "max", "fp8_max"]),
    ("MultiModal_data", ["multi", "modal", "data", "multimodal_data"]),
    ("__init__", ["init", "__init__"]),
    ("trust_remote_code", ["trust", "remote", "code", "trust_remote_code"]),
    ("hello", ["hello"]),
])
def test_identifiers(text: str, expected: list[str]) -> None:
    """Identifiers yield their parts, then the whole lowercased token."""
    assert tok(text) == expected


def test_filters() -> None:
    """Stopwords (any case), 1-char tokens and pure underscores are gone."""
    assert tok("The What is a x 2 ___ _") == []


def test_degenerate_queries() -> None:
    """Empty and punctuation-only text give no tokens."""
    assert tok("") == []
    assert tok("   ???  !!") == []


def test_counts_kept() -> None:
    """Repeated words stay repeated: BM25 needs term frequencies."""
    assert tok("lora adapter lora") == ["lora", "adapter", "lora"]


def test_query_and_chunk_agree() -> None:
    """A question's identifier yields the same tokens as the code."""
    assert set(tok("LoRAConfig")) <= set(tok("class LoRAConfig:"))
