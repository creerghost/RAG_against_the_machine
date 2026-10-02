"""All tunable settings in one place; each field says which files use it."""

from pydantic import BaseModel, Field


class RagConfig(BaseModel):
    """Project settings; CLI flags override the defaults."""
    # Corpus root to index. Used in: cli.py, index.py, loader.py (via index).
    raw_dir: str = "data/raw"
    # Where the index is written and read. Used in: cli.py, index.py,
    # retriever.py.
    processed_dir: str = "data/processed"
    # Hugging Face id of the answer model. Used in: cli.py, generator.py.
    model_name: str = "Qwen/Qwen3-0.6B"
    # Default chunk width in characters. Used in: cli.py,
    # chunking/__init__.py.
    max_chunk_size: int = Field(default=2000, ge=1)
    # Moulinette limit on a source's width, upper bound for max_chunk_size.
    # Used in: cli.py.
    max_context_length: int = Field(default=2000, ge=1)
    # Sources per question when --k is omitted. Used in: cli.py.
    default_k: int = Field(default=10, ge=1)
    # Cut-offs reported by evaluate. Used in: cli.py.
    eval_ks: tuple[int, ...] = (1, 3, 5, 10)
    # Split points tried, best first, before a hard cut. Used in:
    # chunking/base.py, chunking/__init__.py.
    separators: tuple[str, ...] = ("\n\n", "\n")
    # Indexed extensions and the Chunker.kind for each; other files are
    # skipped. Used in: chunking/__init__.py, loader.py (via index).
    chunker_by_extension: dict[str, str] = {
        ".md": "markdown", ".txt": "text", ".py": "text"}
    # Tokens shorter than this are dropped (index and query). Used in:
    # tokenizer.py (via index.py, retriever.py).
    min_token_length: int = Field(default=2, ge=1)
    # Words never indexed or searched (lowercase). Used in: tokenizer.py
    # (via index.py, retriever.py).
    stopwords: frozenset[str] = frozenset({
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
        "for", "from", "how", "if", "in", "is", "it", "its", "of", "on", "or",
        "that", "the", "this", "to", "was", "what", "when", "where", "which",
        "who", "why", "with"})
