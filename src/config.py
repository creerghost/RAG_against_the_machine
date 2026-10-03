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
        ".md": "markdown", ".txt": "text", ".py": "python"}
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
    # BM25 term-frequency saturation (~1.2-2.0). Used in: bm25.py (via
    # index.py).
    bm25_k1: float = Field(default=1.5, gt=0)
    # BM25 length normalization, 0 (none) to 1 (full). Used in: bm25.py
    # (via index.py).
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    # Minimum IoU for a retrieved span to count as finding a reference
    # (moulinette rule). Used in: evaluate.py (via cli.py).
    min_iou: float = Field(default=0.05, ge=0, le=1)
    # Token budget for the numbered sources in a prompt; lower is faster
    # (~8 new tok/s on CPU). Used in: generator.py.
    max_context_tokens: int = Field(default=1500, ge=1)
    # Upper bound on answer length in tokens. Used in: generator.py.
    max_new_tokens: int = Field(default=200, ge=1)
    # Reply when there is nothing to answer from. Used in: generator.py.
    no_answer: str = "The provided sources do not contain the answer."
    # System message: grounding rules for the answer model. Used in:
    # generator.py.
    system_prompt: str = (
        "You answer questions about the vLLM codebase using only the "
        "numbered sources provided by the user.\n"
        "Rules:\n"
        "- Use only facts stated in the sources. Do not use outside "
        "knowledge and do not guess.\n"
        "- If the sources do not contain the answer, reply exactly: "
        "\"The provided sources do not contain the answer.\"\n"
        "- Answer the question directly in 1 to 4 sentences.\n"
        "- Copy names exactly as written in the sources: functions, "
        "classes, parameters, flags, endpoints, environment variables "
        "and default values.\n"
        "- Mention the number of the source you used, like [1].")
