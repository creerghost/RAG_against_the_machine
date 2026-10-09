"""All tunable settings in one place; each field says which files use it."""

from typing import Literal

from pydantic import BaseModel, Field

# Ranker names accepted by --mode and ranking.ranker_for.
RankMode = Literal["lexical", "semantic", "hybrid"]


class RagConfig(BaseModel):
    """Project settings; CLI flags override the defaults."""
    # Corpus root to index. Used in: cli.py, index.py, loader.py (via index).
    raw_dir: str = "data/raw"
    # Where the index is written and read. Used in: cli.py, index.py,
    # retriever.py.
    processed_dir: str = "data/processed"
    # Hugging Face id of the answer model. Used in: cli.py, generator.py.
    model_name: str = "Qwen/Qwen3-0.6B"
    # Weight precision of the answer model. bfloat16 is ~1.8x faster on CPUs
    # with native bf16 (AVX512_BF16 / AMX) but can be slower without it.
    # Used in: cli.py, generator.py.
    model_dtype: Literal["float32", "bfloat16"] = "float32"
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
    # Hugging Face id of the sentence embedding model (loaded with plain
    # transformers, 384-dim vectors). Used in: embedder.py (via index.py,
    # ranking/__init__.py).
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Tokens per text fed to the embedder; longer chunks are truncated
    # (MiniLM was trained on 256). Used in: embedder.py.
    embedding_max_length: int = Field(default=256, ge=1)
    # Texts embedded per forward pass; higher is faster until memory runs
    # out. Used in: embedder.py (via index.py).
    embedding_batch_size: int = Field(default=32, ge=1)
    # Weight precision of the embedder; bfloat16 is faster on CPUs with
    # native bf16, like model_dtype. Used in: embedder.py (via index.py,
    # ranking/__init__.py).
    embedding_dtype: Literal["float32", "bfloat16"] = "float32"
    # Which ranker scores chunks: BM25 only, embeddings only (needs
    # ``index --semantic True``), or both fused by RRF. Used in: cli.py,
    # ranking/__init__.py.
    mode: RankMode = "lexical"
    # Reciprocal rank fusion constant: score = sum of 1 / (rrf_k + rank);
    # larger values flatten the gap between top ranks. Used in:
    # ranking/__init__.py (via HybridRanker).
    rrf_k: int = Field(default=60, ge=1)
    # Chunks each ranker returns before fusion (at least k); a deeper pool
    # lets a chunk ranked low by one ranker still collect its vote. Used in:
    # ranking/__init__.py (via HybridRanker).
    rrf_candidates: int = Field(default=50, ge=1)
    # Minimum IoU for a retrieved span to count as finding a reference
    # (moulinette rule). Used in: evaluate.py (via cli.py).
    min_iou: float = Field(default=0.05, ge=0, le=1)
    # Token budget for the numbered sources in a prompt; lower is faster
    # (reading the prompt is ~40% of generation time). Used in:
    # generator.py.
    max_context_tokens: int = Field(default=1500, ge=1)
    # Upper bound on answer length in tokens. Used in: generator.py.
    max_new_tokens: int = Field(default=200, ge=1)
    # Prompt lookup decoding: the model drafts up to this many next tokens by
    # matching the end of its answer against the prompt (answers copy names
    # from the sources) and verifies them in one pass. Greedy output is
    # unchanged; ~20% faster with bfloat16. 0 disables. Used in:
    # generator.py.
    prompt_lookup_tokens: int = Field(default=10, ge=0)
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
        "- End the answer with the number of the source you used in "
        "brackets, for example: … is /v1/load_lora_adapter. [1]")
