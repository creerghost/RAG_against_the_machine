"""All tunable settings in one place, grouped by pipeline stage.

Defaults below, overridden by a ``.env`` file, then by ``RAG_*`` environment
variables (``RAG_BM25__K1=1.2``); settings are never CLI flags. Each field
says which files use it.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ranker names accepted by --mode and ranking.ranker_for.
RankMode = Literal["lexical", "semantic", "hybrid"]
# Weight precisions accepted by --model_dtype and --embedding_dtype.
DType = Literal["float32", "bfloat16"]


class Group(BaseModel):
    """Base of every settings group: immutable, unknown keys rejected."""
    model_config = ConfigDict(frozen=True, extra="forbid")


class PathsConfig(Group):
    """Where the corpus is read and the index is written.

    raw_dir: corpus root to index; via cli.py, index.py, loader.py.
    processed_dir: where the index is written and read; via cli.py,
        index.py, retriever.py, ranking/factory.py.
    """
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"


class ChunkingConfig(Group):
    """How files are cut into chunks.

    max_chunk_size: default chunk width in characters; via cli.py,
        chunking/factory.py.
    max_context_length: moulinette limit on a source's width, upper bound
        for max_chunk_size; via cli.py.
    separators: split points tried, best first, before a hard cut; via
        chunking/base.py, chunking/factory.py.
    chunker_by_extension: indexed extensions and the Chunker.kind for each;
        other files are skipped; via chunking/factory.py, index.py,
        loader.py.
    """
    max_chunk_size: int = Field(default=2000, ge=1)
    max_context_length: int = Field(default=2000, ge=1)
    separators: tuple[str, ...] = ("\n\n", "\n")
    chunker_by_extension: dict[str, str] = {
        ".md": "markdown", ".txt": "text", ".py": "python"}


class TokenizerConfig(Group):
    """How text becomes BM25 terms (same for chunks and queries).

    min_token_length: tokens shorter than this are dropped; via
        tokenizer.py, index.py, ranking/factory.py.
    stopwords: words never indexed or searched (lowercase); via
        tokenizer.py, index.py, ranking/factory.py.
    """
    min_token_length: int = Field(default=2, ge=1)
    stopwords: frozenset[str] = frozenset({
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
        "for", "from", "how", "if", "in", "is", "it", "its", "of", "on", "or",
        "that", "the", "this", "to", "was", "what", "when", "where", "which",
        "who", "why", "with"})


class BM25Config(Group):
    """BM25 scoring parameters.

    k1: term-frequency saturation (~1.2-2.0); via bm25.py, index.py.
    b: length normalization, 0 (none) to 1 (full); via bm25.py, index.py.
    """
    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)


class EmbeddingConfig(Group):
    """The sentence embedding model of semantic search (bonus).

    model: Hugging Face id, loaded with plain transformers (384-dim
        vectors); via cli.py, embedder.py, index.py, ranking/factory.py.
    max_length: tokens per text fed to the model; longer chunks are
        truncated (MiniLM was trained on 256); via embedder.py.
    batch_size: texts embedded per forward pass; higher is faster until
        memory runs out; via embedder.py, index.py.
    dtype: weight precision; bfloat16 is faster on CPUs with native bf16,
        like generation.dtype; via embedder.py, index.py,
        ranking/factory.py.
    """
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = Field(default=256, ge=1)
    batch_size: int = Field(default=32, ge=1)
    dtype: DType = "float32"


class RetrievalConfig(Group):
    """Which ranker searches, and how hybrid mode fuses rankings.

    default_k: sources per question when --k is omitted; via cli.py.
    mode: BM25 only, embeddings only (needs ``index --semantic True``), or
        both fused by RRF; via cli.py, ranking/factory.py.
    rrf_k: reciprocal rank fusion constant, score = sum of weight / (rrf_k
        + rank); larger values flatten the gap between top ranks (5 beat
        the usual 60 on the public sets); via ranking/factory.py,
        HybridRanker.
    rrf_weights: vote weight of each ranker in hybrid mode, in order
        (lexical, semantic); semantic is weaker, so it only breaks
        near-ties; via ranking/factory.py, HybridRanker.
    rrf_candidates: chunks each ranker returns before fusion (at least k);
        a deeper pool lets a chunk ranked low by one ranker still collect
        its vote; via ranking/factory.py, HybridRanker.
    """
    default_k: int = Field(default=10, ge=1)
    mode: RankMode = "lexical"
    rrf_k: int = Field(default=5, ge=1)
    rrf_weights: tuple[float, float] = (1.0, 0.5)
    rrf_candidates: int = Field(default=10, ge=1)


class GenerationConfig(Group):
    """The answer model and its prompt.

    model: Hugging Face id of the answer model; via generator.py.
    dtype: weight precision; bfloat16 is ~1.8x faster on CPUs with native
        bf16 (AVX512_BF16 / AMX) but can be slower without it; via
        generator.py.
    max_context_tokens: token budget for the numbered sources in a prompt;
        lower is faster (reading the prompt is ~40% of generation time);
        via generator.py.
    max_new_tokens: upper bound on answer length in tokens; via
        generator.py.
    prompt_lookup_tokens: prompt lookup decoding, the model drafts up to
        this many next tokens by matching the end of its answer against the
        prompt (answers copy names from the sources) and verifies them in
        one pass; greedy output is unchanged, ~20% faster with bfloat16,
        0 disables; via generator.py.
    no_answer: reply when there is nothing to answer from; via cli.py,
        generator.py.
    system_prompt: system message with the grounding rules for the answer
        model; via generator.py.
    """
    model: str = "Qwen/Qwen3-0.6B"
    dtype: DType = "float32"
    max_context_tokens: int = Field(default=1500, ge=1)
    max_new_tokens: int = Field(default=200, ge=1)
    prompt_lookup_tokens: int = Field(default=10, ge=0)
    no_answer: str = "The provided sources do not contain the answer."
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


class EvaluationConfig(Group):
    """How evaluate scores search results (the moulinette's rule).

    ks: cut-offs reported by evaluate; via cli.py.
    min_iou: minimum IoU for a retrieved span to count as finding a
        reference; via evaluate.py, cli.py.
    """
    ks: tuple[int, ...] = (1, 3, 5, 10)
    min_iou: float = Field(default=0.05, ge=0, le=1)


class ServerConfig(Group):
    """The local HTTP API (bonus).

    host: interface the server binds; 127.0.0.1 keeps it local; via cli.py.
    port: TCP port of the server; via cli.py.
    """
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class RagConfig(BaseSettings):
    """Project settings: one immutable group per pipeline stage.

    paths, chunking, tokenizer, bm25, embedding, retrieval, generation,
        evaluation, server: the groups above, each set as
        ``RAG_<GROUP>__<FIELD>``.
    model_config: unknown keys are ignored at the root only (a shared .env
        may hold other tools' keys); typos inside a group (RAG_BM25__K2)
        are still rejected.
    """
    model_config = SettingsConfigDict(
        frozen=True, extra="ignore", env_prefix="RAG_",
        env_nested_delimiter="__", env_file=".env",
        env_file_encoding="utf-8")

    paths: PathsConfig = Field(default_factory=PathsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
