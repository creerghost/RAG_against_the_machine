*This project has been created as part of the 42 curriculum by vlnikola.*

# RAG Against the Machine

## Description

This project implements a Retrieval-Augmented Generation (RAG) system for
answering questions about the vLLM codebase. It indexes source files, retrieves
the most relevant code or documentation snippets with BM25, and gives those
snippets to the local `Qwen/Qwen3-0.6B` model to produce a grounded answer.

## Instructions

The project uses uv for dependency management. From the repository root:

```bash
uv sync
```

The evaluator-compatible commands are:

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How do I configure vLLM?" --k 5
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src answer "How do I configure vLLM?" --k 5
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

All paths can be overridden with CLI flags. The default index is written to
`data/processed/index.pkl`, search results to `data/output/search_results`,
and answer results to `data/output/search_results_and_answer`.

## System architecture

The pipeline has five stages:

1. The indexer walks the configured corpus and routes Python files to the AST
   chunker and Markdown/text files to the document chunker.
2. The chunkers emit Pydantic `Chunk` objects containing source text and exact
   character offsets.
3. The BM25 index stores normalized lexical tokens and the corresponding
   chunks in a pickle file.
4. The retriever ranks chunks for a question and returns the minimal source
   locations required by the evaluator.
5. The generator reads those ranges and asks Qwen to answer only from the
   retrieved context.

Dataset search and answer commands serialize their results through the
Pydantic output models.

## Chunking strategy

Python files are parsed with `ast` and are initially segmented at top-level
syntax nodes. Markdown files are segmented at headings; plain text and any
oversized section use the same bounded character fallback. The fallback keeps
contiguous source ranges and guarantees that every chunk is no longer than
`--max_chunk_size` (2,000 by default). `.py`, `.md`, `.txt`, and `.text` files
are supported.

## Retrieval method

BM25Okapi provides lexical ranking. Text is lowercased and tokenized into
prose terms, identifier terms, and underscore-separated identifier components.
This lets a natural-language question match both phrases such as “batched
MoE” and identifiers such as `fused_batched_moe`. Results retain BM25 ranking
order and return file paths plus half-open character spans.

## Performance analysis

The assignment requires indexing within five minutes, processing 200 queries
within 90 seconds, and reaching Recall@5 of 80% for documentation and 50% for
code. Use the `evaluate` command to measure Recall@1, @3, @5, and @10 against
an AnsweredQuestions dataset. The implementation keeps indexing and retrieval
in-memory during each command so the expensive corpus scan happens once and
batch search can reuse the persisted BM25 index.

## Design decisions

- BM25 is lightweight, deterministic, and appropriate for exact code symbols.
- Pydantic models make source spans and JSON boundaries explicit.
- Character offsets are preserved instead of reconstructing text from tokens.
- The default paths match the defense walkthrough, while flags keep local
  experiments reproducible.
- Qwen is loaded lazily only when an answer has retrieved context.

## Challenges faced

The main challenges are matching natural-language questions to code symbols,
keeping source paths byte-for-byte compatible with the evaluator, and splitting
large AST nodes without returning invalid spans. Shared tokenization improves
the first problem, explicit path handling solves the second, and bounded
fallback splitting solves the third.

## Example usage

Index a local corpus at a custom location:

```bash
uv run python -m src index \
  --corpus_path data/raw \
  --max_chunk_size 1500 \
  --index_path data/processed/index-1500.pkl
```

Search that index and save a dataset result elsewhere:

```bash
uv run python -m src search_dataset \
  --dataset_path datasets_public/public/UnansweredQuestions/dataset_code_public.json \
  --k 10 \
  --index_path data/processed/index-1500.pkl \
  --save_directory /tmp/rag-search
```

## Resources

- [Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [rank-bm25 documentation](https://github.com/dorianbrown/rank_bm25)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Python `ast` documentation](https://docs.python.org/3/library/ast.html)
- [Qwen3 model documentation](https://huggingface.co/Qwen/Qwen3-0.6B)

AI was used to help inspect the assignment requirements, identify edge cases,
suggest test cases, and review the implementation. All generated suggestions
were tested against the real code and adapted manually.
