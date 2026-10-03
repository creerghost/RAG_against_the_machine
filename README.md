# RAG_against_the_machine

# Resources

https://www.youtube.com/watch?v=swvzKSOEluc&t=3060s
slack 42born2code
https://www.youtube.com/watch?v=Ub3GoFaUcds&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy
https://www.youtube.com/watch?v=hiJcEaiuw_E
https://www.youtube.com/watch?v=ruBm9WywevM


 iou = intersection / union (reference vs my result)
 recall vs precision:
 answer is in 1 place and i returned 5 chunks. one of them
 is the right one.
 recall = 1 / 1 = 100%, precision = 1 / 5 = 20%
 recall = i didn't miss result. precision = 4 results are noise
 for llm, low recall will matter more



## Run rag with:

### Code dataset

```bash
uv run python -m src index

uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json
```

### Docs dataset

```bash
uv run python -m src index

uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```
## Performance analysis

Recall@k on the public datasets (`evaluate`, identical to the moulinette),
`--max_chunk_size 2000`, BM25 `k1 = 1.5`, `b = 0.75`. Each row adds one change
to the row above. Required: recall@5 ≥ 0.80 (docs) and ≥ 0.50 (code).

| # | Change | Chunks | docs @1 | @3 | @5 | @10 | code @1 | @3 | @5 | @10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Baseline: Markdown chunker, `.py` as plain text, identifier-aware tokenizer | 13,478 | 0.630 | 0.800 | 0.860 | 0.870 | 0.404 | 0.616 | 0.707 | 0.737 |
| 2 | + chunk title and file path tokens indexed | 13,478 | 0.680 | 0.830 | 0.870 | 0.890 | 0.545 | 0.717 | 0.808 | 0.879 |
| 3 | + `PythonChunker`: sections at top-level `def`/`class` (`ast`) | 19,601 | 0.660 | 0.830 | 0.870 | 0.880 | 0.596 | 0.758 | 0.859 | 0.899 |
| 4 | + big classes cut at methods (`Class.method` titles), small neighbours packed up to the limit | 15,559 | 0.640 | 0.840 | 0.870 | 0.890 | 0.616 | 0.828 | 0.879 | 0.919 |

Notes:

- One question ≈ 1 point (99–100 questions per set): differences under ~3
  points may be noise. Row 3's docs @1 dip (0.68 → 0.66) is 2 questions.
- Title/path tokens are the biggest single gain: questions name the class,
  function or file (`mamba_mixer2.py`, "the `LLM` class") that the answering
  chunk's text often does not contain.
- A throwaway prototype before row 3 measured one-chunk-per-function *without*
  packing small neighbours: code @5 dropped to 0.65 (twice as many tiny
  chunks). Packing small neighbours (row 4) fixes that: fewer, fuller chunks
  with clean boundaries and `Class.method` titles.

Timing (whole corpus, CPU): `index` ≈ 4 s (limit 5 min); `search_dataset`
on 100 questions ≈ 0.3 s (limit 90 s for 200).
