# RAG Against The Machine

## Pipeline Commands

### 1. Indexing
Before you can search or generate answers, you must index your corpus. This parses all `.py` and `.md` files, chunks them, and builds a BM25 index (`index.pkl`).

```bash
# Example: Indexing your own source code
uv run python -m src index \
  --corpus_path src/ \
  --max_chunk_size 2000
```

### 2. Search (Single Query)
Search the indexed corpus for the top `k` relevant chunks.
```bash
uv run python -m src search \
  --question "How does pipeline work?" \
  --k 5
```

### 3. Answer (Single Query)
Retrieve relevant chunks and use Qwen to generate a natural-language answer.
```bash
uv run python -m src answer \
  --question "How does pipeline work?" \
  --k 5
```

### 4. Search Dataset (Batch)
Run a batch search for a JSON dataset of questions and save the retrieved sources.
```bash
uv run python -m src search_dataset \
  --dataset_path datasets_public/public/UnansweredQuestions/dataset_code_public.json \
  --k 10 \
  --save_directory output/
```

### 5. Answer Dataset (Batch)
Generate answers for the batch search results using Qwen.
```bash
uv run python -m src answer_dataset \
  --student_search_results_path output/dataset_code_public.json \
  --save_directory output/answers/
```

### 6. Evaluate
Use Moulinette to grade your answers against the ground truth.
```bash
./moulinette evaluate_student_answers \
  output/answers/dataset_code_public.json \
  datasets_public/public/AnsweredQuestions/dataset_code_public.json
```

# Resources

https://google.github.io/python-fire/guide/

https://www.youtube.com/watch?v=swvzKSOEluc&list=LL&index=2

https://www.geeksforgeeks.org/nlp/what-is-bm25-best-matching-25-algorithm/

https://www.geeksforgeeks.org/machine-learning/understanding-tf-idf-term-frequency-inverse-document-frequency/