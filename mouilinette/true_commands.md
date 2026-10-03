## Index
```bash
uv run python -m src index
```

## Search dataset

### Search docs

```bash
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
```

### Search code

```bash
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
```

## Score results

### Score docs

```bash
./mouilinette/moulinette-ubuntu evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000
```

### Score code

```bash
./mouilinette/moulinette-ubuntu evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json --k 10 --max_context_length 2000
```
