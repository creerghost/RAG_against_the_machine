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