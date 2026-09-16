# Use `make exports` to store model caches on sgoinfre at 42 clusters.
LOGIN = vlnikola
EXPORT_STATE := $(shell cat .export_state 2>/dev/null || echo "1")
ifeq ($(EXPORT_STATE),1)
export HF_HOME = /sgoinfre/$(LOGIN)/.cache/huggingface
export TORCH_HOME = /sgoinfre/$(LOGIN)/.cache/torch
endif

UV = uv
PYTHON = $(UV) run python

Q ?= What are the enforcement guidelines in code of conduct?
K ?= 5
CORPUS_PATH ?= data/raw
CHUNK_SIZE ?= 2000
INDEX_PATH ?= data/processed/index.pkl

all: install

install:
	$(UV) sync

run:
	$(PYTHON) -m src --help

run-index: install
	$(PYTHON) -m src index \
		--corpus_path $(CORPUS_PATH) \
		--max_chunk_size $(CHUNK_SIZE) \
		--index_path $(INDEX_PATH)

run-search-single: install
	$(PYTHON) -m src search \
		--question "$(Q)" \
		--k $(K) \
		--index_path $(INDEX_PATH)

run-answer-single: install
	$(PYTHON) -m src answer \
		--question "$(Q)" \
		--k $(K) \
		--index_path $(INDEX_PATH)

debug:
	$(PYTHON) -m pdb src/__main__.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .pytest_cache

lint:
	$(UV) run flake8 .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict

exports:
	@if [ "$(EXPORT_STATE)" = "1" ]; then \
		echo "0" > .export_state; \
	else \
		echo "1" > .export_state; \
	fi

.PHONY: all install run run-index run-search-single run-answer-single debug \
	clean lint lint-strict exports
