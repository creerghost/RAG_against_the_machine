# The size of HDD space in 42 clusters is very limited.
# Write 'make exports' to save caches on ~/sgoinfre folder
#	(if you are running this project on 42 prague devices).
LOGIN = vlnikola

# Read export state (defaults to 1 if file doesn't exist)
EXPORT_STATE := $(shell cat .export_state 2>/dev/null || echo "1")
ifeq ($(EXPORT_STATE),1)
export HF_HOME = /sgoinfre/$(LOGIN)/.cache/huggingface
export TORCH_HOME = /sgoinfre/$(LOGIN)/.cache/torch
endif

UV = uv
PYTHON = $(UV) run python

ARGS ?=

RESET = \033[0m
BOLD = \033[1m
RED = \033[1;31m
GREEN = \033[1;32m
YELLOW = \033[1;33m
BLUE = \033[1;34m
MAGENTA = \033[1;35m
CYAN = \033[1;36m

all: install

install:
	@printf "$(CYAN)Syncing dependencies with uv...$(RESET)\n"
	$(UV) sync

run: install
	@printf "$(MAGENTA)Running main pipeline...$(RESET)\n"
	$(PYTHON) -m src $(ARGS)

debug:
	@printf "$(YELLOW)Starting debugger...$(RESET)\n"
	$(PYTHON) -m pdb src/__main__.py

clean:
	@printf "$(RED)Cleaning all caches and environments...$(RESET)\n"
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .venv


clean-cache:
	@printf "$(RED)Cleaning caches...$(RESET)\n"
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	@printf "$(CYAN)Running standard linting...$(RESET)\n"
	$(UV) run flake8 src/ tests/
	$(UV) run mypy src/ tests/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@printf "$(CYAN)Running strict linting...$(RESET)\n"
	$(UV) run flake8 src/ tests/
	$(UV) run mypy src/ tests/ --strict

exports:
	@if [ "$(EXPORT_STATE)" = "1" ]; then \
		echo "0" > .export_state; \
		printf "$(RED)Exports toggled OFF (will not use /sgoinfre caches)$(RESET)\n"; \
	else \
		echo "1" > .export_state; \
		printf "$(GREEN)Exports toggled ON (will use /sgoinfre caches)$(RESET)\n"; \
	fi

.PHONY: all install run debug clean clean-cache lint lint-strict exports