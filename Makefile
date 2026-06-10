.DEFAULT_GOAL := help
.PHONY: help install install-all check lint type test test-model fmt clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the project + dev tooling (lightweight ONNX backend)
	uv sync

install-all: ## Install with every optional extra (transformers, langfuse, langchain)
	uv sync --all-extras

check: lint type test ## Run the full gate: lint + type + test (run before every commit)

lint: ## Lint with ruff
	uv run ruff check .
	uv run ruff format --check .

type: ## Type-check with mypy
	uv run mypy

test: ## Run the test suite (model-download tests are skipped unless PII_RUN_MODEL_TESTS=1)
	uv run pytest

test-model: ## Run the tests that download and exercise the real model
	PII_RUN_MODEL_TESTS=1 uv run pytest tests/test_detect_model.py tests/test_vault.py tests/test_model_download.py

fmt: ## Auto-format and auto-fix
	uv run ruff format .
	uv run ruff check --fix .

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
