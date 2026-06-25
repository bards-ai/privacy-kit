.DEFAULT_GOAL := help
.PHONY: help install install-all check lint type test test-model fmt setup setup-claude-code setup-codex setup-cursor setup-remove setup-claude-code-remove setup-codex-remove setup-cursor-remove run serve clean docker-build docker-run

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2}'

install: ## Install the project + dev tooling (ONNX backend + gateway)
	uv sync --extra gateway

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
	PII_RUN_MODEL_TESTS=1 uv run pytest tests/test_detect_model.py tests/test_vault.py \
		tests/test_model_download.py tests/test_gateway_e2e.py tests/test_cli.py

setup: setup-claude-code setup-codex setup-cursor ## One-time: route all supported tools through the gateway (Claude Code + Codex + Cursor hooks)
	@echo "Cursor chat-panel pseudonymization is set in its Settings UI: uv run privacy-kit setup cursor"

setup-claude-code: ## Route Claude Code through the gateway (edits ~/.claude/settings.json)
	uv run privacy-kit setup claude-code --apply

setup-codex: ## Route Codex through the gateway (edits ~/.codex/config.toml)
	uv run privacy-kit setup codex --apply

setup-cursor: ## Install Cursor hooks to audit Composer/agent (edits ~/.cursor/hooks.json)
	uv run privacy-kit setup cursor --apply

setup-remove: setup-claude-code-remove setup-codex-remove setup-cursor-remove ## Undo: stop routing all supported tools through the gateway

setup-claude-code-remove: ## Undo the Claude Code routing
	uv run privacy-kit setup claude-code --remove

setup-codex-remove: ## Undo the Codex routing
	uv run privacy-kit setup codex --remove

setup-cursor-remove: ## Remove the Cursor hooks
	uv run privacy-kit setup cursor --remove

run: install serve ## One-liner: install deps then run the gateway proxy

serve: ## Run the gateway proxy locally
	uv run privacy-kit serve

docker-build: ## Build the container image (ONNX only, model baked in)
	docker build -t privacy-kit .

docker-run: ## Serve the gateway from the container, port published to localhost only
	docker run --rm -p 127.0.0.1:8787:8787 -v privacy-kit-data:/data privacy-kit

fmt: ## Auto-format and auto-fix
	uv run ruff format .
	uv run ruff check --fix .

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
