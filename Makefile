.DEFAULT_GOAL := help
.PHONY: help install install-all check lint type test test-model fmt \
	setup run up down logs serve serve-nomodel dev dev-nomodel web-install web-dev web-build \
	route route-remove route-claude-code route-codex route-cursor \
	route-claude-code-remove route-codex-remove route-cursor-remove \
	clean docker-build docker-run

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2}'

# --- Docker stack: gateway (api) + dashboard (web) -------------------------

setup: ## Build the Docker images (gateway + dashboard)
	docker compose build

run: ## Start the stack — dashboard on http://127.0.0.1:3000, gateway on :8787
	@echo "Dashboard → http://127.0.0.1:3000   Gateway → http://127.0.0.1:8787"
	docker compose up -d

down: ## Stop the stack
	docker compose down

logs: ## Tail logs from the running stack
	docker compose logs -f

# --- Local dev without Docker ----------------------------------------------

serve: ## Run just the gateway locally (loads the on-device model)
	uv run privacy-kit serve

serve-nomodel: ## Run the gateway with detection off (no model download; saves conversations only)
	PII_DETECTOR=null uv run privacy-kit serve

web-install: ## Install the dashboard's npm dependencies
	npm --prefix web install

web-dev: ## Run the dashboard in dev mode (expects the gateway on :8787)
	npm --prefix web run dev

web-build: ## Build the dashboard for production
	npm --prefix web run build

dev: ## Run gateway + dashboard locally together (Ctrl-C stops both)
	@echo "Starting gateway (:8787) and dashboard (:3000)…"
	@trap 'kill 0' INT TERM; \
		uv run privacy-kit serve & \
		API_URL=http://127.0.0.1:8787 npm --prefix web run dev & \
		wait

dev-nomodel: ## Like `dev` but with detection off (no model; saves conversations only)
	@echo "Starting gateway (:8787, detector=null) and dashboard (:3000)…"
	@trap 'kill 0' INT TERM; \
		PII_DETECTOR=null uv run privacy-kit serve & \
		API_URL=http://127.0.0.1:8787 npm --prefix web run dev & \
		wait

# --- Python backend tooling -------------------------------------------------

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

fmt: ## Auto-format and auto-fix
	uv run ruff format .
	uv run ruff check --fix .

# --- Route AI tools through the gateway (edits their own config files) ------

route: route-claude-code route-codex route-cursor ## Route Claude Code + Codex + Cursor through the gateway

route-claude-code: ## Route Claude Code through the gateway (edits ~/.claude/settings.json)
	uv run privacy-kit setup claude-code --apply

route-codex: ## Route Codex through the gateway (edits ~/.codex/config.toml)
	uv run privacy-kit setup codex --apply

route-cursor: ## Install Cursor hooks to audit Composer/agent (edits ~/.cursor/hooks.json)
	uv run privacy-kit setup cursor --apply

route-remove: route-claude-code-remove route-codex-remove route-cursor-remove ## Undo: stop routing tools through the gateway

route-claude-code-remove: ## Undo the Claude Code routing
	uv run privacy-kit setup claude-code --remove

route-codex-remove: ## Undo the Codex routing
	uv run privacy-kit setup codex --remove

route-cursor-remove: ## Remove the Cursor hooks
	uv run privacy-kit setup cursor --remove

# --- Misc -------------------------------------------------------------------

docker-build: ## Build only the gateway image (ONNX, model baked in)
	docker build -t privacy-kit .

docker-run: ## Run only the gateway container (port published to localhost)
	docker run --rm -p 127.0.0.1:8787:8787 -v privacy-kit-data:/data privacy-kit

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
