# Changelog

## Unreleased

### Added

- **Saved request texts**: the audit DB gains an `interactiontext` table
  storing, per proxied request, each text segment's original and anonymized
  form (plaintext, linked to the interaction). `PII_SAVE_TEXTS` picks the
  scope: `anonymized` (default — only segments where PII was replaced) or
  `all` (every segment). This is a deliberate change from the previous
  metadata-only design: the local SQLite file now contains raw originals of
  PII-bearing segments by default. The OTel sink and `count_tokens` remain
  text-free; model output is never stored. Existing DB files get the new
  table automatically on next start.
- **Codex with a ChatGPT-account login** (free/Plus/Pro, no API key) can now be
  routed through the gateway (experimental). Codex sends its model call to
  `openai_base_url` in both auth modes; the gateway recognizes a subscription
  request by its `chatgpt-account-id` header and forwards it to chatgpt.com's
  backend (`/codex/responses`) with the login token untouched, while API-key
  requests still go to api.openai.com. `privacy-kit setup codex --apply` /
  `--remove` writes/clears a single `openai_base_url` in `~/.codex/config.toml`.
- The gateway decodes `zstd`- and `brotli`-compressed request bodies (Codex
  sends zstd streaming frames); `zstandard` + `brotli` added to the `[gateway]`
  extra. Body-decode failures now return a 400 naming the cause.

### Fixed

- `setup codex --apply` inserts `openai_base_url` into the config root instead of
  appending it after a trailing `[table]` (which TOML read as a member of that
  table and broke Codex's config load).

## 0.3.0 — 2026-06-10

The Sieve gateway merges into privacy-kit: one model, one core, two deployment
modes (in-process blocks and a network-path gateway).

### Added

- **Gateway** (`pip install 'privacy-kit[gateway]'`, console script
  `privacy-kit`): a local proxy for AI tools that can't import a library.
  Routes Claude Code (Anthropic Messages + count_tokens), Codex (OpenAI
  Responses), and Cursor/Codex chat (OpenAI Chat Completions) via their
  `*_BASE_URL` overrides. Pseudonymizes request text before it leaves the
  machine, forwards with the client's own auth, rehydrates responses
  (including SSE streams, with placeholders split across chunks buffered and
  restored), and records a metadata-only audit row. CLI:
  `serve` / `setup claude-code|codex|cursor` / `report` / `scan`.
- **OTLP sink** mounted on the gateway (`/v1/logs`, `/v1/traces`,
  `/v1/metrics`, OTLP/HTTP JSON): scrubs every string value one-way, audits
  logs, optionally re-exports scrubbed payloads downstream.
- **`Vault`** — reversible, idempotent pseudonymization as a core block:
  `anonymize` / `anonymize_into` / `deanonymize`, value ↔ `[TYPE_N]` mapping
  stable within a vault. Exported from the package top level.
- `Span.text_of()` helper; `Span` is now a frozen slots dataclass.
- **PII preview UI** at `/ui` on the gateway: a single-file local page (no
  external assets) with a live preview — paste text, see detected spans
  highlighted by entity type and the pseudonymized output — plus the
  metadata-only audit dashboard (totals by type, recent interactions). The
  preview runs in memory only: never stored, logged, or audited.
- **Auto-routing for Claude Code** — no manual exports:
  `privacy-kit setup claude-code --apply` writes `ANTHROPIC_BASE_URL` into the
  `env` block of `~/.claude/settings.json` (undo with `--remove`), and
  `privacy-kit serve --route claude-code` applies the override for the
  server's lifetime and restores the previous value on shutdown. The CLI
  always prints exactly which file it overrides; a value the user edited
  manually in the meantime is never clobbered. Subscription (Max/Pro OAuth)
  auth is supported — the gateway forwards the login token and preserves
  Claude Code's system identifier verbatim.
- Dockerfile (torch-free, model baked in at build time), Makefile, CI matrix
  on Python 3.10/3.12, ruff + mypy --strict gate, pre-commit config.

### Changed

- **Detector pipeline upgrade** (`BardsAiOnnxDetector`): long inputs now run
  as overlapping token windows batched through the model (8 per forward
  pass) with the most confident prediction kept per token; subword
  predictions aggregate to whole words (prevents partial-token leakage);
  consecutive same-type words merge; trailing punctuation is trimmed.
  Span boundaries and scores can differ slightly from 0.2.x on long or
  edge-case texts (verified identical on the multilingual fixture suite).
- ONNX session options now default to onnxruntime defaults (graph
  optimization on, automatic threading) instead of the pinned
  single-threaded/no-optimization configuration; tune via the new
  `intra_op_num_threads` / `graph_optimization` constructor parameters.
- The tokenizer's baked-in 512-token truncation is explicitly disabled —
  encoding never silently clips long inputs.
- All configuration consolidated under the `PII_*` env prefix (gateway
  settings included: `PII_HOST`, `PII_PORT`, `PII_DB_PATH`,
  `PII_ANTHROPIC_UPSTREAM`, `PII_OPENAI_UPSTREAM`, `PII_OTEL_DOWNSTREAM`).

### Notes for clients migrating from Sieve

- CLI `sieve …` → `privacy-kit …`; env `SIEVE_*` → `PII_*`
  (`SIEVE_SCORE_THRESHOLD` → `PII_THRESHOLD`); default audit DB file is now
  `privacy_kit.sqlite`.
- The `x-sieve-source` request header still works but is deprecated in favor
  of `x-privacy-kit-source`; it will be removed in a future release.
