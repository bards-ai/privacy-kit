# Changelog

## Unreleased

### Added

- **Deterministic secret & checksum detection.** Detection is no longer
  model-only: a deterministic layer now always runs alongside the NER model
  (`build_detector("local")` returns the composite; `"regex"` builds the
  deterministic layer alone — instant startup, no model download; `"model"`
  is the NER model alone).
  - `SecretDetector` (`SECRET_*` labels): AWS/GCP/GitHub/GitLab/OpenAI/
    Anthropic/Hugging Face/Slack/Stripe/SendGrid/npm/PyPI keys and tokens,
    JWTs, PEM private-key blocks, credentials in connection URLs (masks only
    the password), `Authorization:` header values, and generic
    `API_KEY=…`-style assignments gated by a Shannon-entropy floor. Patterns
    are vendored from the gitleaks ruleset (MIT) as a curated in-tree table;
    `make sync-secret-rules` diffs them against upstream.
  - `ChecksumPiiDetector`: payment cards (Luhn), IBANs (mod-97),
    international `+…` phones, dashed US SSNs — emitting the model's own
    labels so Vault placeholders unify across both paths.
  - `CompositeDetector` unions spans; on overlap, `SECRET_*` wins over any
    model span, then the wider span, then the higher score.
  - The gateway and `privacy-kit scan` use the composite automatically: a
    `.env` file in a proxied tool read is now caught (audited in `monitor`,
    replaced in `pseudonymize`) even though the NER model has no label for
    credentials.
- **Per-entity policy overrides** (`PII_POLICY_OVERRIDES`, JSON): layer
  per-type actions over the global `PII_POLICY` mode. Keys are entity labels
  or prefix wildcards (`SECRET_*`); actions are `keep`, `redact` (one-way
  `[REDACTED]`, never rehydrated), `pseudonymize` (reversible `[TYPE_N]`),
  and `block` — the request is refused with a 403 (`privacy_kit_blocked`,
  with entity counts) before anything reaches the upstream, and the audit row
  records `policy="block"`. Overrides apply in both modes, so
  `{"SECRET_*": "block"}` stops credentials even under `monitor`. Cursor
  hooks honor `block`-typed entities too (deny), alongside the existing
  blunter `PII_CURSOR_BLOCK`. Exact label beats wildcard; longest wildcard
  prefix wins.
- Behavior note: under `monitor` with no overrides, the response
  de-anonymization vault now stays empty (the upstream saw originals, so no
  placeholder can legitimately come back). Previously monitor mode populated
  the vault as a side effect and would rehydrate placeholder-shaped text
  echoed by the upstream.

### Changed

- **Audit DB default location & permissions**: `PII_DB_PATH` now defaults to a
  stable per-user path `~/.privacy-kit/audit.sqlite` (was `privacy_kit.sqlite`
  relative to the working directory, which scattered stray DBs and started a
  fresh empty log when launched from a different directory). The file is created
  `0o600` (owner-only) since it holds plaintext PII, and its parent directory is
  created on demand. Set `PII_DB_PATH` to override; the Docker image still uses
  `/data/privacy_kit.sqlite`. Existing logs at the old path are not migrated —
  move the file or set `PII_DB_PATH` to keep prior history.

### Added

- **Saved request texts**: the audit DB gains an `interactiontext` table
  storing, per proxied request, each text segment's original and anonymized
  form (plaintext, linked to the interaction). `PII_SAVE_TEXTS` picks the
  scope: `all` (default — every user/tool segment) or `anonymized` (only
  segments where PII was replaced). This is a deliberate change from the
  previous metadata-only design: the local SQLite file now contains raw
  originals of user/tool segments by default. The OTel sink and `count_tokens` remain
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
