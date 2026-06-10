## Repo Notes

- Product direction: `privacy-kit` is a kit of ready-to-use privacy blocks at two
  altitudes — in-process library blocks (`Redactor`, `Vault`, integrations) and an
  out-of-process gateway for tools you can't modify (Claude Code, Codex, Cursor).
  Prefer import-use-done APIs over model-wrapper-only UX.
- Two core operations, pick by destination: **redact** (one-way, `[REDACTED]`) when
  the destination stores/observes data; **pseudonymize** (`Vault`, reversible
  `[TYPE_N]`) when a live LLM loop needs referential consistency and the caller
  needs real values back.
- Default install stays lightweight and uses the ONNX backend (no torch). Heavy or
  service deps live behind extras: `[gateway]`, `[transformers]` (legacy `PiiModel`),
  `[langfuse]`, `[langchain]`.
- Privacy invariants are enforced, not aspirational: code under `src/` never
  prints/logs the text it processes (ruff T20 + tests/test_log_safety.py), the audit
  store is metadata-only (tests/test_store.py dumps every DB cell to prove it), and
  the base import pulls no optional deps (tests/test_packaging.py).
- `make check` (ruff + mypy --strict + pytest) must be green before every commit.
  Tests that download the real model are gated behind `PII_RUN_MODEL_TESTS=1`
  (`make test-model`).
- All configuration uses the `PII_*` env prefix — shared knobs (model id, cache dir,
  threshold) are read identically by the core library and the gateway settings.
- Observability integrations default to `[REDACTED]`; legacy `PiiModel.anonymize()`
  keeps label placeholders like `[PERSON_NAME]`.
- Examples live under `examples/` as repo onboarding assets, not installed package
  modules. Add examples only for integrations that are actually supported.
