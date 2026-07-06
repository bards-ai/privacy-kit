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
  prints/logs the text it processes (ruff T20 + tests/test_log_safety.py), raw PII
  lives only in the audit store's dedicated `interactiontext` table — scoped by
  `PII_SAVE_TEXTS` and limited to user-authored text and tool/file data; it appears
  in no other table, log, or output (tests/test_store.py and tests/test_gateway_e2e.py
  dump every other DB cell to prove it), and the base import pulls no optional deps
  (tests/test_packaging.py).
- `make check` (ruff + mypy --strict + pytest) must be green before every commit.
  Tests that download the real model are gated behind `PII_RUN_MODEL_TESTS=1`
  (`make test-model`).
- All configuration uses the `PII_*` env prefix — shared knobs (model id, cache dir,
  threshold) are read identically by the core library and the gateway settings.
- Observability integrations default to `[REDACTED]`; legacy `PiiModel.anonymize()`
  keeps label placeholders like `[PERSON_NAME]`.
- Examples live under `examples/` as repo onboarding assets, not installed package
  modules. Add examples only for integrations that are actually supported.
- LangSmith masking must be client-side via `Client(anonymizer=...)` /
  `hide_metadata=True`. LangChain's PIIMiddleware does NOT protect traces —
  metadata, error strings, and non-agent `@traceable`s still leak, and the JS
  variant restores originals after the model call. Route trace scrubbing through
  the LangSmith client, not the agent middleware.
- LangSmith has no post-ingestion scrub. Once a run is uploaded raw, the only
  remediation is trace deletion or a workspace TTL — so anonymize before upload.
- `make_anonymizer` passes `Redactor.redact` directly as the anonymizer (it's
  already `dict -> dict`, walks nesting, and honors `include_paths`/`exclude_paths`).
  We deliberately do NOT wrap `redact_text` via `langsmith.anonymizer.create_anonymizer`
  because that calls per-string and would drop path filtering. LangSmith applies
  the anonymizer to inputs/outputs and a `{"error": ...}`-wrapped error string.
