# privacy-kit — Development Plan

_Companion to [development-research.md](development-research.md). That doc is the
analysis; this is the build plan. Ordered so we can start at the top._

**North star:** the only local-first, torch-free privacy layer that
redacts/pseudonymizes on-device _and_ routes the subscription-auth coding tools
(Claude Code, Codex, Cursor) proven to leak secrets — no containers, no cloud,
no API key. Lead with the **secret-leak** story.

Each task lists the files it touches and a concrete **acceptance** bar. Boxes are
unchecked; check them as we land them.

---

## Milestone 0 — Housekeeping & safety (½ day, do first)

Low-risk fixes that de-risk everything after them.

- [x] **Fix `.gitignore`.** Added `*.sqlite`, `*.sqlite-wal`, `*.sqlite-shm`
  (plus `.env*`). `git check-ignore privacy_kit.sqlite` now exits 0. ✅
- [x] **Stable DB path + perms.** `db_path` now defaults to
  `~/.privacy-kit/audit.sqlite` via `default_factory`; `AuditStore` creates the
  parent dir and `chmod 0o600`s the file. `PII_DB_PATH` override intact; Docker
  still pins `/data`. Covered by `tests/test_store.py::test_db_file_created_owner_only`
  and the updated `test_gateway_smoke.py`. ✅
- [x] **Reconcile the docs with the real defaults.** README already documented
  `monitor` default + the honesty caveat; fixed the remaining `PII_SAVE_TEXTS`
  line (now shows `all` as default) and the CHANGELOG scope line + DB-path-move
  note. Decision: keep `save_texts=all`. ✅

**M0 landed:** `make check` green (ruff + mypy --strict + 250 tests). Commit when ready.

## Milestone 1 — The secret-detection wedge (1–2 weeks, highest value)

The differentiating feature. Ships the "stop Claude Code leaking your `.env`" story.

**M1 landed** (core scope): `core/secret_rules.py` (18 vendored gitleaks-derived
rules + provenance, `scripts/sync_secret_rules.py` + `make sync-secret-rules`),
`core/detectors_secret.py` (SecretDetector + entropy gate),
`core/detectors_regex.py` (Luhn / IBAN mod-97 / E.164 / SSN),
`CompositeDetector` with secret-wins-overlap precedence, wired into
`build_detector` ("local" = model + deterministic, "regex" = deterministic-only),
the gateway (`build_default_app`) and `privacy-kit scan`. 44 new tests incl. the
model-free gateway e2e (`tests/test_gateway_secrets.py`: .env via tool_result —
audited in monitor, replaced in pseudonymize). README + CHANGELOG updated.
Remaining below: the live end-to-end demo with real Claude Code.

- [ ] **Deterministic detector layer.** New `SecretDetector` + `RegexChecksumDetector`
  producing `Span`s, unioned with the model's spans. **Secret patterns: vendor
  the gitleaks ruleset** — adapt gitleaks' `.toml` rules into a curated, tested,
  in-tree pattern table (no runtime dependency on gitleaks; we sync the rules
  periodically). Covers AWS/GitHub/OpenAI/GCP/Slack keys, JWTs, PEM private-key
  blocks, `.env`-style `KEY=value` secrets, connection strings, high-entropy
  strings. Plus hand-rolled checksummed PII (credit-card Luhn, IBAN mod-97,
  E.164 phone, US SSN).
  - Files: new `core/detectors_secret.py` (+ vendored `core/rules/gitleaks.toml`
    or a generated `_secret_rules.py`) / `core/detectors_regex.py`;
    wire into `core/detectors.py:321` (re-enable the rejected `regex`/composite
    backend); `Redactor` and `Vault` consume the unioned spans.
  - Note: keep the gitleaks rule provenance/license (MIT) recorded in-tree; add a
    `make sync-secret-rules` (or a script) that regenerates the table from an
    upstream gitleaks version so updates are a mechanical refresh, not hand-edits.
  - Acceptance: unit tests with positive/negative fixtures per pattern; a `.env`
    file body run through the gateway shows every secret detected; no network /
    model needed for these detectors (fast, deterministic).
- [ ] **Union/merge semantics.** Define precedence when a regex span and a model
  span overlap (prefer the wider; secrets always win over `keep`).
  - Files: `core/detectors.py` (postprocess/overlap resolution)
  - Acceptance: overlap tests in `tests/test_detectors.py`.
- [ ] **Gateway coverage.** Ensure the new detectors run on every extracted
  segment (tool_result / file-read content especially).
  - Files: `gateway/proxy/transform.py`
  - Acceptance: e2e test — a tool-result block containing an AWS key is
    caught in both `monitor` (audited) and `pseudonymize` (replaced) policy.

## Milestone 2 — Per-entity policy & block mode (1 week)

Turn the global `policy` into a per-type action map; add a real block mode.

- [x] **Per-entity action map.** `PII_POLICY_OVERRIDES` (JSON) layers
  `keep | redact | pseudonymize | block` per entity type over the global
  `PII_POLICY` default (monitor → keep, pseudonymize → pseudonymize). Prefix
  wildcards (`SECRET_*`); exact beats wildcard, longest prefix wins.
  `gateway/policy.py` (PolicyResolver + apply_policy) applied per span in the
  proxy's anon path. `hash`/`fake` deferred to M3 operators. ✅
- [x] **Block action.** A `block`-typed detection returns 403
  (`privacy_kit_blocked` + entity counts) before anything is forwarded; audit
  row records `policy="block"` with full detection counts. Cursor hooks deny on
  block-typed entities (generalizing `cursor_block`, which remains as the
  deny-on-any-PII switch). ✅

**M2 landed:** 12 new tests (`tests/test_policy.py`) — resolver precedence,
mixed actions in one text, 403-block e2e (nothing forwarded, audited), redact
is one-way, monitor+override mixing, cursor-hook deny. One behavior change:
monitor mode no longer populates the forward vault (no rehydration of
placeholder-shaped upstream text — correct, since the upstream saw originals).
`make check` green (306 tests).

## Milestone 3 — Output quality & coverage (1–2 weeks)

- [ ] **Format-preserving / fake operator.** Optional Faker-backed stand-ins
  (realistic names/emails) instead of `[TYPE_N]`, still reversible via the Vault
  mapping. Off by default.
  - Files: `core/vault.py`, new `core/operators.py`
  - Acceptance: round-trip test (fake → deanonymize restores original);
    referential consistency preserved (same value → same fake).
- [ ] **Custom recognizers + deny/allow lists.** `PII_DENY_TERMS` / `PII_ALLOW_TERMS`
  and a small pluggable-recognizer hook for project-specific secrets.
  - Files: `core/detectors*.py`, `core/config`/env plumbing
  - Acceptance: a denied term is always caught; an allowed term is never flagged.
- [ ] **Vision / OCR scrubbing (stretch).** Extract base64 images, OCR, scan with
  the same DLP engine, no extra LLM call. Behind a flag.
  - Files: new `gateway/proxy/vision.py`, `transform.py`
  - Acceptance: an image with an embedded email is detected in `monitor`.

## Milestone 4 — Prove it & ship it (ongoing)

- [ ] **Precision/recall benchmark.** Labeled multilingual corpus; assert minimum
  recall (a miss = a leak). Wire a nightly CI job with `PII_RUN_MODEL_TESTS=1`.
  - Files: new `tests/benchmark/`, `.github/workflows/ci.yml`
  - Acceptance: CI fails if recall drops below the agreed floor; model tests run
    somewhere automatically (today they never do).
- [ ] **Refactor to detector/operator split** (absorbs M1–M3 cleanly). Detectors
  contribute spans; operators decide the per-type transform. Presidio's model,
  incremental on the existing `Detector` protocol.
- [ ] **Retention + `privacy-kit purge`**; guard `/ui` plaintext on non-loopback binds.
- [ ] **Release automation.** Tag-triggered GitHub Actions publish with PyPI
  trusted publishing (OIDC); stop committing `dist/`.
- [ ] **Integrations backlog** (from research §8): OpenAI SDK + Anthropic SDK
  in-process wrappers on `Vault`; **LiteLLM** (highest leverage); Ollama/vLLM;
  decouple `langchain.py` from Langfuse.

---

## Suggested Sprint 1 (start here)

1. **Milestone 0** in full (½ day) — safety + doc truth.
2. **M1 deterministic detector layer** for secrets + checksummed PII — the wedge.
3. One end-to-end demo: route Claude Code through the gateway, read a file with a
   fake AWS key + `.env`, show it caught in the `/ui` audit (monitor) and replaced
   (pseudonymize). This demo _is_ the product story.

**Scope guard:** stay on PII + secrets. Prompt-injection / toxicity / budget caps
(LLM Guard, AI Security Gateway territory) are out — expose extensibility so
others can add them, don't own that surface.

## Decisions (locked for Sprint 1)

- **Secret-detection source:** ✅ **vendor the gitleaks ruleset** — curated,
  tested, in-tree table regenerated from upstream (no runtime dep). See M1.
- **`PII_SAVE_TEXTS` default:** ✅ **keep `all`** (full plaintext local audit);
  fix the README to match. Mitigated by M0 DB hardening (stable path, `0o600`,
  gitignored). See M0.
- **Default `policy`:** ✅ **`monitor`** (log, don't replace); document plainly
  that real PII reaches upstream in this mode.

## Still open (not blocking Sprint 1)

- Whether to eventually add a `PII_SAVE_TEXTS=none` metadata-only option for
  hosted/multi-user deployments (pairs with `expose_plaintext=false`).
- Per-tool policy overrides (e.g. Cursor stricter than Claude Code) — revisit at M2.
