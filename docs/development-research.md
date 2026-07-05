# privacy-kit — Development Research & Issue Analysis

_Research date: 2026-07-04 · Audited at branch `feat/make-setup`, version 0.3.0 (Alpha)_

This document is the output of an extensive code-level audit of privacy-kit
across three subsystems — the core detection engine, the gateway proxy, and the
integrations/tests/packaging surface. It records **what the project is today,
the concrete issues found (with file references), and a prioritized roadmap for
further development.** Findings marked **[verified]** were confirmed directly in
the code during this research.

---

## 1. Executive summary

privacy-kit is in good architectural shape for an alpha: a clean two-altitude
model (in-process blocks + an out-of-process gateway), a genuinely torch-free
default install, and a set of privacy invariants that are _enforced by tests_
rather than merely documented (no raw text in logs, raw PII confined to one DB
table, no optional deps on base import). The streaming placeholder rehydration
and the USER/TOOL/MACHINE novelty de-duplication are the strongest-engineered
parts of the codebase.

The most important gaps cluster into four themes:

1. **Detection is 100% ML with no deterministic safety net.** There is zero
   regex/checksum detection. Structured secrets (API keys, JWTs, credit cards,
   IBANs, US SSNs) that an NER model structurally cannot catch reliably will
   leak. For a privacy tool this is the single biggest correctness gap.
2. **The gateway leaks on unrecognized request shapes.** Each wire-format
   transform only rewrites the fields it explicitly knows; any unmodeled
   content block passes through verbatim, PII included, with no fallback scrub
   and no telemetry that it happened.
3. **Plaintext PII at rest is easy to leak.** The default `PII_SAVE_TEXTS=all`
   writes raw originals to an unencrypted SQLite file that is **not gitignored**,
   world-readable by umask, and served unauthenticated at `/ui`. The documented
   default contradicts the code default.
4. **No accuracy measurement.** There is no precision/recall benchmark anywhere;
   a missed entity is a silent leak, and nothing in CI would catch a model
   regression. The "24-language" claim is exercised by only 5 languages.

None of these are architectural dead-ends — they are addressable increments on a
sound foundation. The rest of this document details them and proposes a phased
plan.

---

## 2. Current state

### 2.1 Architecture

| Altitude | Blocks | Destination |
| --- | --- | --- |
| In-process (you own the code) | `Redactor` (one-way `[REDACTED]`), `Vault` (reversible `[TYPE_N]`), Langfuse `make_mask`, LangChain callback | data is stored/observed, or in a live LLM loop |
| Out-of-process (you don't) | Gateway proxy (Claude Code, Codex, Cursor), OTLP sink | network path |

- **Core detector**: `BardsAiOnnxDetector` (ONNX Runtime + `tokenizers`, char-offset
  `Span` output, windowing/batching). This is the production path used by
  `Redactor` and `Vault`.
- **Legacy detector**: `PiiModel` (`model.py`, torch/transformers, whitespace/token-index
  based) — exported at top level but **not wired into** the core pipeline. A
  second, divergent implementation.
- **Gateway**: FastAPI proxy with per-wire-format transforms (Anthropic Messages,
  OpenAI Chat Completions, OpenAI Responses), an SSE rehydration decoder, a
  metadata+text SQLite audit store, an OTLP/HTTP JSON sink, and an inline `/ui`.

### 2.2 What is solid (keep / build on)

- Privacy invariants enforced by tests: `test_log_safety.py` (no raw PII in
  logs, even on a crashing forwarder), `test_store.py` / `test_gateway_e2e.py`
  (raw PII in no DB table but the dedicated one), `test_packaging.py` (base
  import pulls no optional deps).
- `PlaceholderStreamDecoder` (`gateway/proxy/streaming.py`) correctly buffers
  placeholders split across SSE chunks.
- USER/TOOL/MACHINE author model + `novel` de-dup avoids re-saving conversation
  history and never stores system prompts / assistant turns / tool-call args.
- Clean gateway transform registry (`REQUEST_TRANSFORMS` / `RESPONSE_TRANSFORMS`)
  — adding a wire format is a self-contained function pair.
- Torch-free default is real and test-enforced.

---

## 3. Critical issues

Ordered by severity. The first three are the ones to fix first.

### 3.1 No deterministic detection layer — structured secrets leak

**[verified]** The only `re` usage in core is cosmetic punctuation trimming;
`build_detector` explicitly rejects a `"regex"` backend
(`core/detectors.py:321`). Detection is entirely the multilingual NER model.

NER models are unreliable on format-defined, high-entropy data — and there is no
label at all for API keys/tokens. The following have **no safety net** and will
leak when the model misses them:

- API keys / secrets (`sk-...`, `AKIA...`, `ghp_...`, `xoxb-...`), JWTs, bearer
  tokens, PEM private-key blocks — no entity type exists; guaranteed leak.
- Credit cards, IBANs, US SSNs, bare numeric IDs — caught only when surrounding
  context looks right; a bare `4111111111111111` is often missed. No checksum
  validation (Luhn / IBAN mod-97) anywhere.
- Non-EU PII and non-Latin scripts beyond the model's training coverage.

This is especially acute because the gateway sits in front of coding tools
(Claude Code, Codex, Cursor) whose **file reads routinely contain exactly these
secrets** (`.env` files, config, keys).

**Fix direction:** add a deterministic regex+checksum layer and union its spans
with the model's. Re-enable the `"regex"` backend the detector currently
rejects. This is the highest-leverage single change in this document.

### 3.2 Gateway silently forwards unrecognized shapes

Each transform in `gateway/proxy/transform.py` rewrites only known fields
(`_map_text_blocks` touches `str` content or list blocks with a `text` key;
`_anthropic_content` descends string / `{text}` blocks). Anything else —
multimodal parts (`image_url`), alternate block keys (`input_text`), structured
tool-result objects — is **forwarded verbatim, PII included.** There is no
defense-in-depth fallback scrub and no counter/warning when an unmodeled shape
is seen. The whole product promise is "PII never leaves the machine," so
silent passthrough on unknown shapes is the top _gateway_ correctness risk.

**Fix direction:** after the structural transform, run an optional recursive
fallback scrub over remaining string leaves (the OTLP sink already does a full
recursive walk — reuse that), or at minimum emit a metric when a request
contains content blocks the transform didn't recognize.

### 3.3 Plaintext PII at rest is easy to leak

- **[verified]** `PII_SAVE_TEXTS` defaults to `"all"` in code
  (`gateway/config.py:36`) — every eligible segment's raw original is written to
  SQLite in plaintext. **README documents the default as `anonymized`**
  (README "Configuration"). Doc/code mismatch on a privacy-critical default.
  (Note: even `"anonymized"` still stores plaintext _originals_ for segments
  that contained PII — the name means "only PII-bearing rows," not "no raw
  text.")
- **[verified]** The default DB `privacy_kit.sqlite` is **not gitignored** —
  `.gitignore` line 8 is `.env*.sqlite`, which does not match it
  (`git check-ignore` → not ignored). It currently shows as untracked in
  `git status`; one `git add .` commits real PII.
- The DB file is created with no `chmod`/umask hardening (`store/audit.py`) →
  world-readable per process umask. No encryption at rest.
- `/ui` and `/ui/api/texts` are mounted **unauthenticated** (`gateway/ui.py`)
  and return stored plaintext originals. Safe on the default `127.0.0.1` bind,
  but `PII_HOST=0.0.0.0` (or a permissive Docker port map) exposes the entire
  plaintext PII log over the network with no warning.
- No retention / rotation / size cap. Multi-turn tools re-send history each
  turn; the DB grows unbounded with plaintext PII.

**Fix direction:** (a) fix `.gitignore` to `*.sqlite`; (b) align the default —
make `anonymized` (or a new `none`) the real default and fix the docs;
(c) `chmod 0o600` the DB; (d) refuse/gate `/ui/api/texts` on non-loopback binds;
(e) add retention + a `privacy-kit purge` command.

### 3.4 No accuracy/recall measurement; model tests never run in CI

There is **no precision/recall/F1 benchmark** anywhere. `test_detect_model.py`
only asserts that _some_ expected type is found per sample — it never checks
false negatives (leaks) or false positives. All real-model tests are gated
behind `PII_RUN_MODEL_TESTS=1`, and CI (`.github/workflows/ci.yml`) never sets
it — so the behavioral contract and e2e pipeline **never execute in CI**. A
model regression can merge undetected. The "24-language" claim is exercised by
only 5 languages (en, pl, de, fr, es) in `tests/fixtures/pii_samples.json`.

**Fix direction:** build a labeled multilingual benchmark corpus; assert a
minimum recall in a nightly/scheduled CI job with `PII_RUN_MODEL_TESTS=1`.

---

## 4. Detailed findings by subsystem

### 4.1 Core detection / redaction / vault

- **Score-averaging can suppress real PII** (`core/detectors.py`): merged
  multi-word spans average subword scores; the threshold is applied _after_
  averaging, so a strong word diluted by weak neighbors can drop below
  threshold. Consider min/max instead of mean for the merge.
- **Overlap dedup silently drops different-label spans** (`detectors.py`
  longest-first keep): fine for redaction, but changes referential grouping for
  pseudonymization and can't represent two genuinely overlapping PII types.
- **Vault vs Redactor have different leak profiles for the same detector
  output.** The Redactor expands partial spans to word boundaries
  (`core/redactor.py`); `Vault.anonymize_into` splices the raw detector span
  verbatim (`core/vault.py`) — a partial-token detection leaks its untagged
  remainder under the Vault path only. Unify the span preparation.
- **`deanonymize` is blind `str.replace`** (`core/vault.py`): if the original
  text (or a real PII value) contains a literal `[PERSON_NAME_1]`-shaped token,
  round-trip fidelity is not guaranteed. Needs positional tracking or escaping.
- **Vault is not thread-safe**: three mutable dicts, read-modify-write with no
  lock, yet the docstring encourages sharing one vault across strings. Add a
  lock or document single-thread-only. Memory growth is unbounded (no eviction).
- **Inference is synchronous and CPU-only**: provider hardcoded to
  `CPUExecutionProvider` (`detectors.py`), no async wrapper — calling `detect()`
  from an event loop blocks it. Model load does three blocking `hf_hub_download`
  calls.
- **Two divergent detector implementations**: `PiiModel` (`model.py`) is a
  second, token-index-based detector exported at top level but unused by the
  core pipeline. Deprecate it or converge it onto the `Span` contract.

### 4.2 Gateway proxy

- **Streaming rehydration is robust** — split-placeholder buffering, flush on
  broken streams, audit fires in `finally`. Minor risk: the flush injects the
  held remainder into the last-seen block index, which could theoretically be
  wrong for interleaved text+tool_use blocks.
- **No max body / max decompressed size limit** on the proxy path — a
  highly-compressible gzip/zstd body (decompression bomb) could exhaust memory
  (`_decompress` in `proxy/app.py`).
- **Streaming forwarder uses `timeout=None`** (`proxy/app.py`) — an unresponsive
  upstream can hang a streaming connection indefinitely. No retries/backoff on
  transient upstream 5xx.
- **Audit writes are best-effort** (`suppress(Exception)`), so SQLite
  `database is locked` under concurrency silently drops audit rows — safe for
  the proxy, but audit completeness isn't guaranteed and failures are invisible.
- **Client-controlled `x-privacy-kit-source` header** is stored unvalidated as
  the audit `source` (rendered safely via `textContent`, so no XSS — but it's
  unvalidated free text landing in the DB).
- **Coverage gaps**: no Gemini, no Ollama/vLLM, no legacy `/v1/completions`, no
  WebSocket (Codex falls back to HTTPS which is handled). `count_tokens` returns
  an approximate count because the sanitized text tokenizes differently.

### 4.3 Integrations, tests, packaging, DX

- **`langchain.py` is entirely Langfuse-coupled** despite its name — there is no
  standalone LangChain callback that works without Langfuse, and the `langchain`
  extra force-bundles Langfuse + Groq + OpenAI + LangGraph. Heavy and opinionated.
- **In-process integrations are bespoke** — no `Integration` base class or
  entry-point plugin surface. Won't scale cleanly to 10+ integrations.
- **Promised-but-absent**: OpenAI SDK wrapper and Pydantic AI (both "Coming
  soon" in the README) have no code or stubs.
- **Missing high-value integrations**: LiteLLM (one integration → many
  providers — biggest leverage), Anthropic SDK, Ollama/vLLM (natural fit for a
  local-first tool), LlamaIndex, Haystack, DSPy, and any JS/TS ecosystem.
- **OTLP sink scrubs all strings generically** rather than targeting the
  OpenTelemetry GenAI semantic conventions (`gen_ai.*`).
- **No JS/TS package**: `/ui` is an inline HTML string in Python; the `web/`
  Next.js scaffold is untracked WIP. Nothing on npm despite the gateway serving
  TS-based tools.
- **Release is manual**: `scripts/publish.sh` (twine) only; no tag-triggered
  GitHub Actions release, no PyPI trusted publishing (OIDC).
- **DX**: no root `.env.example`, no `privacy-kit doctor`/`config` command to
  show effective settings + model cache status, no CONTRIBUTING or
  threat-model/architecture doc. `make setup` applies routing _before_
  `make run` starts the gateway, so tools can point at a dead proxy; all Make
  targets assume `uv` (no pip-only path).

---

## 5. Roadmap

### Phase 0 — Quick wins (hours, high value)

- [ ] Fix `.gitignore`: `.env*.sqlite` → `*.sqlite` (prevents committing PII). **[verified issue]**
- [ ] Resolve the `PII_SAVE_TEXTS` default mismatch and document it truthfully. **[verified issue]**
- [ ] `chmod 0o600` the audit DB on creation.
- [ ] Add root `.env.example` documenting all `PII_*` vars.
- [ ] Make `make setup` warn if the gateway isn't running / order it after `make run`.

### Phase 1 — Close the leak paths (the trust story)

- [ ] **Deterministic detection layer**: regex + checksum (Luhn/IBAN mod-97) for
      cards, IBANs, emails, E.164 phones, US SSN, JWTs, API-key prefixes, PEM
      blocks, high-entropy strings; union with model spans; re-enable the
      `regex` backend.
- [ ] **Gateway fallback scrub** for unrecognized content shapes + a metric when
      an unmodeled block is seen.
- [ ] **Unify Vault/Redactor span preparation** so both paths have the same leak
      profile; harden `deanonymize` against placeholder-shaped input.
- [ ] **DB hardening**: retention/rotation + `privacy-kit purge`; gate
      `/ui/api/texts` on non-loopback binds; max body/decompressed-size limits;
      streaming read timeout.

### Phase 2 — Prove correctness

- [ ] **Precision/recall benchmark** with a labeled multilingual corpus; enforce
      a minimum recall in CI.
- [ ] **Run model tests in CI** (nightly, `PII_RUN_MODEL_TESTS=1`).
- [ ] Expand fixtures to the full 24-language claim (or soften the claim).
- [ ] Add adversarial-PII, concurrency, and placeholder-injection tests.

### Phase 3 — Grow the integration surface

- [ ] Lightweight **integration plugin surface** (base class / entry points)
      before adding more one-off modules.
- [ ] Ship the promised **OpenAI SDK** + **Anthropic SDK** in-process wrappers on
      `Vault`.
- [ ] **LiteLLM** integration (highest leverage) and **Ollama/vLLM** proxy support.
- [ ] Decouple `langchain.py` from Langfuse; slim the extra.
- [ ] Map the OTLP sink to GenAI semantic conventions.

### Phase 4 — Distribution & platform

- [ ] GitHub Actions **release workflow** with PyPI trusted publishing (OIDC),
      tag-triggered; stop committing `dist/`.
- [ ] Decide `web/`'s fate — a real tracked TS UI + npm SDK, or remove the
      scaffold; publish a multi-arch Docker image.
- [ ] Async/GPU inference path; `privacy-kit doctor`; CONTRIBUTING + threat model.

---

## 6. Strategic bets & open questions

- **Local-first proxy for local models.** The detector is already local; adding
  an Ollama/vLLM upstream makes privacy-kit a fully-offline privacy layer — a
  differentiated position no cloud DLP tool can match.
- **Secrets as a first-class type.** Extending beyond EU-PII into
  credentials/secrets (the deterministic layer) reframes the product from
  "GDPR anonymizer" to "nothing sensitive leaves your machine" — a broader,
  security-buyer story that fits the coding-tool gateway use case.
- **Who is the buyer?** In-process blocks target developers; the gateway targets
  security/compliance teams governing AI-tool usage. The plugin surface and the
  benchmark both serve the second audience — worth deciding which to lead with.
- **The "24 languages" and accuracy claims need evidence** before they can be
  marketing load-bearing; the benchmark in Phase 2 is the prerequisite.

---

---

## 7. Correction: current config posture (§3.3 update)

The three audits in §3–4 predate the current `gateway/config.py`, which has since
gained fields the agents didn't see. Correcting the record:

- **`policy: "monitor" | "pseudonymize" = "monitor"`** (`config.py:53`). The
  **default is `monitor`: detect and log PII, but forward the prompt _unchanged_
  — real values reach the upstream LLM.** Only `pseudonymize` replaces PII with
  reversible placeholders and rehydrates the response. This is the intended
  default ("log, don't replace"). **Consequence to document honestly:** the
  README's "What Gets Sent Where" claim that the gateway means "the LLM provider
  never sees the raw values" is only true under `policy=pseudonymize`. In the
  default `monitor` mode privacy-kit is an _observability/DLP-audit_ tool, not a
  data-blocking one. The README should state this plainly.
- **`expose_plaintext: bool = True`** (`config.py:87`) — the dashboard API can
  return raw originals; set `false` for hosted/multi-user so originals are
  redacted in API responses. Good that the gate exists; the §3.3 "unauthenticated
  /ui serves plaintext" concern is mitigated for hosted deploys but the default
  is still expose-on (correct for local single-user).
- **`cursor_block: bool = False`** (`config.py:60`) — Cursor hooks can _deny_ an
  action on PII detection (they can't rewrite). This is effectively a per-surface
  **block** policy and a good seed for a general proxy-wide block mode (see §8).

### sqlite storage investigation (`PII_DB_PATH`)

- **[verified]** `db_path` defaults to **`Path("privacy_kit.sqlite")` — relative
  to the current working directory** (`config.py:38`). For a long-running
  gateway this is fragile: the DB lands wherever you happened to launch from, so
  running from a different directory silently starts a fresh empty audit log and
  scatters stray `privacy_kit.sqlite` files. Compare `route.py`, which correctly
  anchors tool config under `Path.home()`. **Recommend** defaulting the DB to a
  stable per-user location (`~/.privacy-kit/audit.sqlite`, or an XDG/`platformdirs`
  data dir), created with `0o600`.
- `AuditStore` already sets `PRAGMA journal_mode=WAL` + `busy_timeout=5000`
  (`store/audit.py:50-57`) — sound for concurrent dashboard reads + proxy writes.
- Because the default `policy=monitor` does **not** pseudonymize, the audit DB is
  the _only_ place PII is captured locally — which makes securing it (path,
  perms, gitignore, retention, `expose_plaintext`) more important, not less.

---

## 8. Competitive landscape & functionality to add

Research against comparable tools (July 2026). The takeaway: privacy-kit occupies
a **genuinely under-served niche** — an _embedded, local-ML, torch-free_ PII layer
with _native subscription-auth routing_ for the exact coding tools proven to leak
secrets, plus a _reversible Vault_. No single competitor combines all three.

### 8.1 Where each competitor sits

| Tool | What it is | Reversible? | Local ML, no extra container? | Routes Claude Code / Codex / Cursor (subscription auth)? |
| --- | --- | --- | --- | --- |
| **Microsoft Presidio** | OSS detection+anonymization library (recognizers + operators) | Only via `encrypt` operator | Library (you host) | No (not a gateway) |
| **LLM Guard** (protectai, MIT) | Self-hosted scanner library: Anonymize/Deanonymize/**Vault** + injection, secrets, toxicity | Yes (Vault) | Library | No |
| **LiteLLM** | Dominant LLM gateway (100+ providers) | Via Presidio | **No** — PII needs 2 separate Presidio containers; built-in filter is regex-only; secrets are enterprise-only | No (API-key routing, not subscription OAuth) |
| **AI Security Gateway** | Embedded-Presidio proxy | No (redact only) | Yes (embedded) | No (cloud-provider focused) |
| **Skyflow / Protecto / Private AI** | Enterprise SaaS privacy vault; deterministic + format-preserving tokenization | Yes (detokenize) | No (SaaS/containers) | No |
| **Zenity / MintMCP** | Enterprise governance for coding assistants (visibility, block, redact) | N/A (policy layer) | No (SaaS) | Partial (governance, not local scrub) |
| **privacy-kit** | Embedded local-ML blocks + gateway | **Yes (Vault)** | **Yes (ONNX, torch-free)** | **Yes** — unique |

**Market validation:** independent testing confirms Claude Code and Copilot
"send the full contents of your file, including hardcoded passwords, API keys,
and connection strings, to their cloud APIs, with no client-side filtering of
secrets." That is precisely privacy-kit's reason to exist — and it makes **secret
detection** the highest-value feature to add, not a nice-to-have.

### 8.2 Functionality to add (prioritized, mapped to what competitors ship)

1. **Secret detection (highest value, coding-tool wedge).** First-class detectors
   for API keys (`sk-`, `AKIA`, `ghp_`, `xoxb-`), JWTs, PEM private-key blocks,
   `.env`-style `KEY=value` secrets, connection strings, high-entropy strings.
   LLM Guard, AI Security Gateway, Zenity, and LiteLLM-enterprise all treat
   secrets as a distinct category from PII. This is what Claude Code/Copilot fail
   at and what privacy-kit's gateway is uniquely placed to catch on-device.

2. **Deterministic regex + checksum layer** (complements #1 for PII). AI Security
   Gateway markets "pattern matching and checksum validation" _alongside_ the ML
   model; LiteLLM's built-in filter is regex for SSN/credit-card. Add Luhn (cards),
   mod-97 (IBAN), E.164 (phones), SSN — union with model spans. Closes §3.1.

3. **Per-entity policy / operator config.** Today `policy` is global
   (`monitor`/`pseudonymize`). Presidio and Microsoft "PII Shield" offer
   **per-entity-type actions**: keep / redact / pseudonymize / hash / encrypt /
   fake / **block**. Generalize `policy` into a per-type map (e.g. secrets→block,
   names→pseudonymize, org names→keep). Fold the existing `cursor_block` into a
   general **block** action that returns a 4xx when a configured type is detected.

4. **Format-preserving / realistic pseudonyms (optional "fake" operator).**
   Skyflow and Protecto emphasize "context-preserving" tokens because
   `[PERSON_NAME_1]` degrades LLM output quality vs a realistic stand-in. Offer a
   Faker-backed operator (like Presidio's `fake` and PII Shield's `fake` strategy)
   as an alternative to `[TYPE_N]` — still reversible via the Vault mapping.

5. **Vision / OCR scrubbing for images.** Coding tools and chat send base64
   images; privacy-kit currently passes them through unscrubbed (§4.2). AI
   Security Gateway does OCR + DLP on base64 images without an extra LLM call.
   Higher-effort but differentiating and closes a real leak path.

6. **Custom recognizers + sensitive-terms dictionary (deny/allow lists).**
   Skyflow's "sensitive data dictionary" and Presidio's custom recognizers let
   teams add project-specific secrets (internal codenames, customer names) and
   allowlist false positives. A simple `PII_DENY_TERMS` / `PII_ALLOW_TERMS` +
   pluggable recognizer API.

7. **Extensibility refactor toward Presidio's recognizer/operator split.** The
   cleanest way to ship #1–#6 without a tangle: separate **detectors**
   (model, regex, checksum, dictionary — each contributing `Span`s) from
   **operators** (redact/pseudonymize/hash/fake/block per type). Privacy-kit
   already has the `Detector` protocol and a `policy` knob — this is an
   incremental generalization, not a rewrite.

8. **Adjacent-but-decide (scope guard).** LLM Guard bundles prompt-injection,
   toxicity, and jailbreak scanners; AI Security Gateway adds hard budget stops
   (402 before forwarding). These are real markets but pull privacy-kit away from
   "privacy layer" toward "LLM firewall." Recommend staying focused on
   PII+secrets; expose enough extensibility (the operator/detector split) that
   others _could_ add injection scanners, rather than owning that surface.

### 8.3 Positioning (one line)

> The only **local-first, torch-free** privacy layer that both **redacts/pseudonymizes
> on-device** _and_ **routes the subscription-auth coding tools** (Claude Code,
> Codex, Cursor) that are proven to leak secrets — no containers, no cloud, no API
> key required.

Lead with the **coding-tool secret-leak** story (concrete, validated, urgent);
the Langfuse/observability blocks and the reversible Vault are the supporting
proof that the same engine works in-process too.

---

_Method: three parallel code audits (core engine; gateway; integrations/tests/
packaging) reading source + tests, then a competitive-research pass against
Presidio, LLM Guard, LiteLLM, AI Security Gateway, Skyflow/Protecto/Private AI,
and Zenity/MintMCP (web, July 2026). Highest-impact findings — the `policy` /
`save_texts` / `expose_plaintext` defaults, the CWD-relative `db_path`, and the
`.gitignore` gap — verified directly against the code._

### Sources (§7–8 research)
- [Microsoft Presidio — deanonymization](https://deepwiki.com/microsoft/presidio/3.2.2-deanonymization)
- [Microsoft "PII Shield" privacy proxy](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/introducing-pii-shield-a-privacy-proxy-for-every-llm-call/4514726)
- [LiteLLM — Presidio PII masking (needs containers)](https://docs.litellm.ai/docs/proxy/guardrails/pii_masking_v2)
- [LiteLLM — secret detection (enterprise)](https://docs.litellm.ai/docs/proxy/guardrails/secret_detection)
- [AI Security Gateway — LiteLLM alternative (embedded, checksum, OCR, secrets, budget)](https://aisecuritygateway.ai/docs/litellm-alternative)
- [LLM Guard — open-source guardrails (Vault, scanners)](https://appsecsanta.com/llm-guard)
- [Skyflow — LLM Privacy Vault (deterministic/format-preserving tokenization)](https://www.skyflow.com/post/generative-ai-data-privacy-skyflow-llm-privacy-vault)
- [Protecto — context-preserving tokenization](https://www.protecto.ai/skyflow-alternative/)
- [Are your secrets leaking to AI? Claude Code vs Copilot](https://medium.com/@damien.vandekerckhove/are-your-secrets-leaking-to-ai-claude-code-vs-copilot-under-the-microscope-08d04aacabd9)
- [Zenity — securing agentic coding assistants](https://zenity.io/blog/product/from-ide-to-cli-securing-agentic-coding-assistants)
