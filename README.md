# Privacy Kit

Ready-to-use privacy blocks for LLM apps, observability pipelines, and AI
tools, powered by
[`bardsai/eu-pii-anonimization-multilang`](https://huggingface.co/bardsai/eu-pii-anonimization-multilang)
— an on-device, 24-language EU-PII NER model. No text ever leaves your machine
to be classified.

The goal is simple: put a privacy block at every point where data crosses a
trust boundary. There are two operations and two deployment modes:

|  | You own the code (in-process) | You don't (network path) |
| --- | --- | --- |
| **Data is stored / observed** → redact, one-way | `Redactor`, Langfuse mask, LangChain callback | OTLP sink scrubbing telemetry |
| **Data is in a live LLM loop** → pseudonymize, reversible | `Vault` (`anonymize` / `deanonymize`) | **Gateway** for Claude Code, Codex, Cursor |

## The Gateway — a local proxy that scrubs PII on the LLM request path

A local proxy for Claude Code, Codex, and Cursor — tools you can't add a mask callback to, but that all honor a `*_BASE_URL` override. It sits on that path, detects PII, and (optionally) replaces values with `[TYPE_N]` placeholders before they leave your machine, rehydrating the response on the way back.

### Quick start

**Docker** (gateway + dashboard — only Docker needed on the host; model baked into the image at build time):

```bash
make setup
make route             # route Claude Code + Codex
make run   # dashboard → http://127.0.0.1:3000, gateway → :8787
```


Open **http://127.0.0.1:3000**. Start a **new** tool session after routing. For Cursor, also set the chat panel base URL in Settings → Models (`privacy-kit setup cursor` prints the value). Undo with `make route-remove` or `privacy-kit setup … --remove`.

### How it works

```
                  ┌─────────────────────────────┐
   request        │     privacy-kit gateway     │     sanitized* request
    with    ─────▶│     detect PII + audit      │─────▶   sent to the
    PII           │     pseudonymize*           │         real LLM API
                  │                             │
   response       │                             │     response with
    real    ◀─────│     rehydrate*              │◀─────   placeholders
   values         └─────────────────────────────┘
                        * enforce mode only (PII_POLICY=pseudonymize)
```


Claude Code, Codex, and Cursor send your prompts (and file reads) to cloud LLMs — you can't add a mask callback, but they all honor a `*_BASE_URL` override. The gateway is a local proxy on that path: it detects PII, writes an audit row, and — in **enforce** mode (`PII_POLICY=pseudonymize`) — replaces values with `[TYPE_N]` placeholders before they leave your machine, rehydrating the response on the way back (streaming included). The default **monitor** mode only detects and logs; the prompt is forwarded unchanged. Your auth passes through untouched, and every routed tool's interactions land in the same shared [audit log](#dashboard).

Start the gateway first (loads the on-device model, listens on `127.0.0.1:8787`), then route each tool through it:

```bash
privacy-kit serve            # start the gateway
privacy-kit report           # summarize the audit log
privacy-kit scan secrets.txt # one-off detection; --anonymize to mask
privacy-kit import           # import past Claude Code / Codex conversations
```

#### Claude Code

Routes via `ANTHROPIC_BASE_URL` in `~/.claude/settings.json` (applies to **new** sessions). A Claude Max/Pro **subscription works with no API key** — the gateway forwards your OAuth token and preserves Claude Code's system identifier. (If your version won't send that token to a custom base URL, run `claude setup-token` once.)

```bash
privacy-kit setup claude-code --apply   # route persistently   (undo: --remove)
privacy-kit serve --route claude-code   # only while the gateway runs (auto-restores)
# make route-claude-code / make route-claude-code-remove
```

#### Codex

`setup codex --apply` adds a dedicated `[model_providers.privacy-kit]` table to `~/.codex/config.toml` and selects it with `model_provider = "privacy-kit"`. It sets `supports_websockets = false` so Codex sends plain HTTP the gateway can sanitize (instead of its default WebSocket transport). Works in **both** auth modes from one setting: a ChatGPT-account login (no API key — experimental) or an API key.

```bash
privacy-kit setup codex --apply   # route persistently   (undo: --remove)
# make route-codex / make route-codex-remove
```

#### Cursor

Cursor's surfaces sit on two backends, so it needs two layers:

1. **Chat/plan panel** — the only surface the gateway can pseudonymize. Set it manually: Settings → Models → Override OpenAI Base URL = `http://127.0.0.1:8787/v1` plus your OpenAI key (`privacy-kit setup cursor` prints these).
2. **Composer, agent loop, inline edit (Cmd+K), Apply, Tab** bypass that base URL, so they **can't** be pseudonymized — audited via [Cursor hooks](https://cursor.com/docs/hooks). Three hooks are registered: `beforeSubmitPrompt` and `beforeReadFile` (prompt and file-read audit; set `PII_CURSOR_BLOCK=1` to deny on PII, fails open if the gateway is down) and `afterAgentResponse` (records the agent's reply alongside the prompt in the dashboard, observe-only — hooks can't rewrite content, so no redaction is possible here).

```bash
privacy-kit setup cursor --apply                  # install hooks in ~/.cursor/hooks.json
privacy-kit setup cursor --apply --scope project  # ...in .cursor/hooks.json instead
# undo: --remove  ·  make route-cursor / make route-cursor-remove
```

`make route` / `make route-remove` apply or undo all three at once.

### Dashboard

While the gateway runs, the **web dashboard** — a self-contained JS web UI (`make run` or `make dev`, then `http://127.0.0.1:3000`) — shows the shared audit log for all three tools: overview charts, per-interaction before/after text, filter / sort / search, CSV/JSON export, delete/clear, and an in-memory live PII preview (never stored). No external assets, no CDN.

### Importing past conversations

The proxy only sees traffic from the moment you route a tool through it. `privacy-kit import` backfills the audit log from the history already on disk: Claude Code session transcripts (`~/.claude/projects/*/*.jsonl`) and Codex rollout files (`~/.codex/sessions/**/rollout-*.jsonl`). Every message runs through the same on-device detection pipeline as live traffic — one placeholder vault per conversation, so a value keeps its `[TYPE_N]` placeholder across turns — and lands in the dashboard with its **original timestamps**, marked `source = claude-code-import` / `codex-import` and `policy = imported` so the existing filters separate it from live traffic. `PII_SAVE_TEXTS` is honored: with `anonymized`, only segments that contained PII are stored.

```bash
privacy-kit import                                # import everything from both tools
privacy-kit import --dry-run                      # discover + dedupe only, write nothing
privacy-kit import -s claude-code --since 2026-06-01
privacy-kit import -s claude-code --project my-repo
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--source`, `-s` | `claude-code`, `codex` | History source(s) to import; repeat the flag to pass several. |
| `--since` | — | Only sessions whose file was modified on/after this date. Accepts `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM:SS`, or `"YYYY-MM-DD HH:MM:SS"` (local time). |
| `--until` | now | Only sessions whose file was modified on/before this date (same formats; a date-only value covers that whole day). |
| `--project` | — | Claude Code only: substring filter on the project directory name (e.g. `my-repo` matches `~/.claude/projects/-home-me-my-repo`). |
| `--dry-run` | off | Discover sessions and report what would be imported without loading the model or writing anything. |
| `--db` | `PII_DB_PATH` | Audit database to write into, overriding the configured path. |

Re-running is **idempotent**: a session's own UUID becomes its `conversation_id`, and sessions already present (imported earlier *or* captured live through the proxy) are skipped — so periodic re-runs import only what's new. One limitation: a session that gained new turns after being imported is skipped whole; delete it in the dashboard and re-import to pick up the additions. Skipped on principle, mirroring the proxy's storage policy: subagent sidechains, slash-command wrappers, thinking blocks, tool-call arguments, and injected system/developer instructions.

The same importer is exposed in the dashboard (**Settings → Import history**): pick sources, filter by date range or Claude Code project, dry-run to see counts without writing, and watch progress live — the new-vs-imported split updates as you change the filters. **Show sessions** expands an inspectable list of the individual conversations the filters cover — title (the session's first human prompt), project, last-modified date, and a new/imported badge — and unchecking a session skips it on import (everything beyond the list's display cap stays included). Under the hood that's `GET /api/v1/import/preview` (optional `since` / `until` / `project` query params), `GET /api/v1/import/preview/sessions` (same params plus `sources` and `limit`; newest first, titles honor `expose_plaintext`), `POST /api/v1/import` (body `{"sources": ["claude-code", "codex"], "since": "2026-06-01", "until": "2026-06-30", "project": "my-repo", "exclude_session_ids": ["<uuid>"], "dry_run": false}`, every field optional, runs in the background, `409` if a run is active), and `GET /api/v1/import/status`. The dashboard always writes to the gateway's configured store — the CLI's `--db` override has no equivalent there.

### Wire formats and telemetry

Supported wire formats: Anthropic Messages (`/v1/messages`, `/v1/messages/count_tokens`), OpenAI Chat Completions, OpenAI Responses. Cursor hooks post to `/v1/cursor-hook`.


## In-process library

Use these blocks when you own the code path — Langfuse callbacks, LangChain
agents, direct redaction, or a `Vault` in your own LLM loop.

### Install

```bash
pip install privacy-kit                # library blocks (ONNX, no torch)
pip install 'privacy-kit[gateway]'     # + the local gateway proxy & CLI
```

The default install uses a local ONNX backend:

- no `torch`
- no `transformers`
- model downloaded from Hugging Face on first use
- model cached locally by `huggingface-hub`

### Langfuse In One Line

If you already use Langfuse, pass `make_mask()` to the Langfuse client:

```python
from langfuse import Langfuse
from privacy_kit.integrations.langfuse import make_mask

langfuse = Langfuse(mask=make_mask())
```

`make_mask()` recursively scans every string Langfuse sends through the mask
callback, including nested `input`, `output`, `metadata`, `messages`, tool calls,
reasoning fields, and custom JSON fields.

By default, you do not need to configure field paths. Everything text-like is scanned.

### LangGraph / LangChain

LangChain agents are LangGraph-backed. Install the LangChain extra:

```bash
pip install 'privacy-kit[langchain]'
```

Then create the Langfuse callback with privacy-kit:

```python
from langchain.agents import create_agent
from privacy_kit.integrations.langchain import make_langfuse_callback

langfuse_handler = make_langfuse_callback()

agent = create_agent(
    model="groq:llama-3.1-8b-instant",
    tools=[],
    system_prompt="You are a concise assistant.",
)

agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Jan Kowalski, jan.kowalski@example.com, +48 501 222 333",
            }
        ]
    },
    config={"callbacks": [langfuse_handler]},
)
```

The callback protects LangChain/LangGraph payloads before they are exported to
Langfuse. This includes nested message objects and metadata that LangChain passes
through callback events.

### Direct Redaction

Use `Redactor` directly if you want to redact text or arbitrary JSON-like payloads.

```python
from privacy_kit import Redactor

redactor = Redactor()

redactor.redact_text("Kontakt: jan.kowalski@example.com. Telefon +48 501 222 333.")
# "Kontakt: [REDACTED]. Telefon [REDACTED]."
```

Structured payloads:

```python
redactor.redact(
    {
        "input": "Jan Kowalski, jan.kowalski@example.com",
        "output": "Wyślemy odpowiedź do Jana Kowalskiego.",
        "metadata": {"phone": "+48 501 222 333"},
    }
)
# {
#   "input": "[REDACTED], [REDACTED]",
#   "output": "Wyślemy odpowiedź do [REDACTED].",
#   "metadata": {"phone": "[REDACTED]"},
# }
```

### Reversible Pseudonymization

When the consumer of the text still needs referential consistency — and you
need the real values back — use a `Vault` instead of redacting:

```python
from privacy_kit import Vault, anonymize, deanonymize

clean, vault = anonymize(
    "Jan Kowalski (jan.kowalski@example.com) met Jan Kowalski.",
    detector,  # any Detector, e.g. privacy_kit.core.build_detector()
)
# clean == "[PERSON_NAME_1] ([EMAIL_ADDRESS_1]) met [PERSON_NAME_1]."
#           same value -> same placeholder, every time

deanonymize(clean, vault)
# back to the original text

vault.type_counts        # {"PERSON_NAME": 1, "EMAIL_ADDRESS": 1} — safe to log
```

This is the primitive the gateway is built on: an LLM can reason about
`[PERSON_NAME_1]` consistently across a whole conversation, and the response
is rehydrated before the user sees it.

### Field Selection

The default is intentionally broad: scan every string in the payload.

If needed, limit or exclude paths:

```python
from langfuse import Langfuse
from privacy_kit.integrations.langfuse import make_mask

langfuse = Langfuse(
    mask=make_mask(
        include_paths=[
            "input",
            "output",
            "metadata",
            "messages.*.content",
            "tool_calls.*.args",
        ],
        exclude_paths=[
            "metadata.trace_id",
            "metadata.model",
            "usage",
        ],
    )
)
```

`exclude_paths` wins over `include_paths`.

Environment variables are also supported:

```bash
export PII_INCLUDE_PATHS="input,output,metadata,messages.*.content"
export PII_EXCLUDE_PATHS="metadata.trace_id,usage"
export PII_EXCLUDE_LABELS="ORGANIZATION_NAME"
```

## Configuration

Everything reads the same `PII_*` environment variables:

```bash
export PII_MODEL_ID=bardsai/eu-pii-anonimization-multilang   # default
export PII_MODEL_CACHE_DIR=/path/to/model-cache              # optional
export PII_THRESHOLD=0.5            # min confidence for a span to count as PII

# gateway only:
export PII_POLICY=monitor           # "monitor" (default): detect + log, forward prompt unchanged
                                    #   (real PII reaches the upstream); "pseudonymize": replace
                                    #   PII with [TYPE_N] placeholders before forwarding, then rehydrate
export PII_HOST=127.0.0.1
export PII_PORT=8787
export PII_DB_PATH=privacy_kit.sqlite
export PII_SAVE_TEXTS=anonymized    # save request texts: "anonymized" (default) or "all"
export PII_ANTHROPIC_UPSTREAM=https://api.anthropic.com
export PII_OPENAI_UPSTREAM=https://api.openai.com
export PII_CHATGPT_UPSTREAM=https://chatgpt.com/backend-api  # Codex subscription mode upstream
export PII_OTEL_DOWNSTREAM=          # optional collector for scrubbed telemetry
```

The model is not bundled in the package. It is downloaded on first use and reused
from cache after that.

## Supported Integrations

| Target | Status | API |
| --- | --- | --- |
| Langfuse SDK | Ready | `Langfuse(mask=make_mask())` |
| LangChain / LangGraph | Ready | `make_langfuse_callback()` with `config={"callbacks": [...]}` |
| Claude Code | Ready | gateway: `privacy-kit setup claude-code` |
| Codex (API key) | Ready | gateway: `privacy-kit setup codex --apply` |
| Codex (ChatGPT subscription) | Experimental | gateway: `privacy-kit setup codex --apply` |
| Cursor (chat panel) | Ready | gateway base URL: `privacy-kit setup cursor` (pseudonymizes) |
| Cursor (Composer / agent) | Ready | hooks: `privacy-kit setup cursor --apply` (audit; block via `PII_CURSOR_BLOCK=1`) |
| OpenTelemetry logs | Ready | gateway OTLP sink (`OTEL_EXPORTER_OTLP_PROTOCOL=http/json`) |
| Pydantic AI | Coming soon | Planned integration example |
| OpenAI SDK | Coming soon | Planned in-process wrapper on `Vault` |

Runnable examples live in [`examples/`](examples/). They are kept in the
repository for copy-paste onboarding and are not installed as runtime package
modules.

## What Gets Sent Where

privacy-kit runs on your machine; the model runs locally in-process.

For Langfuse/LangChain (observability) use cases:

1. Your app receives or creates raw LLM data.
2. Your app may still send raw data to your chosen LLM provider.
3. Before observability export, privacy-kit redacts the Langfuse/LangChain payload.
4. Langfuse receives the redacted payload.

For the gateway use case, what reaches the provider depends on `PII_POLICY`. In
the default **monitor** mode the prompt is forwarded **unchanged** (raw values
reach the provider) while the detection is logged locally. Under
**`PII_POLICY=pseudonymize`** the guarantee is stronger: the **LLM provider itself
never sees the raw values** — only placeholders. Either way, the local audit store
keeps entity counts plus the request texts selected by `PII_SAVE_TEXTS` (original
and anonymized, in plaintext on your machine).

## Transformers Backend

privacy-kit also keeps the original `PiiModel` API for label-preserving
anonymization and direct entity extraction.

Install the optional Transformers backend:

```bash
pip install 'privacy-kit[transformers]'
```

Use `PiiModel`:

```python
from privacy_kit import PiiModel

model = PiiModel().from_pretrained("bardsai/eu-pii-anonimization-multilang")

text = "Anna Wiśniewska mieszka na ul. Piękna 22, 00-549 Warszawa."

model.anonymize(text)
# "[PERSON_NAME] mieszka na [LOCATION]."

model.anonymize(text, mode="ids")
# {
#   "anonymized_text": "[PERSON_NAME:1] mieszka na [LOCATION:1].",
#   "entities": {
#       "PERSON_NAME:1": "Anna Wiśniewska",
#       "LOCATION:1": "ul. Piękna 22, 00-549 Warszawa",
#   },
# }

model.extract_pii(text)
# {
#   "Anna Wiśniewska": "PERSON_NAME",
#   "ul. Piękna 22, 00-549 Warszawa": "LOCATION",
# }
```

Use this lower-level API when you want entity labels such as `[PERSON_NAME]` or
extracted PII mappings. The integration APIs use `[REDACTED]` by default because
observability systems usually should not store the PII type either. For
reversible, indexed placeholders prefer the `Vault` API above — it is idempotent
and offset-exact.

## Supported Entity Types

The underlying model recognizes EU-relevant PII categories including:

`PERSON_NAME`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `LOCATION`,
`FINANCIAL_AMOUNT`, `PERSON_IDENTIFIER`, `ORGANIZATION_IDENTIFIER`,
`PROPER_NAME`, `RELIGION_OR_BELIEF`, `TRADE_UNION_MEMBERSHIP`, and more.

## Development

```bash
make install      # uv sync with the gateway extra
make check        # ruff + mypy --strict + pytest — must be green before commit
make test-model   # tests that download and run the real model
make serve        # run the gateway locally
```

A privacy invariant is enforced by lint and tests: code under `src/` must
never print or log the text it processes (ruff T20 + the log-safety test
suite), and raw text lives only in the audit store's dedicated texts table
(scoped by `PII_SAVE_TEXTS`) — never in any other table, log, or output.

## License

Apache-2.0
