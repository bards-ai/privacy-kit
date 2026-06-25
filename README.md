# privacy-kit

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

## Quick start

Route your AI tools through the local privacy gateway in two commands:

```bash
make setup    # one-time: route Claude Code + Codex through the gateway
make run      # install deps + start the gateway on http://127.0.0.1:8787
```

Then start a **new** tool session and open `http://127.0.0.1:8787/ui` to see
detected PII and the before/after text. (Cursor takes two steps: point its chat
panel's base URL at the gateway in Settings → Models to pseudonymize that panel,
and run `privacy-kit setup cursor --apply` to install hooks that audit the
Composer/agent surfaces the base URL can't reach.) Undo routing with
`privacy-kit setup claude-code --remove` / `privacy-kit setup codex --remove` /
`privacy-kit setup cursor --remove`.

## Install

```bash
pip install privacy-kit                # library blocks (ONNX, no torch)
pip install 'privacy-kit[gateway]'     # + the local gateway proxy & CLI
```

The default install uses a local ONNX backend:

- no `torch`
- no `transformers`
- model downloaded from Hugging Face on first use
- model cached locally by `huggingface-hub`

## Langfuse In One Line

If you already use Langfuse, pass `make_mask()` to the Langfuse client:

```python
from langfuse import Langfuse
from privacy_kit.integrations.langfuse import make_mask

langfuse = Langfuse(mask=make_mask())
```

`make_mask()` recursively scans every string Langfuse sends through the mask callback, including nested `input`, `output`, `metadata`, `messages`, tool calls, reasoning fields, and custom JSON fields.

By default, you do not need to configure field paths. Everything text-like is scanned.

## LangGraph / LangChain

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

The callback protects LangChain/LangGraph payloads before they are exported to Langfuse. This includes nested message objects and metadata that LangChain passes through callback events.

## The Gateway — privacy for tools you can't modify

Claude Code, Codex, and Cursor send your prompts (and your tools' file reads!)
to cloud LLMs. You can't add a mask callback to them — but they all honor a
`*_BASE_URL` override. The gateway is a local proxy that sits in that path:

```
AI tool ──> privacy-kit gateway ──> real LLM API
            detect + audit
            pseudonymize ──>  (enforce mode only)
                          <── rehydrate
```

By default the gateway runs in **monitor** mode (`PII_POLICY=monitor`): for every
request it detects PII and records an **audit row** — entity types and counts, plus
the request text segments selected by `PII_SAVE_TEXTS` (original and the
would-be-anonymized text, in plaintext in the local SQLite file) — but forwards the
prompt **unchanged**, so the real values reach the upstream. Use it to see what
privacy-kit would catch before turning on enforcement.

Set `PII_POLICY=pseudonymize` to enforce: PII is replaced with `[TYPE_N]`
placeholders before the text leaves your machine, the sanitized body is forwarded
upstream with your own auth, and the real values are **rehydrated** into the
response (streaming included — placeholders split across SSE chunks are buffered and
restored). The audit row is written either way; `PII_SAVE_TEXTS=all` stores every
eligible segment instead of only those where PII was found. Conversation history is
re-sent by the tools each turn, so saved texts accumulate per turn — keep an eye on
the DB size, especially with `all`.

```bash
pip install 'privacy-kit[gateway]'

privacy-kit serve                      # loads the model, listens on 127.0.0.1:8787
privacy-kit serve --route claude-code  # ...and auto-route Claude Code while it runs
privacy-kit setup claude-code --apply  # route Claude Code persistently (no manual exports)
privacy-kit setup claude-code --remove # undo the persistent routing
privacy-kit setup codex --apply        # route Codex persistently (edits ~/.codex/config.toml)
privacy-kit setup codex --remove       # undo the Codex routing
privacy-kit setup cursor               # Cursor: chat-panel base URL (manual) + hooks help
privacy-kit setup cursor --apply       # install Cursor hooks (audit Composer/agent; ~/.cursor/hooks.json)
privacy-kit report                     # summarize the audit log
privacy-kit scan secrets.txt           # one-off detection; --anonymize to mask
```

While the gateway runs, a **PII preview UI** is served at
`http://127.0.0.1:8787/ui`: paste any text to see detected spans highlighted by
entity type next to the pseudonymized version that would leave your machine,
and browse the audit log (totals by type, recent interactions).
The page is a single inline file — no external assets, no CDN — and the live
preview is processed in memory only: never stored, logged, or audited.

No manual `export` needed for Claude Code: `--apply` (or `serve --route
claude-code`) writes `ANTHROPIC_BASE_URL` into the `env` block of
`~/.claude/settings.json` — the CLI prints exactly which file it overrides, the
override applies to new Claude Code sessions, and `serve --route` restores the
previous value on shutdown (a value you changed yourself in the meantime is
never clobbered). Works with a Claude Max/Pro subscription — no API key
required; the gateway forwards your OAuth login token and keeps Claude Code's
system identifier intact so Anthropic accepts the request.

`privacy-kit setup codex --apply` does the same for Codex, writing
`openai_base_url` into `~/.codex/config.toml`. Codex routes its model call
through that base URL in **both** auth modes, so one setting covers both: a
ChatGPT-account login (free/Plus/Pro, **no API key** — experimental) and an API
key. The gateway recognizes a subscription request by its `chatgpt-account-id`
header and forwards it to chatgpt.com's backend with your login token untouched;
an API-key request goes to api.openai.com. Known limitations of subscription
mode: Codex tries a WebSocket first and falls back to HTTPS (the gateway routes
the HTTPS call), and Codex's plugin/MCP "apps" panel is unaffected by the
gateway, so it may log a harmless startup warning.

**Cursor needs two layers**, because its surfaces sit on two backends. Only the
**chat/plan panel** honors a custom OpenAI base URL — point it at the gateway
(Settings → Models → Override OpenAI Base URL = `http://127.0.0.1:8787/v1`, plus
your own OpenAI key) and, under `PII_POLICY=pseudonymize`, its prompts are redacted
and rehydrated exactly like Claude Code / Codex. **Composer, the agent loop, inline
edit (Cmd+K), Apply, and Tab stay on Cursor's own backend** and bypass that base URL
entirely, so they cannot be pseudonymized. `privacy-kit setup cursor --apply` installs
[Cursor hooks](https://cursor.com/docs/hooks) (`beforeSubmitPrompt`, `beforeReadFile`)
into `~/.cursor/hooks.json` (`--scope project` for `.cursor/hooks.json`) so privacy-kit
still **audits** those surfaces; hooks can only allow or deny, never rewrite, so set
`PII_CURSOR_BLOCK=1` to **deny** a prompt or file read that contains PII. The hook is
a thin client that calls the running gateway and **fails open** — if the gateway is
down, Cursor is never blocked.

Supported wire formats: Anthropic Messages (`/v1/messages`,
`/v1/messages/count_tokens`), OpenAI Chat Completions, OpenAI Responses. Cursor
hooks post to `/v1/cursor-hook`.

The gateway also mounts an **OTLP/HTTP JSON sink** (`/v1/logs`, `/v1/traces`,
`/v1/metrics`): telemetry is scrubbed one-way (observability data keeps the
placeholders), logs are audited, and scrubbed payloads can be re-exported to a
downstream collector via `PII_OTEL_DOWNSTREAM`.

Run it in Docker (model baked in at build time, no torch):

```bash
docker build -t privacy-kit .
docker run --rm -p 127.0.0.1:8787:8787 -v privacy-kit-data:/data privacy-kit
```

## Direct Redaction

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

## Reversible Pseudonymization

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

## Field Selection

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

The model is not bundled in the package. It is downloaded on first use and reused from cache after that.

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

Runnable examples live in [`examples/`](examples/). They are kept in the repository for copy-paste onboarding and are not installed as runtime package modules.

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

privacy-kit also keeps the original `PiiModel` API for label-preserving anonymization and direct entity extraction.

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

Use this lower-level API when you want entity labels such as `[PERSON_NAME]` or extracted PII mappings. The integration APIs use `[REDACTED]` by default because observability systems usually should not store the PII type either. For reversible, indexed placeholders prefer the `Vault` API above — it is idempotent and offset-exact.

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
