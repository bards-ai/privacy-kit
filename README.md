# privacy-kit

Ready-to-use privacy blocks for LLM apps and observability pipelines, powered by
[`bardsai/eu-pii-anonimization-multilang`](https://huggingface.co/bardsai/eu-pii-anonimization-multilang).

The main goal is simple: add local PII redaction before prompts, responses, metadata, tool calls, and other observability payloads leave your app.

## Install

```bash
pip install privacy-kit
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

## Supported Integrations

| Framework | Status | API |
| --- | --- | --- |
| Langfuse SDK | Ready | `Langfuse(mask=make_mask())` |
| LangChain / LangGraph | Ready | `make_langfuse_callback()` with `config={"callbacks": [...]}` |
| Pydantic AI | Coming soon | Planned integration example |
| OpenAI SDK | Coming soon | Planned integration example |

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

## Model Config

Default model:

```bash
export PII_MODEL_ID=bardsai/eu-pii-anonimization-multilang
```

Optional cache directory:

```bash
export PII_MODEL_CACHE_DIR=/path/to/model-cache
```

The model is not bundled in the package. It is downloaded on first use and reused from cache after that.

## What Gets Sent Where

privacy-kit runs in your Python process.

For Langfuse/LangChain use cases:

1. Your app receives or creates raw LLM data.
2. Your app may still send raw data to your chosen LLM provider.
3. Before observability export, privacy-kit redacts the Langfuse/LangChain payload.
4. Langfuse receives the redacted payload.

The goal is to keep raw PII out of observability storage.

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

Use this lower-level API when you want entity labels such as `[PERSON_NAME]` or extracted PII mappings. The integration APIs use `[REDACTED]` by default because observability systems usually should not store the PII type either.

## Supported Entity Types

The underlying model recognizes EU-relevant PII categories including:

`PERSON_NAME`, `LOCATION`, `FINANCIAL_AMOUNT`, `PERSON_IDENTIFIER`,
`ORGANIZATION_IDENTIFIER`, `PROPER_NAME`, `RELIGION_OR_BELIEF`,
`TRADE_UNION_MEMBERSHIP`, and more.

## License

Apache-2.0
