# privacy-kit

Ready-to-use privacy blocks for LLM apps and observability pipelines, powered by
[`bardsai/eu-pii-anonimization-multilang`](https://huggingface.co/bardsai/eu-pii-anonimization-multilang).

## Installation

```bash
pip install privacy-kit
```

The default install uses the local ONNX backend. The model is downloaded from Hugging Face on first use and cached locally.

## Langfuse

```python
from langfuse import Langfuse
from privacy_kit.integrations.langfuse import make_mask

langfuse = Langfuse(mask=make_mask())
```

That is the main API. `make_mask()` recursively scans every string Langfuse sends through the mask callback, including nested `input`, `output`, `metadata`, `messages`, tool calls, and custom fields.

Optional tuning:

```python
langfuse = Langfuse(
    mask=make_mask(
        include_paths=["input", "output", "metadata", "messages.*.content"],
        exclude_paths=["metadata.trace_id", "usage"],
    )
)
```

By default, no path config is needed. Everything text-like is scanned.

## LangChain / LangGraph

```bash
pip install 'privacy-kit[langchain]'
```

```python
from langchain.agents import create_agent
from privacy_kit.integrations.langchain import make_langfuse_callback

langfuse_handler = make_langfuse_callback()
agent = create_agent(model="groq:llama-3.1-8b-instant", tools=[])

agent.invoke(
    {"messages": [{"role": "user", "content": "Jan Kowalski, jan.kowalski@example.com"}]},
    config={"callbacks": [langfuse_handler]},
)
```

`create_agent` runs on LangGraph internally, so this is the shortest LangGraph-backed agent path.

## Direct Redaction

```python
from privacy_kit import Redactor

redactor = Redactor()
redactor.redact_text("Kontakt: jan.kowalski@example.com")
# "Kontakt: [REDACTED]"
```

For structured observability payloads:

```python
redactor.redact({
    "input": "Jan Kowalski, jan.kowalski@example.com",
    "metadata": {"phone": "+48 501 222 333"},
})
```

## Supported Integrations

| Framework | Status | How to use |
| --- | --- | --- |
| Langfuse SDK | Ready | `Langfuse(mask=make_mask())` |
| LangChain / LangGraph | Ready | `make_langfuse_callback()` with `config={"callbacks": [...]}` |
| Pydantic AI | Coming soon | Planned example |
| OpenAI SDK | Coming soon | Planned example |

## Model Config

```bash
export PII_MODEL_ID=bardsai/eu-pii-anonimization-multilang
export PII_MODEL_CACHE_DIR=/path/to/model-cache
```

## Transformers Backend

The older `PiiModel` API is still available for label-preserving anonymization and entity extraction:

```bash
pip install 'privacy-kit[transformers]'
```

```python
from privacy_kit import PiiModel

model = PiiModel().from_pretrained("bardsai/eu-pii-anonimization-multilang")
model.anonymize("Anna Wiśniewska mieszka w Warszawie.")
# "[PERSON_NAME] mieszka w [LOCATION]."

model.extract_pii("Anna Wiśniewska mieszka w Warszawie.")
# {"Anna Wiśniewska": "PERSON_NAME", "Warszawie": "LOCATION"}
```

The default integration APIs use `[REDACTED]`. `PiiModel` keeps label-specific placeholders such as `[PERSON_NAME]`.

## License

Apache-2.0
