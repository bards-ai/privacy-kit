# Examples

These examples are copy-paste starting points for supported integrations.

They are repository examples, not runtime package modules.

## Langfuse SDK

```bash
pip install 'privacy-kit[langfuse]'
python examples/langfuse_mask.py
```

To send to a real Langfuse project, set:

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

## LangSmith SDK

```bash
pip install 'privacy-kit[langsmith]'
python examples/langsmith_anonymizer.py
```

To send to a real LangSmith project, set:

```bash
export LANGSMITH_API_KEY=lsv2_...
```

Masking happens client-side via `Client(anonymizer=...)`; `hide_metadata=True` is set by default because metadata is not routed through the anonymizer.

## LangChain / LangGraph

```bash
pip install 'privacy-kit[langchain]'
export GROQ_API_KEY=gsk_...
python examples/langchain_langgraph_langfuse.py
```

The LangChain example creates a LangGraph-backed agent and sends masked traces to Langfuse through the privacy-kit callback.
