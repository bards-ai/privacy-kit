"""Local LLM gateway proxy.

FastAPI app speaking multiple wire formats so AI tools can route through it via
their ``*_BASE_URL`` overrides:

* Anthropic Messages (``/v1/messages``, ``/v1/messages/count_tokens``) — Claude Code
* OpenAI Chat Completions (``/v1/chat/completions``) — Cursor chat, Codex (chat)
* OpenAI Responses (``/v1/responses``) — Codex (default)

Flow: anonymize request text -> forward to the real upstream -> stream the
response back while de-anonymizing placeholders -> log to the audit store.
"""

from privacy_kit.gateway.proxy.app import ForwardResult, build_default_app, create_app

__all__ = ["ForwardResult", "build_default_app", "create_app"]
