"""Wire-format-aware text transforms for the proxy.

Each request/response transform walks the *known* structure of a given API
format and rewrites only the human-text fields in place, leaving everything else
untouched. ``anon`` and ``deanon`` are ``str -> str`` callables supplied by the
proxy (backed by a shared per-request vault).

Supported formats:

* ``anthropic``         — Anthropic Messages API (Claude Code)
* ``openai_chat``       — OpenAI Chat Completions (Cursor, Codex chat)
* ``openai_responses``  — OpenAI Responses API (Codex default)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

TextFn = Callable[[str], str]


# --- shared helpers ---------------------------------------------------------


def _map_text_blocks(content: Any, fn: TextFn, *, text_key: str = "text") -> Any:
    """Apply ``fn`` to a content value that is either a string or a block list."""
    if isinstance(content, str):
        return fn(content)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get(text_key), str):
                block[text_key] = fn(block[text_key])
    return content


def _walk_strings(value: Any, fn: TextFn) -> Any:
    """Apply ``fn`` to every string value in a JSON-ish structure (keys untouched)."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, list):
        return [_walk_strings(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _walk_strings(item, fn) for key, item in value.items()}
    return value


# --- Anthropic Messages -----------------------------------------------------

# Claude Code's subscription (Max/Pro OAuth) requests carry this exact string as
# the first system block. Anthropic validates it as an anti-spoofing check and
# rejects OAuth requests whose identifier has been altered. It must therefore
# reach the upstream verbatim — but the detector tags "Claude Code"/"Anthropic"
# in it as PERSON_NAME, so we explicitly preserve the identifier preamble and
# anonymize only the text that follows it.
CLAUDE_CODE_SYSTEM_IDENTIFIER = "You are Claude Code, Anthropic's official CLI for Claude."


def _anon_preserving_identifier(text: str, fn: TextFn) -> str:
    """Anonymize ``text`` while keeping a leading Claude Code identifier verbatim."""
    if text.startswith(CLAUDE_CODE_SYSTEM_IDENTIFIER):
        rest = text[len(CLAUDE_CODE_SYSTEM_IDENTIFIER) :]
        return CLAUDE_CODE_SYSTEM_IDENTIFIER + (fn(rest) if rest else "")
    return fn(text)


def _anthropic_system(system: Any, fn: TextFn) -> Any:
    """Rewrite the Anthropic ``system`` field, preserving the Claude Code identifier.

    Subscription/OAuth auth needs the identifier preamble to reach Anthropic
    unchanged; everything after it (the user's CLAUDE.md, environment, etc.) is
    still anonymized. ``system`` is a plain string or a list of text blocks.
    """
    if isinstance(system, str):
        return _anon_preserving_identifier(system, fn)
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                block["text"] = _anon_preserving_identifier(block["text"], fn)
        return system
    return system


def _anthropic_content(content: Any, fn: TextFn) -> Any:
    """Rewrite text in Anthropic content: plain text blocks AND tool blocks.

    Tool blocks carry user data too: ``tool_result`` content is what the
    client's tools read off the local machine (file contents!), and re-sent
    ``tool_use`` inputs may embed values the gateway rehydrated in an earlier
    turn. Skipping them would leak raw PII upstream.
    """
    if isinstance(content, str):
        return fn(content)
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if isinstance(block.get("text"), str):
                block["text"] = fn(block["text"])
            if block.get("type") == "tool_result" and "content" in block:
                block["content"] = _anthropic_content(block["content"], fn)
            elif block.get("type") == "tool_use" and "input" in block:
                block["input"] = _walk_strings(block["input"], fn)
    return content


def anthropic_request(body: dict[str, Any], anon: TextFn) -> None:
    if "system" in body:
        body["system"] = _anthropic_system(body["system"], anon)
        if body.get("system") is None:
            body.pop("system", None)
    for message in body.get("messages", []):
        if isinstance(message, dict) and "content" in message:
            message["content"] = _anthropic_content(message["content"], anon)


def anthropic_response(body: dict[str, Any], deanon: TextFn) -> None:
    _map_text_blocks(body.get("content"), deanon)


# --- OpenAI Chat Completions ------------------------------------------------


def openai_chat_request(body: dict[str, Any], anon: TextFn) -> None:
    for message in body.get("messages", []):
        if not isinstance(message, dict):
            continue
        if "content" in message:
            message["content"] = _map_text_blocks(message["content"], anon)
        # Re-sent assistant turns: tool-call arguments may embed values the
        # gateway rehydrated earlier. Placeholders contain no JSON
        # metacharacters, so rewriting inside the serialized-JSON string keeps
        # it valid.
        for call in message.get("tool_calls") or []:
            fn_block = call.get("function") if isinstance(call, dict) else None
            if isinstance(fn_block, dict) and isinstance(fn_block.get("arguments"), str):
                fn_block["arguments"] = anon(fn_block["arguments"])


def openai_chat_response(body: dict[str, Any], deanon: TextFn) -> None:
    for choice in body.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            message["content"] = deanon(message["content"])


# --- OpenAI Responses -------------------------------------------------------


def openai_responses_request(body: dict[str, Any], anon: TextFn) -> None:
    if isinstance(body.get("instructions"), str):
        body["instructions"] = anon(body["instructions"])
    inp = body.get("input")
    if isinstance(inp, str):
        body["input"] = anon(inp)
    elif isinstance(inp, list):
        for item in inp:
            if not isinstance(item, dict):
                continue
            if "content" in item:
                item["content"] = _map_text_blocks(item["content"], anon)
            # Tool outputs are what Codex's tools read off the local machine;
            # re-sent call arguments may embed previously rehydrated values.
            if item.get("type") == "function_call_output" and isinstance(item.get("output"), str):
                item["output"] = anon(item["output"])
            elif item.get("type") == "function_call" and isinstance(item.get("arguments"), str):
                item["arguments"] = anon(item["arguments"])


def openai_responses_response(body: dict[str, Any], deanon: TextFn) -> None:
    for item in body.get("output", []):
        if isinstance(item, dict):
            _map_text_blocks(item.get("content"), deanon)
    if isinstance(body.get("output_text"), str):
        body["output_text"] = deanon(body["output_text"])


# --- registry + usage extraction --------------------------------------------

REQUEST_TRANSFORMS: dict[str, Callable[[dict[str, Any], TextFn], None]] = {
    "anthropic": anthropic_request,
    "openai_chat": openai_chat_request,
    "openai_responses": openai_responses_request,
}

RESPONSE_TRANSFORMS: dict[str, Callable[[dict[str, Any], TextFn], None]] = {
    "anthropic": anthropic_response,
    "openai_chat": openai_chat_response,
    "openai_responses": openai_responses_response,
}


def extract_tokens(wire: str, body: dict[str, Any]) -> tuple[int | None, int | None]:
    """Pull (input_tokens, output_tokens) from a response's usage block."""
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    assert isinstance(usage, dict)
    if wire == "openai_chat":
        return usage.get("prompt_tokens"), usage.get("completion_tokens")
    return usage.get("input_tokens"), usage.get("output_tokens")
