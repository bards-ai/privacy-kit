"""Streaming (SSE) de-anonymization for the proxy.

Placeholders such as ``[PERSON_NAME_1]`` are emitted by the upstream model across
several SSE deltas (``[``, ``PERSON``, ``_NAME``, ``_1``, ``]``). We therefore
buffer a trailing fragment that *could* still become a placeholder and only emit
text once it's safe to de-anonymize, carrying the fragment to the next delta.

``rewrite_sse`` walks the upstream SSE lines, rewrites the text-delta fields per
wire format, and injects any buffered remainder as a synthetic delta event before
the stream's terminal marker so no trailing text is ever dropped.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from privacy_kit.core.vault import Vault, deanonymize

# Characters that may appear *inside* a placeholder, i.e. between "[" and "]".
_INTERIOR = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


@dataclass
class StreamUsage:
    """Token usage gathered from a stream's metadata events.

    Anthropic reports input tokens in ``message_start`` and output tokens in
    ``message_delta``; OpenAI chat sends a final ``usage`` chunk (only with
    ``stream_options.include_usage``); Responses carries it on
    ``response.completed``. Fields stay ``None`` when upstream never reports.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None

    def take(self, usage: Any, in_key: str, out_key: str) -> None:
        """Absorb a usage dict, keeping the latest reported values."""
        if not isinstance(usage, dict):
            return
        if isinstance(usage.get(in_key), int):
            self.input_tokens = usage[in_key]
        if isinstance(usage.get(out_key), int):
            self.output_tokens = usage[out_key]


class PlaceholderStreamDecoder:
    """Incrementally de-anonymizes a text stream, buffering partial placeholders."""

    def __init__(self, vault: Vault) -> None:
        self._vault = vault
        self._pending = ""

    def feed(self, delta: str) -> str:
        """Add a text delta; return the de-anonymized, safe-to-emit prefix."""
        self._pending += delta
        safe, held = self._split(self._pending)
        self._pending = held
        return deanonymize(safe, self._vault)

    def flush(self) -> str:
        """Return whatever remains (end of stream), de-anonymized."""
        out = deanonymize(self._pending, self._vault)
        self._pending = ""
        return out

    @staticmethod
    def _split(text: str) -> tuple[str, str]:
        """Split into (emit-now, hold-back). Holds a trailing partial placeholder."""
        idx = text.rfind("[")
        if idx == -1:
            return text, ""
        after = text[idx + 1 :]
        if "]" in after:  # the last "[" is already closed -> nothing to hold
            return text, ""
        if all(c in _INTERIOR for c in after):  # could still grow into a placeholder
            return text[:idx], text[idx:]
        return text, ""


class CapturingStreamDecoder(PlaceholderStreamDecoder):
    """A :class:`PlaceholderStreamDecoder` that also accumulates the full text.

    ``anonymized_text`` is the raw upstream stream (placeholders intact);
    ``original_text`` is the de-anonymized rendering the client received. The
    proxy uses these to persist the agent's response for the conversation view,
    without changing how the stream is rewritten.
    """

    def __init__(self, vault: Vault) -> None:
        super().__init__(vault)
        self._raw_parts: list[str] = []
        self._out_parts: list[str] = []

    def feed(self, delta: str) -> str:
        self._raw_parts.append(delta)
        out = super().feed(delta)
        self._out_parts.append(out)
        return out

    def flush(self) -> str:
        out = super().flush()
        self._out_parts.append(out)
        return out

    @property
    def anonymized_text(self) -> str:
        return "".join(self._raw_parts)

    @property
    def original_text(self) -> str:
        return "".join(self._out_parts)


# --- per-format delta rewriting + flush injection ---------------------------


def _rewrite_anthropic(obj: dict[str, Any], dec: PlaceholderStreamDecoder) -> None:
    if obj.get("type") == "content_block_delta":
        delta = obj.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("text"), str):
            delta["text"] = dec.feed(delta["text"])


def _rewrite_openai_chat(obj: dict[str, Any], dec: PlaceholderStreamDecoder) -> None:
    for choice in obj.get("choices", []):
        delta = choice.get("delta") if isinstance(choice, dict) else None
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            delta["content"] = dec.feed(delta["content"])


def _rewrite_openai_responses(obj: dict[str, Any], dec: PlaceholderStreamDecoder) -> None:
    if obj.get("type") == "response.output_text.delta" and isinstance(obj.get("delta"), str):
        obj["delta"] = dec.feed(obj["delta"])


def _usage_anthropic(obj: dict[str, Any], usage: StreamUsage) -> None:
    message = obj.get("message")
    block = message.get("usage") if isinstance(message, dict) else obj.get("usage")
    usage.take(block, "input_tokens", "output_tokens")


def _usage_openai_chat(obj: dict[str, Any], usage: StreamUsage) -> None:
    usage.take(obj.get("usage"), "prompt_tokens", "completion_tokens")


def _usage_openai_responses(obj: dict[str, Any], usage: StreamUsage) -> None:
    response = obj.get("response")
    if isinstance(response, dict):
        usage.take(response.get("usage"), "input_tokens", "output_tokens")


def _flush_anthropic(text: str, index: int) -> str:
    data = json.dumps(
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        }
    )
    return f"event: content_block_delta\ndata: {data}\n\n"


def _flush_openai_chat(text: str, index: int) -> str:
    data = json.dumps({"choices": [{"index": 0, "delta": {"content": text}}]})
    return f"data: {data}\n\n"


def _flush_openai_responses(text: str, index: int) -> str:
    data = json.dumps({"type": "response.output_text.delta", "delta": text})
    return f"event: response.output_text.delta\ndata: {data}\n\n"


@dataclass(frozen=True)
class _Format:
    rewrite: Callable[[dict[str, Any], PlaceholderStreamDecoder], None]
    flush_block: Callable[[str, int], str]
    inject_before: Callable[[str], bool]
    is_done: Callable[[str], bool]
    extract_usage: Callable[[dict[str, Any], StreamUsage], None]


_FORMATS: dict[str, _Format] = {
    "anthropic": _Format(
        rewrite=_rewrite_anthropic,
        flush_block=_flush_anthropic,
        inject_before=lambda line: line.startswith("event: content_block_stop"),
        is_done=lambda payload: False,
        extract_usage=_usage_anthropic,
    ),
    "openai_chat": _Format(
        rewrite=_rewrite_openai_chat,
        flush_block=_flush_openai_chat,
        inject_before=lambda line: line.strip() == "data: [DONE]",
        is_done=lambda payload: payload == "[DONE]",
        extract_usage=_usage_openai_chat,
    ),
    "openai_responses": _Format(
        rewrite=_rewrite_openai_responses,
        flush_block=_flush_openai_responses,
        inject_before=lambda line: line.startswith("event: response.completed"),
        is_done=lambda payload: False,
        extract_usage=_usage_openai_responses,
    ),
}


async def rewrite_sse(
    lines: AsyncIterator[str],
    wire: str,
    decoder: PlaceholderStreamDecoder,
    usage: StreamUsage | None = None,
) -> AsyncIterator[str]:
    """Yield rewritten SSE text from upstream ``lines`` for the given wire format.

    If ``usage`` is given, token counts found in the stream's metadata events
    are recorded on it as a side effect.
    """
    fmt = _FORMATS[wire]
    block_index = 0
    flushed = False

    async for line in lines:
        if not flushed and fmt.inject_before(line):
            remainder = decoder.flush()
            if remainder:
                yield fmt.flush_block(remainder, block_index)
            flushed = True

        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
            if fmt.is_done(payload):
                yield line + "\n"
                continue
            try:
                obj = json.loads(payload)
            except ValueError:
                yield line + "\n"
                continue
            if isinstance(obj, dict):
                if isinstance(obj.get("index"), int):
                    block_index = obj["index"]
                if usage is not None:
                    fmt.extract_usage(obj, usage)
                fmt.rewrite(obj, decoder)
                yield "data: " + json.dumps(obj) + "\n"
            else:
                yield line + "\n"
        else:
            yield line + "\n"

    if not flushed:
        remainder = decoder.flush()
        if remainder:
            yield fmt.flush_block(remainder, block_index)
