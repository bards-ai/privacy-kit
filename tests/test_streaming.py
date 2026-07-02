"""Streaming SSE de-anonymization tests. No network, no model."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from privacy_kit.core.types import Span
from privacy_kit.core.vault import Vault
from privacy_kit.gateway.config import Settings
from privacy_kit.gateway.proxy import create_app
from privacy_kit.gateway.proxy.streaming import (
    CapturingStreamDecoder,
    PlaceholderStreamDecoder,
    StreamUsage,
    rewrite_sse,
)
from privacy_kit.gateway.store import AuditStore


def anthropic_delta(text: str) -> str:
    """An Anthropic content_block_delta SSE data line carrying ``text``."""
    obj = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
    return "data: " + json.dumps(obj)


def chat_delta(content: str) -> str:
    """An OpenAI chat-completions SSE data line carrying ``content``."""
    return "data: " + json.dumps({"choices": [{"delta": {"content": content}}]})


def vault_with(mapping: dict[str, str]) -> Vault:
    """Build a vault whose placeholders map to the given real values."""
    vault = Vault()
    for placeholder, value in mapping.items():
        # placeholder like [PERSON_NAME_1] -> recreate via type/value
        etype = placeholder.strip("[]").rsplit("_", 1)[0]
        assert vault.placeholder_for(etype, value) == placeholder
    return vault


def test_decoder_reassembles_split_placeholder() -> None:
    vault = vault_with({"[PERSON_NAME_1]": "John Smith"})
    dec = PlaceholderStreamDecoder(vault)
    out = "".join(dec.feed(tok) for tok in ["Hi ", "[", "PERSON", "_NAME", "_1", "]", "!"])
    out += dec.flush()
    assert out == "Hi John Smith!"


def test_decoder_passes_plain_text_through_promptly() -> None:
    dec = PlaceholderStreamDecoder(Vault())
    assert dec.feed("hello ") == "hello "  # no '[' -> nothing buffered
    assert dec.feed("world") == "world"
    assert dec.flush() == ""


def test_decoder_flushes_trailing_unclosed_bracket_text() -> None:
    # Real text that merely looks like a placeholder start must not be dropped.
    dec = PlaceholderStreamDecoder(Vault())
    emitted = dec.feed("see [TODO")
    assert emitted == "see "  # holds the ambiguous tail
    assert dec.flush() == "[TODO"  # ...but never loses it


def test_capturing_decoder_accumulates_both_renderings() -> None:
    """The capturing decoder records the raw placeholder stream and the
    de-anonymized text, split across deltas exactly like the base decoder."""
    vault = vault_with({"[PERSON_NAME_1]": "John Smith"})
    dec = CapturingStreamDecoder(vault)
    for tok in ["Hi ", "[", "PERSON", "_NAME", "_1", "]", "!"]:
        dec.feed(tok)
    dec.flush()
    assert dec.anonymized_text == "Hi [PERSON_NAME_1]!"
    assert dec.original_text == "Hi John Smith!"


async def _collect(lines: list[str], wire: str, vault: Vault) -> str:
    async def gen() -> AsyncIterator[str]:
        for line in lines:
            yield line

    dec = PlaceholderStreamDecoder(vault)
    return "".join([chunk async for chunk in rewrite_sse(gen(), wire, dec)])


@pytest.mark.asyncio
async def test_rewrite_sse_anthropic_split_across_events() -> None:
    vault = vault_with({"[PERSON_NAME_1]": "John Smith"})
    lines = [
        anthropic_delta("Hi ["),
        "",
        anthropic_delta("PERSON_NAME_1]"),
        "",
        "event: content_block_stop",
        'data: {"type":"content_block_stop","index":0}',
        "",
    ]
    out = await _collect(lines, "anthropic", vault)
    assert "John Smith" in out
    assert "[PERSON_NAME_1]" not in out


@pytest.mark.asyncio
async def test_rewrite_sse_openai_chat_done_passthrough() -> None:
    vault = vault_with({"[EMAIL_ADDRESS_1]": "a@b.com"})
    lines = [
        chat_delta("mail [EMAIL"),
        "",
        chat_delta("_ADDRESS_1]"),
        "",
        "data: [DONE]",
        "",
    ]
    out = await _collect(lines, "openai_chat", vault)
    assert "a@b.com" in out
    assert "[EMAIL_ADDRESS_1]" not in out
    assert "data: [DONE]" in out


class FakeStreamForwarder:
    def __init__(
        self, lines: list[str], status: int = 200, headers: dict[str, str] | None = None
    ) -> None:
        self._lines = lines
        self._status = status
        self._headers = headers or {}
        self.last_payload: dict[str, Any] | None = None

    @asynccontextmanager
    async def open(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> AsyncIterator[tuple[int, dict[str, str], AsyncIterator[str]]]:
        self.last_payload = payload

        async def lines() -> AsyncIterator[str]:
            for line in self._lines:
                yield line

        yield self._status, self._headers, lines()


class LiteralDetector:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for literal, etype in self.mapping.items():
            start = 0
            while (i := text.find(literal, start)) >= 0:
                spans.append(Span(i, i + len(literal), etype, 0.99))
                start = i + len(literal)
        return spans


def test_streaming_end_to_end_through_app(tmp_path: Path) -> None:
    upstream_lines = [
        anthropic_delta("Hello ["),
        "",
        anthropic_delta("PERSON_NAME_1]"),
        "",
        "event: content_block_stop",
        'data: {"type":"content_block_stop","index":0}',
        "",
    ]
    store = AuditStore(tmp_path / "audit.sqlite")
    fwd = FakeStreamForwarder(upstream_lines)
    app = create_app(
        detector=LiteralDetector({"John Smith": "PERSON_NAME"}),
        store=store,
        stream_forwarder=fwd,
        settings=Settings(_env_file=None, policy="pseudonymize"),
    )
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-8",
            "stream": True,
            "messages": [{"role": "user", "content": "I'm John Smith"}],
        },
    )
    assert resp.status_code == 200
    # Upstream got the placeholder, not the real name.
    assert "John Smith" not in str(fwd.last_payload)
    assert "[PERSON_NAME_1]" in str(fwd.last_payload)
    # Client sees the rehydrated stream, never the placeholder.
    assert "John Smith" in resp.text
    assert "[PERSON_NAME_1]" not in resp.text
    # Audited, including the saved request text.
    assert store.summary()["interactions"] == 1
    row = store.recent()[0]
    assert row.id is not None
    texts = store.texts(row.id)
    assert texts and texts[0].original == "I'm John Smith"
    assert texts[0].anonymized == "I'm [PERSON_NAME_1]"
    # The streamed agent response is captured too (turn had PII).
    assistant = [t for t in texts if t.category == "assistant"]
    assert len(assistant) == 1
    assert assistant[0].anonymized == "Hello [PERSON_NAME_1]"
    assert assistant[0].original == "Hello John Smith"


def test_streaming_pii_free_turn_stores_no_response(tmp_path: Path) -> None:
    upstream_lines = [
        anthropic_delta("Hello there"),
        "",
        "event: content_block_stop",
        'data: {"type":"content_block_stop","index":0}',
        "",
    ]
    store = AuditStore(tmp_path / "audit.sqlite")
    app = create_app(
        detector=LiteralDetector({"John Smith": "PERSON_NAME"}),
        store=store,
        stream_forwarder=FakeStreamForwarder(upstream_lines),
        settings=Settings(_env_file=None, policy="pseudonymize", save_texts="anonymized"),
    )
    client = TestClient(app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "stream": True,
            "messages": [{"role": "user", "content": "just say hi"}],
        },
    )
    assert resp.status_code == 200
    row = store.recent()[0]
    assert row.id is not None
    assert all(t.category != "assistant" for t in store.texts(row.id))


# --- hardening: usage capture + upstream error passthrough -------------------


@pytest.mark.asyncio
async def test_usage_extracted_from_anthropic_metadata_events() -> None:
    lines = [
        'data: {"type":"message_start","message":{"usage":{"input_tokens":42}}}',
        "",
        anthropic_delta("hello"),
        "",
        'data: {"type":"message_delta","usage":{"output_tokens":17}}',
        "",
    ]

    async def gen() -> AsyncIterator[str]:
        for line in lines:
            yield line

    usage = StreamUsage()
    async for _ in rewrite_sse(gen(), "anthropic", PlaceholderStreamDecoder(Vault()), usage):
        pass
    assert (usage.input_tokens, usage.output_tokens) == (42, 17)


@pytest.mark.asyncio
async def test_usage_extracted_from_openai_chat_final_chunk() -> None:
    lines = [
        chat_delta("hi"),
        "",
        'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}',
        "",
        "data: [DONE]",
        "",
    ]

    async def gen() -> AsyncIterator[str]:
        for line in lines:
            yield line

    usage = StreamUsage()
    async for _ in rewrite_sse(gen(), "openai_chat", PlaceholderStreamDecoder(Vault()), usage):
        pass
    assert (usage.input_tokens, usage.output_tokens) == (5, 2)


@pytest.mark.asyncio
async def test_usage_extracted_from_responses_completed_event() -> None:
    lines = [
        'data: {"type":"response.output_text.delta","delta":"hi"}',
        "",
        "event: response.completed",
        "data: "
        + json.dumps(
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 9, "output_tokens": 4}},
            }
        ),
        "",
    ]

    async def gen() -> AsyncIterator[str]:
        for line in lines:
            yield line

    usage = StreamUsage()
    async for _ in rewrite_sse(gen(), "openai_responses", PlaceholderStreamDecoder(Vault()), usage):
        pass
    assert (usage.input_tokens, usage.output_tokens) == (9, 4)


def test_streaming_audit_records_token_usage(tmp_path: Path) -> None:
    upstream_lines = [
        'data: {"type":"message_start","message":{"usage":{"input_tokens":42}}}',
        "",
        anthropic_delta("Hello"),
        "",
        'data: {"type":"message_delta","usage":{"output_tokens":17}}',
        "",
    ]
    store = AuditStore(tmp_path / "audit.sqlite")
    app = create_app(
        detector=LiteralDetector({"John Smith": "PERSON_NAME"}),
        store=store,
        stream_forwarder=FakeStreamForwarder(upstream_lines),
    )
    client = TestClient(app)
    resp = client.post(
        "/v1/messages",
        json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    row = store.recent()[0]
    assert (row.input_tokens, row.output_tokens) == (42, 17)


def test_stream_request_upstream_error_passes_status_through(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite")
    fwd = FakeStreamForwarder(
        ['{"error": {"type": "authentication_error", "message": "invalid x-api-key"}}'],
        status=401,
        headers={"content-type": "application/json"},
    )
    app = create_app(
        detector=LiteralDetector({"John Smith": "PERSON_NAME"}),
        store=store,
        stream_forwarder=fwd,
    )
    client = TestClient(app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "stream": True,
            "messages": [{"role": "user", "content": "I'm John Smith"}],
        },
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"
    # The request carried PII upstream (as placeholders) -> still audited,
    # texts included.
    assert store.summary()["interactions"] == 1
    row = store.recent()[0]
    assert row.id is not None
    assert any(t.original == "I'm John Smith" for t in store.texts(row.id))


def test_stream_request_non_sse_json_success_is_rehydrated(tmp_path: Path) -> None:
    # Some upstreams may answer a stream request with a plain JSON body.
    body = {
        "content": [{"type": "text", "text": "Hi [PERSON_NAME_1]"}],
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    store = AuditStore(tmp_path / "audit.sqlite")
    fwd = FakeStreamForwarder(
        [json.dumps(body)], status=200, headers={"content-type": "application/json"}
    )
    app = create_app(
        detector=LiteralDetector({"John Smith": "PERSON_NAME"}),
        store=store,
        stream_forwarder=fwd,
    )
    client = TestClient(app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "stream": True,
            "messages": [{"role": "user", "content": "I'm John Smith"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "Hi John Smith"
    assert store.recent()[0].input_tokens == 3
