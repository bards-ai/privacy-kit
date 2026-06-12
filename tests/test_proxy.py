"""Gateway proxy tests. No network, no model: stub detector + fake forwarder."""

from __future__ import annotations

import gzip
import json
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from privacy_kit.core.types import Span
from privacy_kit.gateway.config import Settings
from privacy_kit.gateway.proxy import ForwardResult, create_app
from privacy_kit.gateway.store import AuditStore


class LiteralDetector:
    """Finds every occurrence of known literal -> entity-type pairs."""

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


class CapturingForwarder:
    """Records what was sent upstream and returns a canned response."""

    def __init__(self, factory: Callable[[dict[str, Any]], ForwardResult]) -> None:
        self._factory = factory
        self.last_url: str | None = None
        self.last_headers: dict[str, str] | None = None
        self.last_payload: dict[str, Any] | None = None

    async def __call__(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> ForwardResult:
        self.last_url = url
        self.last_headers = headers
        self.last_payload = payload
        return self._factory(payload)


PII = {"John Smith": "PERSON_NAME", "john@x.com": "EMAIL_ADDRESS"}


def build(
    tmp_path: Path,
    factory: Callable[[dict[str, Any]], ForwardResult],
    settings: Settings | None = None,
) -> tuple[TestClient, CapturingForwarder, AuditStore]:
    store = AuditStore(tmp_path / "audit.sqlite")
    forwarder = CapturingForwarder(factory)
    app = create_app(
        detector=LiteralDetector(PII), store=store, forwarder=forwarder, settings=settings
    )
    return TestClient(app), forwarder, store


def test_healthz(tmp_path: Path) -> None:
    client, _, _ = build(tmp_path, lambda p: ForwardResult(200, {}, {}))
    assert client.get("/healthz").json() == {"status": "ok"}


def test_anthropic_anonymizes_forwards_rehydrates_audits(tmp_path: Path) -> None:
    def factory(payload: dict[str, Any]) -> ForwardResult:
        return ForwardResult(
            200,
            {
                "content": [{"type": "text", "text": "Hi [PERSON_NAME_1] at [EMAIL_ADDRESS_1]."}],
                "usage": {"input_tokens": 12, "output_tokens": 7},
            },
            {},
        )

    client, forwarder, store = build(tmp_path, factory)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-8",
            "system": "You help John Smith.",
            "messages": [{"role": "user", "content": "I'm John Smith, email john@x.com"}],
        },
        headers={"authorization": "Bearer sk-client", "anthropic-version": "2023-06-01"},
    )
    assert resp.status_code == 200

    # Upstream never saw raw PII.
    sent = forwarder.last_payload
    assert sent is not None
    blob = str(sent)
    assert "John Smith" not in blob
    assert "john@x.com" not in blob
    assert "[PERSON_NAME_1]" in blob and "[EMAIL_ADDRESS_1]" in blob

    # Client sees rehydrated values.
    text = resp.json()["content"][0]["text"]
    assert text == "Hi John Smith at john@x.com."

    # Auth passed through; hop-by-hop headers dropped.
    assert forwarder.last_headers is not None
    assert forwarder.last_headers.get("authorization") == "Bearer sk-client"
    assert "content-length" not in {k.lower() for k in forwarder.last_headers}

    # Audited (metadata only).
    summary = store.summary()
    assert summary["interactions"] == 1
    assert summary["entities_by_type"] == {"PERSON_NAME": 1, "EMAIL_ADDRESS": 1}
    assert store.recent()[0].source == "claude-code"
    assert store.recent()[0].input_tokens == 12


def test_anthropic_forwarded_payload_is_fully_anonymized(tmp_path: Path) -> None:
    """Regression guard: the new classification-aware transform must anonymize every
    field that the last-commit single-``anon`` transform anonymized — system, user
    messages, tool results, tool-use inputs — and must leave the Claude Code
    identifier preamble verbatim.
    """
    import copy

    from privacy_kit.gateway.proxy.transform import (
        CLAUDE_CODE_SYSTEM_IDENTIFIER,
        Author,
        anthropic_request,
    )

    def _anon(t: str, _author: Author = Author.MACHINE, _novel: bool = False) -> str:
        for k in ("John Smith", "john@x.com"):
            t = t.replace(k, "[X]")
        return t

    # --- system string with identifier preamble ---
    body = {
        "model": "m",
        "system": f"{CLAUDE_CODE_SYSTEM_IDENTIFIER}\nJohn Smith",
        "messages": [{"role": "user", "content": "ping john@x.com"}],
    }
    anthropic_request(body, _anon)
    assert body["system"].startswith(CLAUDE_CODE_SYSTEM_IDENTIFIER), "identifier must be preserved"
    assert "John Smith" not in body["system"], "system tail must be anonymized"
    assert "john@x.com" not in str(body["messages"])

    # --- multi-turn: system blocks, tool results, and tool-use inputs ---
    body2 = {
        "model": "m",
        "system": [
            {"type": "text", "text": f"{CLAUDE_CODE_SYSTEM_IDENTIFIER}"},
            {"type": "text", "text": "extra John Smith"},
        ],
        "messages": [
            {"role": "user", "content": "turn1 john@x.com"},
            {"role": "assistant", "content": "ok"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "turn2 John Smith"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "1",
                        "content": [{"type": "text", "text": "file: john@x.com"}],
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "bash",
                        "input": {"cmd": "ls john@x.com"},
                    },
                ],
            },
        ],
    }
    body2_orig = copy.deepcopy(body2)
    anthropic_request(body2, _anon)

    blob = str(body2)
    assert "John Smith" not in blob, "PII must not survive in any field"
    assert "john@x.com" not in blob, "PII must not survive in any field"
    # Identifier block preserved verbatim.
    assert body2["system"][0]["text"] == CLAUDE_CODE_SYSTEM_IDENTIFIER  # type: ignore[index]
    # Sanity: something actually changed (anonymizer ran).
    assert body2 != body2_orig


def test_default_mode_saves_only_changed_segments(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, save_texts="anonymized")
    client, _, store = build(tmp_path, lambda p: ForwardResult(200, {"content": []}, {}), settings)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "just hello"},
                {"role": "user", "content": "I'm John Smith, email john@x.com"},
            ],
        },
    )
    assert resp.status_code == 200
    iid = store.recent()[0].id
    assert iid is not None
    rows = store.texts(iid)
    assert len(rows) == 1
    assert rows[0].original == "I'm John Smith, email john@x.com"
    assert "[PERSON_NAME_1]" in rows[0].anonymized
    assert "[EMAIL_ADDRESS_1]" in rows[0].anonymized
    assert "John Smith" not in rows[0].anonymized


def test_all_mode_saves_every_segment(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, save_texts="all")
    client, _, store = build(tmp_path, lambda p: ForwardResult(200, {"content": []}, {}), settings)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "just hello"},
                {"role": "user", "content": "I'm John Smith, email john@x.com"},
            ],
        },
    )
    assert resp.status_code == 200
    iid = store.recent()[0].id
    assert iid is not None
    rows = store.texts(iid)
    assert [(r.seq, r.original == r.anonymized) for r in rows] == [(0, True), (1, False)]
    assert rows[0].original == "just hello"


def test_count_tokens_saves_no_texts(tmp_path: Path) -> None:
    client, _, store = build(tmp_path, lambda p: ForwardResult(200, {"input_tokens": 5}, {}))
    resp = client.post(
        "/v1/messages/count_tokens",
        json={"model": "m", "messages": [{"role": "user", "content": "I'm John Smith"}]},
    )
    assert resp.status_code == 200
    assert store.summary()["interactions"] == 0


def test_openai_chat_round_trip(tmp_path: Path) -> None:
    def factory(payload: dict[str, Any]) -> ForwardResult:
        return ForwardResult(
            200,
            {
                "choices": [{"message": {"role": "assistant", "content": "ok [PERSON_NAME_1]"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
            {},
        )

    client, forwarder, store = build(tmp_path, factory)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5", "messages": [{"role": "user", "content": "hey John Smith"}]},
    )
    assert resp.status_code == 200
    assert "John Smith" not in str(forwarder.last_payload)
    assert resp.json()["choices"][0]["message"]["content"] == "ok John Smith"
    assert store.recent()[0].source == "openai-chat"
    assert store.recent()[0].output_tokens == 2


def test_openai_responses_round_trip(tmp_path: Path) -> None:
    def factory(payload: dict[str, Any]) -> ForwardResult:
        return ForwardResult(
            200,
            {
                "output": [
                    {"content": [{"type": "output_text", "text": "done for [PERSON_NAME_1]"}]}
                ],
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
            {},
        )

    client, forwarder, store = build(tmp_path, factory)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "ping John Smith"}]}
            ],
        },
    )
    assert resp.status_code == 200
    assert "John Smith" not in str(forwarder.last_payload)
    assert resp.json()["output"][0]["content"][0]["text"] == "done for John Smith"
    assert store.recent()[0].source == "codex"


def test_count_tokens_anonymizes_and_forwards(tmp_path: Path) -> None:
    client, forwarder, _ = build(tmp_path, lambda p: ForwardResult(200, {"input_tokens": 9}, {}))
    resp = client.post(
        "/v1/messages/count_tokens",
        json={"model": "m", "messages": [{"role": "user", "content": "John Smith here"}]},
    )
    assert resp.status_code == 200
    assert resp.json() == {"input_tokens": 9}
    assert "John Smith" not in str(forwarder.last_payload)


def test_upstream_error_status_passes_through(tmp_path: Path) -> None:
    client, _, store = build(tmp_path, lambda p: ForwardResult(429, {"error": "rate limited"}, {}))
    resp = client.post(
        "/v1/messages",
        json={"model": "m", "messages": [{"role": "user", "content": "John Smith"}]},
    )
    assert resp.status_code == 429
    # Still audited (the request did contain PII even if upstream failed).
    assert store.summary()["interactions"] == 1


def test_anthropic_tool_blocks_are_anonymized(tmp_path: Path) -> None:
    # tool_result carries what local tools read (file contents!); tool_use in
    # re-sent history may embed previously rehydrated values. Both must be
    # sanitized — this was the leak found by the log-safety audit.
    client, forwarder, _ = build(tmp_path, lambda p: ForwardResult(200, {"content": []}, {}))
    resp = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "read the file"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "write_file",
                            "input": {"path": "a.txt", "content": "email john@x.com"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "content": [{"type": "text", "text": "John Smith <john@x.com>"}],
                        }
                    ],
                },
            ],
        },
    )
    assert resp.status_code == 200
    blob = str(forwarder.last_payload)
    assert "John Smith" not in blob
    assert "john@x.com" not in blob
    assert "[EMAIL_ADDRESS_1]" in blob


def test_anthropic_string_tool_result_is_anonymized(tmp_path: Path) -> None:
    client, forwarder, _ = build(tmp_path, lambda p: ForwardResult(200, {"content": []}, {}))
    client.post(
        "/v1/messages",
        json={
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tu_1", "content": "John Smith"}
                    ],
                }
            ],
        },
    )
    assert "John Smith" not in str(forwarder.last_payload)


def test_openai_chat_tool_call_arguments_are_anonymized(tmp_path: Path) -> None:
    client, forwarder, _ = build(tmp_path, lambda p: ForwardResult(200, {"choices": []}, {}))
    client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "send_mail",
                                "arguments": '{"to": "john@x.com"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "sent to john@x.com"},
            ],
        },
    )
    blob = str(forwarder.last_payload)
    assert "john@x.com" not in blob


def test_openai_responses_tool_items_are_anonymized(tmp_path: Path) -> None:
    client, forwarder, _ = build(tmp_path, lambda p: ForwardResult(200, {"output": []}, {}))
    client.post(
        "/v1/responses",
        json={
            "model": "m",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "grep",
                    "arguments": '{"pattern": "john@x.com"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": "match: John Smith <john@x.com>",
                },
            ],
        },
    )
    blob = str(forwarder.last_payload)
    assert "John Smith" not in blob
    assert "john@x.com" not in blob


def test_claude_code_identifier_preserved_but_rest_anonymized(tmp_path: Path) -> None:
    # Subscription/OAuth requests must reach Anthropic with the identifier intact
    # (it's an anti-spoofing check), while user PII after it is still anonymized.
    # The detector here DOES flag "Claude"/"Anthropic" (as the real model does),
    # so this proves the identifier survives a detection hit.
    store = AuditStore(tmp_path / "audit.sqlite")
    forwarder = CapturingForwarder(lambda p: ForwardResult(200, {"content": []}, {}))
    detector = LiteralDetector(
        {
            "Claude Code": "PERSON_NAME",
            "Claude": "PERSON_NAME",
            "Anthropic": "ORGANIZATION_NAME",
            "John Smith": "PERSON_NAME",
            "john@x.com": "EMAIL_ADDRESS",
        }
    )
    client = TestClient(create_app(detector=detector, store=store, forwarder=forwarder))
    identifier = "You are Claude Code, Anthropic's official CLI for Claude."
    client.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-8",
            "system": [
                {"type": "text", "text": identifier},
                {"type": "text", "text": "The user is John Smith (john@x.com)."},
            ],
            "messages": [{"role": "user", "content": "ping John Smith"}],
        },
    )
    sent = forwarder.last_payload
    assert sent is not None
    # First block reaches upstream byte-for-byte despite Claude/Anthropic hits.
    assert sent["system"][0]["text"] == identifier
    # ...but PII in the second system block and in messages is gone.
    assert sent["system"][1]["text"] == "The user is [PERSON_NAME_1] ([EMAIL_ADDRESS_1])."
    assert "John Smith" not in str(sent["messages"])


def test_claude_code_identifier_as_string_system_is_preserved(tmp_path: Path) -> None:
    client, forwarder, _ = build(tmp_path, lambda p: ForwardResult(200, {"content": []}, {}))
    identifier = "You are Claude Code, Anthropic's official CLI for Claude."
    client.post(
        "/v1/messages",
        json={
            "model": "m",
            "system": f"{identifier}\n\nHelp John Smith.",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    sent = forwarder.last_payload
    assert sent is not None
    assert sent["system"].startswith(identifier)
    assert "John Smith" not in sent["system"]
    assert "[PERSON_NAME_1]" in sent["system"]


def test_custom_source_header_is_honored(tmp_path: Path) -> None:
    client, _, store = build(tmp_path, lambda p: ForwardResult(200, {"content": []}, {}))
    client.post(
        "/v1/messages",
        json={"model": "m", "messages": [{"role": "user", "content": "John Smith"}]},
        headers={"x-privacy-kit-source": "my-cli"},
    )
    assert store.recent()[0].source == "my-cli"


def test_legacy_sieve_source_header_is_honored(tmp_path: Path) -> None:
    client, _, store = build(tmp_path, lambda p: ForwardResult(200, {"content": []}, {}))
    client.post(
        "/v1/messages",
        json={"model": "m", "messages": [{"role": "user", "content": "John Smith"}]},
        headers={"x-sieve-source": "my-cli"},
    )
    assert store.recent()[0].source == "my-cli"


def test_gzip_encoded_request_body_is_decoded(tmp_path: Path) -> None:
    """Claude Code gzips large request bodies; the proxy must decompress them.

    Regression: ``request.json()`` read the raw gzip bytes and 500'd on parse.
    """
    client, forwarder, store = build(
        tmp_path, lambda p: ForwardResult(200, {"content": [], "usage": {}}, {})
    )
    body = json.dumps(
        {"model": "m", "messages": [{"role": "user", "content": "I'm John Smith, john@x.com"}]}
    ).encode()
    resp = client.post(
        "/v1/messages",
        content=gzip.compress(body),
        headers={"content-type": "application/json", "content-encoding": "gzip"},
    )
    assert resp.status_code == 200
    # The body was decoded, anonymized, and forwarded with placeholders.
    blob = str(forwarder.last_payload)
    assert "John Smith" not in blob and "[PERSON_NAME_1]" in blob
    # The stale content-encoding header is not forwarded to the re-serialized body.
    assert "content-encoding" not in {k.lower() for k in forwarder.last_headers or {}}
    assert store.recent()[0].source == "claude-code"


def test_deflate_encoded_request_body_is_decoded(tmp_path: Path) -> None:
    client, forwarder, _ = build(
        tmp_path, lambda p: ForwardResult(200, {"content": [], "usage": {}}, {})
    )
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "John Smith"}]})
    resp = client.post(
        "/v1/messages",
        content=zlib.compress(body.encode()),
        headers={"content-type": "application/json", "content-encoding": "deflate"},
    )
    assert resp.status_code == 200
    assert "John Smith" not in str(forwarder.last_payload)


def test_undecodable_body_returns_400(tmp_path: Path) -> None:
    client, _, _ = build(tmp_path, lambda p: ForwardResult(200, {}, {}))
    resp = client.post(
        "/v1/messages",
        content=b"not gzip at all",
        headers={"content-type": "application/json", "content-encoding": "gzip"},
    )
    assert resp.status_code == 400


# --- novelty-scoped save: history not re-saved on multi-turn requests --------


def test_anthropic_historical_user_messages_not_re_saved(tmp_path: Path) -> None:
    """On turn 2+ the old user messages re-submitted in history must not be re-saved."""
    settings = Settings(_env_file=None, save_texts="all")
    client, forwarder, store = build(
        tmp_path, lambda p: ForwardResult(200, {"content": []}, {}), settings
    )
    resp = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "turn 1: john@x.com"},
                {"role": "assistant", "content": "got it"},
                {"role": "user", "content": "turn 2: John Smith"},
            ],
        },
    )
    assert resp.status_code == 200
    # Both PII values must be stripped before upstream.
    assert "john@x.com" not in str(forwarder.last_payload)
    assert "John Smith" not in str(forwarder.last_payload)
    # Only the NEW user message (turn 2) should be saved.
    iid = store.recent()[0].id
    assert iid is not None
    rows = store.texts(iid)
    originals = [r.original for r in rows]
    assert not any("john@x.com" in o for o in originals), "historical turn 1 re-saved"
    assert any("John Smith" in o for o in originals), "new turn 2 not saved"


def test_openai_chat_historical_user_messages_not_re_saved(tmp_path: Path) -> None:
    """Chat Completions: old user messages in re-submitted history are not re-saved."""
    settings = Settings(_env_file=None, save_texts="all")
    client, forwarder, store = build(
        tmp_path, lambda p: ForwardResult(200, {"choices": []}, {}), settings
    )
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "turn 1: john@x.com"},
                {"role": "assistant", "content": "got it"},
                {"role": "user", "content": "turn 2: John Smith"},
            ],
        },
    )
    assert resp.status_code == 200
    assert "john@x.com" not in str(forwarder.last_payload)
    assert "John Smith" not in str(forwarder.last_payload)
    iid = store.recent()[0].id
    assert iid is not None
    rows = store.texts(iid)
    originals = [r.original for r in rows]
    assert not any("john@x.com" in o for o in originals), "historical turn 1 re-saved"
    assert any("John Smith" in o for o in originals), "new turn 2 not saved"


def test_openai_responses_historical_user_messages_not_re_saved(tmp_path: Path) -> None:
    """Responses API: stateless-mode history (past user turns) is not re-saved."""
    settings = Settings(_env_file=None, save_texts="all")
    client, forwarder, store = build(
        tmp_path, lambda p: ForwardResult(200, {"output": [], "output_text": ""}, {}), settings
    )
    resp = client.post(
        "/v1/responses",
        json={
            "model": "m",
            "input": [
                {"role": "user", "content": "turn 1: john@x.com"},
                {"role": "assistant", "content": "got it"},
                {"role": "user", "content": "turn 2: John Smith"},
            ],
        },
    )
    assert resp.status_code == 200
    assert "john@x.com" not in str(forwarder.last_payload)
    assert "John Smith" not in str(forwarder.last_payload)
    iid = store.recent()[0].id
    assert iid is not None
    rows = store.texts(iid)
    originals = [r.original for r in rows]
    assert not any("john@x.com" in o for o in originals), "historical turn 1 re-saved"
    assert any("John Smith" in o for o in originals), "new turn 2 not saved"


# --- author-scoped save: system/instructions/assistant excluded --------------


def test_anthropic_system_prompt_not_saved(tmp_path: Path) -> None:
    """System prompt is anonymized upstream but must never appear in InteractionText."""
    settings = Settings(_env_file=None, save_texts="all")
    client, forwarder, store = build(
        tmp_path, lambda p: ForwardResult(200, {"content": []}, {}), settings
    )
    resp = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "system": "You help John Smith.",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert resp.status_code == 200
    # System prompt is still anonymized before reaching upstream.
    assert "John Smith" not in str(forwarder.last_payload)
    # Only the user message is saved.
    iid = store.recent()[0].id
    assert iid is not None
    rows = store.texts(iid)
    originals = [r.original for r in rows]
    assert all("John Smith" not in o for o in originals)
    assert any("Hello" in o for o in originals)


def test_anthropic_assistant_turn_not_saved(tmp_path: Path) -> None:
    """Re-sent assistant turns are anonymized but must not appear in InteractionText."""
    settings = Settings(_env_file=None, save_texts="all")
    client, forwarder, store = build(
        tmp_path, lambda p: ForwardResult(200, {"content": []}, {}), settings
    )
    resp = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "messages": [
                {"role": "assistant", "content": "I know John Smith."},
                {"role": "user", "content": "Tell me more"},
            ],
        },
    )
    assert resp.status_code == 200
    assert "John Smith" not in str(forwarder.last_payload)
    iid = store.recent()[0].id
    assert iid is not None
    rows = store.texts(iid)
    originals = [r.original for r in rows]
    assert all("John Smith" not in o for o in originals)
    assert any("Tell me more" in o for o in originals)


def test_anthropic_tool_use_arguments_not_saved(tmp_path: Path) -> None:
    """LLM-authored tool_use inputs are anonymized but must not be saved."""
    settings = Settings(_env_file=None, save_texts="all")
    client, forwarder, store = build(
        tmp_path, lambda p: ForwardResult(200, {"content": []}, {}), settings
    )
    resp = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "send_mail",
                            "input": {"to": "john@x.com"},
                        }
                    ],
                },
                {"role": "user", "content": "ok"},
            ],
        },
    )
    assert resp.status_code == 200
    assert "john@x.com" not in str(forwarder.last_payload)
    iid = store.recent()[0].id
    assert iid is not None
    rows = store.texts(iid)
    assert all("john@x.com" not in r.original for r in rows)
    assert all("john@x.com" not in r.anonymized for r in rows)


def test_anthropic_tool_result_data_is_saved(tmp_path: Path) -> None:
    """tool_result content (file/command data) must be both anonymized and saved."""
    settings = Settings(_env_file=None, save_texts="anonymized")
    client, forwarder, store = build(
        tmp_path, lambda p: ForwardResult(200, {"content": []}, {}), settings
    )
    resp = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "content": [{"type": "text", "text": "Owner: John Smith"}],
                        }
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200
    # Raw value stripped before upstream.
    assert "John Smith" not in str(forwarder.last_payload)
    # Segment must be saved because it contains PII and is savable.
    iid = store.recent()[0].id
    assert iid is not None
    rows = store.texts(iid)
    assert len(rows) == 1
    assert "John Smith" in rows[0].original
    assert "John Smith" not in rows[0].anonymized


def test_openai_chat_system_message_not_saved(tmp_path: Path) -> None:
    """OpenAI chat system-role messages are anonymized but must not be saved."""
    settings = Settings(_env_file=None, save_texts="all")
    client, forwarder, store = build(
        tmp_path, lambda p: ForwardResult(200, {"choices": []}, {}), settings
    )
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {"role": "system", "content": "Help John Smith."},
                {"role": "user", "content": "Hi"},
            ],
        },
    )
    assert resp.status_code == 200
    assert "John Smith" not in str(forwarder.last_payload)
    iid = store.recent()[0].id
    assert iid is not None
    rows = store.texts(iid)
    originals = [r.original for r in rows]
    assert all("John Smith" not in o for o in originals)
    assert any("Hi" in o for o in originals)


def test_openai_chat_tool_call_arguments_not_saved(tmp_path: Path) -> None:
    """LLM tool_calls arguments are anonymized but must not be saved."""
    settings = Settings(_env_file=None, save_texts="all")
    client, forwarder, store = build(
        tmp_path, lambda p: ForwardResult(200, {"choices": []}, {}), settings
    )
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "send_mail",
                                "arguments": '{"to": "john@x.com"}',
                            },
                        }
                    ],
                },
                {"role": "user", "content": "done"},
            ],
        },
    )
    assert resp.status_code == 200
    assert "john@x.com" not in str(forwarder.last_payload)
    iid = store.recent()[0].id
    assert iid is not None
    rows = store.texts(iid)
    assert all("john@x.com" not in r.original for r in rows)


def test_openai_chat_tool_role_data_is_saved(tmp_path: Path) -> None:
    """tool-role messages (function outputs) are savable data segments."""
    settings = Settings(_env_file=None, save_texts="anonymized")
    client, forwarder, store = build(
        tmp_path, lambda p: ForwardResult(200, {"choices": []}, {}), settings
    )
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {"role": "tool", "tool_call_id": "c1", "content": "sent to john@x.com"},
            ],
        },
    )
    assert resp.status_code == 200
    assert "john@x.com" not in str(forwarder.last_payload)
    iid = store.recent()[0].id
    assert iid is not None
    rows = store.texts(iid)
    assert len(rows) == 1
    assert "john@x.com" in rows[0].original
    assert "john@x.com" not in rows[0].anonymized


def test_openai_responses_instructions_not_saved(tmp_path: Path) -> None:
    """Responses API instructions (system prompt) are anonymized but not saved."""
    settings = Settings(_env_file=None, save_texts="all")
    client, forwarder, store = build(
        tmp_path, lambda p: ForwardResult(200, {"output": [], "output_text": ""}, {}), settings
    )
    resp = client.post(
        "/v1/responses",
        json={
            "model": "m",
            "instructions": "Help John Smith.",
            "input": "Hello",
        },
    )
    assert resp.status_code == 200
    assert "John Smith" not in str(forwarder.last_payload)
    iid = store.recent()[0].id
    assert iid is not None
    rows = store.texts(iid)
    originals = [r.original for r in rows]
    assert all("John Smith" not in o for o in originals)
    assert any("Hello" in o for o in originals)


def test_openai_responses_function_call_output_saved(tmp_path: Path) -> None:
    """function_call_output data (tool return values) is a savable segment."""
    settings = Settings(_env_file=None, save_texts="anonymized")
    client, forwarder, store = build(
        tmp_path, lambda p: ForwardResult(200, {"output": [], "output_text": ""}, {}), settings
    )
    resp = client.post(
        "/v1/responses",
        json={
            "model": "m",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": "Result for John Smith",
                }
            ],
        },
    )
    assert resp.status_code == 200
    assert "John Smith" not in str(forwarder.last_payload)
    iid = store.recent()[0].id
    assert iid is not None
    rows = store.texts(iid)
    assert len(rows) == 1
    assert "John Smith" in rows[0].original
    assert "John Smith" not in rows[0].anonymized


def test_openai_responses_function_call_arguments_not_saved(tmp_path: Path) -> None:
    """function_call arguments (LLM-authored) are anonymized but not saved."""
    settings = Settings(_env_file=None, save_texts="all")
    client, forwarder, store = build(
        tmp_path, lambda p: ForwardResult(200, {"output": [], "output_text": ""}, {}), settings
    )
    resp = client.post(
        "/v1/responses",
        json={
            "model": "m",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "lookup",
                    "arguments": '{"email": "john@x.com"}',
                }
            ],
        },
    )
    assert resp.status_code == 200
    assert "john@x.com" not in str(forwarder.last_payload)
    iid = store.recent()[0].id
    assert iid is not None
    rows = store.texts(iid)
    assert all("john@x.com" not in r.original for r in rows)

