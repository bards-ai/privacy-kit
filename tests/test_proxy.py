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
    # The default policy is "monitor" (forward originals); these tests assert the
    # pseudonymize behavior, so default to it unless the caller overrides.
    app = create_app(
        detector=LiteralDetector(PII),
        store=store,
        forwarder=forwarder,
        settings=settings or Settings(_env_file=None, policy="pseudonymize"),
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
    assert store.recent()[0].policy == "pseudonymize"


def test_conversation_id_groups_turns_of_one_conversation(tmp_path: Path) -> None:
    """Two turns that share a Claude Code session id land under one conversation
    id; a different session id gets a different one."""
    client, _, store = build(tmp_path, lambda p: ForwardResult(200, {"content": []}, {}))

    def metadata(session_id: str) -> dict:
        return {"user_id": json.dumps({"device_id": "d1", "session_id": session_id})}

    # Turn 1 of conversation A.
    client.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-8",
            "metadata": metadata("sess-a"),
            "messages": [{"role": "user", "content": "Let's start project X"}],
        },
    )
    # Turn 2 of conversation A: history resent, plus the new turn.
    client.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-8",
            "metadata": metadata("sess-a"),
            "messages": [
                {"role": "user", "content": "Let's start project X"},
                {"role": "assistant", "content": "Sure"},
                {"role": "user", "content": "Add a feature"},
            ],
        },
    )
    # A separate conversation (e.g. after /clear): new session id.
    client.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-8",
            "metadata": metadata("sess-b"),
            "messages": [{"role": "user", "content": "Totally different topic"}],
        },
    )

    rows = sorted(store.recent(), key=lambda r: r.id or 0)
    assert rows[0].conversation_id is not None
    assert rows[0].conversation_id == rows[1].conversation_id  # same conversation
    assert rows[2].conversation_id != rows[0].conversation_id  # different one

    # The grouping surfaces through the store's conversation view.
    convs, total = store.list_conversations()
    assert total == 2
    turn_counts = sorted(c["turn_count"] for c in convs)
    assert turn_counts == [1, 2]


def test_monitor_mode_forwards_original_but_logs_detection(tmp_path: Path) -> None:
    """In monitor mode (the default) the prompt is forwarded unchanged — real PII
    reaches the upstream — but the detection is still logged: entity counts, the
    would-be-anonymized text, and ``policy="monitor"``."""
    settings = Settings(_env_file=None, policy="monitor", save_texts="all")
    client, forwarder, store = build(
        tmp_path, lambda p: ForwardResult(200, {"content": []}, {}), settings
    )
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-8",
            "messages": [{"role": "user", "content": "I'm John Smith, email john@x.com"}],
        },
    )
    assert resp.status_code == 200

    # Upstream received the ORIGINAL values, not placeholders.
    sent = str(forwarder.last_payload)
    assert "John Smith" in sent and "john@x.com" in sent
    assert "[PERSON_NAME_1]" not in sent

    # ...but the detection was still logged.
    row = store.recent()[0]
    assert row.policy == "monitor"
    assert row.entity_counts == {"PERSON_NAME": 1, "EMAIL_ADDRESS": 1}
    assert row.id is not None
    rows = store.texts(row.id)
    assert len(rows) == 1
    assert rows[0].original == "I'm John Smith, email john@x.com"
    # The anonymized column still records what *would* have been redacted.
    assert "[PERSON_NAME_1]" in rows[0].anonymized
    assert "[EMAIL_ADDRESS_1]" in rows[0].anonymized


def test_monitor_mode_is_the_default(tmp_path: Path) -> None:
    """With no explicit settings, the proxy forwards originals (monitor default)."""
    # build() pins pseudonymize, so construct the app directly with default settings.
    store = AuditStore(tmp_path / "audit.sqlite")
    forwarder = CapturingForwarder(lambda p: ForwardResult(200, {"content": []}, {}))
    app = create_app(
        detector=LiteralDetector(PII),
        store=store,
        forwarder=forwarder,
        settings=Settings(_env_file=None),
    )
    resp = TestClient(app).post(
        "/v1/messages",
        json={"model": "m", "messages": [{"role": "user", "content": "hi John Smith"}]},
    )
    assert resp.status_code == 200
    assert "John Smith" in str(forwarder.last_payload)
    assert store.recent()[0].policy == "monitor"


def test_monitor_mode_count_tokens_forwards_original(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, policy="monitor")
    client, forwarder, _ = build(
        tmp_path, lambda p: ForwardResult(200, {"input_tokens": 9}, {}), settings
    )
    resp = client.post(
        "/v1/messages/count_tokens",
        json={"model": "m", "messages": [{"role": "user", "content": "John Smith here"}]},
    )
    assert resp.status_code == 200
    assert "John Smith" in str(forwarder.last_payload)


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
    system = body["system"]
    assert isinstance(system, str)
    assert system.startswith(CLAUDE_CODE_SYSTEM_IDENTIFIER), "identifier must be preserved"
    assert "John Smith" not in system, "system tail must be anonymized"
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
    settings = Settings(_env_file=None, save_texts="anonymized", policy="pseudonymize")
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
    settings = Settings(_env_file=None, save_texts="all", policy="pseudonymize")
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


def test_agent_response_saved_when_turn_has_pii(tmp_path: Path) -> None:
    """A turn whose prompt contained PII stores the agent's response as an
    ``assistant`` segment (original rehydrated, anonymized placeholders)."""

    def factory(payload: dict[str, Any]) -> ForwardResult:
        return ForwardResult(
            200,
            {"content": [{"type": "text", "text": "Hi [PERSON_NAME_1], noted."}]},
            {},
        )

    client, _, store = build(tmp_path, factory)
    resp = client.post(
        "/v1/messages",
        json={"model": "m", "messages": [{"role": "user", "content": "I'm John Smith"}]},
    )
    assert resp.status_code == 200
    iid = store.recent()[0].id
    assert iid is not None
    assistant = [t for t in store.texts(iid) if t.category == "assistant"]
    assert len(assistant) == 1
    assert assistant[0].anonymized == "Hi [PERSON_NAME_1], noted."
    assert assistant[0].original == "Hi John Smith, noted."


def test_agent_response_not_saved_when_turn_pii_free(tmp_path: Path) -> None:
    """Under ``save_texts="anonymized"`` a PII-free turn stores no agent response."""

    def factory(payload: dict[str, Any]) -> ForwardResult:
        return ForwardResult(200, {"content": [{"type": "text", "text": "Hello there!"}]}, {})

    settings = Settings(_env_file=None, policy="pseudonymize", save_texts="anonymized")
    client, _, store = build(tmp_path, factory, settings=settings)
    resp = client.post(
        "/v1/messages",
        json={"model": "m", "messages": [{"role": "user", "content": "just say hi"}]},
    )
    assert resp.status_code == 200
    iid = store.recent()[0].id
    assert iid is not None
    assert all(t.category != "assistant" for t in store.texts(iid))


def test_agent_response_saved_pii_free_when_save_all(tmp_path: Path) -> None:
    """Under ``save_texts="all"`` a PII-free turn still stores the agent's
    response, so full conversations are captured even with no PII (e.g. the
    "null" detector). This is the save-everything conversation mode."""

    def factory(payload: dict[str, Any]) -> ForwardResult:
        return ForwardResult(200, {"content": [{"type": "text", "text": "Hello there!"}]}, {})

    settings = Settings(_env_file=None, policy="monitor", save_texts="all")
    client, _, store = build(tmp_path, factory, settings=settings)
    resp = client.post(
        "/v1/messages",
        json={"model": "m", "messages": [{"role": "user", "content": "just say hi"}]},
    )
    assert resp.status_code == 200
    iid = store.recent()[0].id
    assert iid is not None
    assistant = [t for t in store.texts(iid) if t.category == "assistant"]
    assert len(assistant) == 1
    assert assistant[0].original == "Hello there!"


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
    client = TestClient(
        create_app(
            detector=detector,
            store=store,
            forwarder=forwarder,
            settings=Settings(_env_file=None, policy="pseudonymize"),
        )
    )
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
    settings = Settings(_env_file=None, save_texts="all", policy="pseudonymize")
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
    settings = Settings(_env_file=None, save_texts="all", policy="pseudonymize")
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
    settings = Settings(_env_file=None, save_texts="all", policy="pseudonymize")
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


def test_entity_count_scoped_to_new_turn_not_resent_history(tmp_path: Path) -> None:
    """entity_total counts only this turn's PII, not the whole re-sent conversation.

    The vault is shared across the request for placeholder consistency, but the
    audited count must describe the new turn alone — otherwise every turn
    re-tallies the growing history and the dashboard shows an inflated number.
    A value that also appears in the (re-sent, machine-authored) system prompt
    still counts, because the user genuinely introduced it this turn.
    """
    client, _, store = build(tmp_path, lambda p: ForwardResult(200, {"content": []}, {}))
    resp = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "system": "You help John Smith.",
            "messages": [
                {"role": "user", "content": "turn 1: john@x.com"},  # history
                {"role": "assistant", "content": "got it"},
                {"role": "user", "content": "turn 2: John Smith"},  # new turn
            ],
        },
    )
    assert resp.status_code == 200
    rec = store.recent()[0]
    # Only the new turn's PERSON_NAME — not the historical EMAIL_ADDRESS.
    assert rec.entity_counts == {"PERSON_NAME": 1}
    assert rec.entity_total == 1


def test_saved_placeholder_numbering_scoped_to_new_turn_not_resent_history(
    tmp_path: Path,
) -> None:
    """Saved [TYPE_N] numbering counts only this turn's novel PII, not the history.

    The forward vault's counter runs through the whole re-sent conversation, so a
    new name on a later turn would otherwise be saved as a high [PERSON_NAME_N]
    that mismatches the per-turn entity_counts. The saved segment must read 1..N
    over just what was new this turn.
    """
    settings = Settings(_env_file=None, save_texts="all", policy="pseudonymize")
    store = AuditStore(tmp_path / "audit.sqlite")
    forwarder = CapturingForwarder(lambda p: ForwardResult(200, {"content": []}, {}))
    app = create_app(
        detector=LiteralDetector({"John Smith": "PERSON_NAME", "Jane Doe": "PERSON_NAME"}),
        store=store,
        forwarder=forwarder,
        settings=settings,
    )
    client = TestClient(app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "turn 1: John Smith"},  # history → forward _1
                {"role": "assistant", "content": "got it"},
                {"role": "user", "content": "turn 2: Jane Doe"},  # new turn
            ],
        },
    )
    assert resp.status_code == 200
    iid = store.recent()[0].id
    assert iid is not None
    rows = store.texts(iid)
    saved = next(r for r in rows if "Jane Doe" in (r.original or ""))
    # Numbered for this turn alone, not [PERSON_NAME_2] from the re-sent history.
    assert "[PERSON_NAME_1]" in saved.anonymized
    assert "[PERSON_NAME_2]" not in saved.anonymized


# --- author-scoped save: system/instructions/assistant excluded --------------


def test_anthropic_system_prompt_not_saved(tmp_path: Path) -> None:
    """System prompt is anonymized upstream but must never appear in InteractionText."""
    settings = Settings(_env_file=None, save_texts="all", policy="pseudonymize")
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
    settings = Settings(_env_file=None, save_texts="all", policy="pseudonymize")
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
    settings = Settings(_env_file=None, save_texts="all", policy="pseudonymize")
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
    settings = Settings(_env_file=None, save_texts="anonymized", policy="pseudonymize")
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
    # Origin tag: file/command data is "tool", not "user".
    assert rows[0].category == "tool"


def test_anthropic_segments_tagged_by_origin(tmp_path: Path) -> None:
    """Saved segments record whether they are user-typed text or tool/file data."""
    settings = Settings(_env_file=None, save_texts="all", policy="pseudonymize")
    client, _forwarder, store = build(
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
                        {"type": "text", "text": "Reply to Jane Doe."},
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "content": [{"type": "text", "text": "Owner: John Smith"}],
                        },
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200
    iid = store.recent()[0].id
    assert iid is not None
    by_origin = {r.category: r.original for r in store.texts(iid)}
    assert "Jane Doe" in by_origin["user"]
    assert "John Smith" in by_origin["tool"]


def test_anthropic_injected_system_reminder_not_saved(tmp_path: Path) -> None:
    """Claude Code embeds <system-reminder> context in user turns; those blocks
    are anonymized for upstream but must never be saved — only the real message."""
    settings = Settings(_env_file=None, save_texts="all", policy="pseudonymize")
    client, forwarder, store = build(
        tmp_path, lambda p: ForwardResult(200, {"content": []}, {}), settings
    )
    reminder = "<system-reminder>\nThe user's email is john@x.com.\n</system-reminder>"
    resp = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": reminder},
                        {"type": "text", "text": "Draft a reply to John Smith."},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200
    # PII inside the reminder must not leak upstream.
    assert "john@x.com" not in str(forwarder.last_payload)
    iid = store.recent()[0].id
    assert iid is not None
    originals = [r.original for r in store.texts(iid)]
    # The injected reminder block is not saved...
    assert all("system-reminder" not in o for o in originals)
    assert all("john@x.com" not in o for o in originals)
    # ...but the genuine user message is.
    assert any("Draft a reply" in o for o in originals)


def test_is_injected_system_text_patterns() -> None:
    """The detector flags every harness-injection shape but not genuine messages."""
    from privacy_kit.gateway.proxy.transform import _is_injected_system_text as inj

    # Injected: tag-wrapped blocks, flattened transcripts, bracket directives.
    assert inj("<system-reminder>\navailable tools\n</system-reminder>")
    assert inj("<session>\nLet it check this window\n</session>")
    assert inj("User: <local-command-stdout>Set model to claude-opus-4-8</local-command-stdout>\n")
    assert inj("<ide_selection>some code</ide_selection>")
    assert inj("[SUGGESTION MODE: Suggest what the user might type next]")
    # Multiple harness wrappers concatenated into one block (Claude Code slash
    # commands) and a leading wrapper followed by trailing text — these open with
    # a known harness tag even though no single tag spans the whole string.
    assert inj(
        "<command-name>/clear</command-name>\n"
        "<command-message>clear</command-message>\n"
        "<command-args></command-args>\n"
        "<local-command-stdout></local-command-stdout>"
    )
    assert inj("<ide_selection>x = 1</ide_selection>\n<system-reminder>note</system-reminder>")
    assert inj("<system-reminder>Plan mode is active</system-reminder>\nactually do the thing")
    # Genuine user messages and tool/file output must NOT be flagged — including a
    # message that merely starts with markup that is not a harness wrapper tag.
    assert not inj("Draft a reply to john@x.com confirming their account.")
    assert not inj("=== app.py 300-320 ===\n    return JSONResponse(...)")
    assert not inj("<div>hello</div> and then some more text I actually typed")
    assert not inj("")


def test_openai_chat_system_message_not_saved(tmp_path: Path) -> None:
    """OpenAI chat system-role messages are anonymized but must not be saved."""
    settings = Settings(_env_file=None, save_texts="all", policy="pseudonymize")
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
    settings = Settings(_env_file=None, save_texts="all", policy="pseudonymize")
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
    settings = Settings(_env_file=None, save_texts="anonymized", policy="pseudonymize")
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
    settings = Settings(_env_file=None, save_texts="all", policy="pseudonymize")
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
    settings = Settings(_env_file=None, save_texts="anonymized", policy="pseudonymize")
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


def test_cursor_hook_monitor_allows_and_audits(tmp_path: Path) -> None:
    # Default monitor: detect + record, but always allow (hooks can't redact).
    settings = Settings(_env_file=None, save_texts="all", policy="monitor")
    client, _, store = build(tmp_path, lambda p: ForwardResult(200, {}, {}), settings)
    resp = client.post(
        "/v1/cursor-hook",
        json={
            "hook_event_name": "beforeSubmitPrompt",
            "prompt": "email John Smith at john@x.com",
            "model": "composer-2.5",
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"continue": True}

    recent = store.recent()
    assert len(recent) == 1
    assert recent[0].source == "cursor"
    assert recent[0].wire_format == "cursor:beforeSubmitPrompt"
    assert recent[0].entity_total == 2  # John Smith + john@x.com


def test_cursor_hook_before_read_file_uses_permission_shape(tmp_path: Path) -> None:
    client, _, _ = build(tmp_path, lambda p: ForwardResult(200, {}, {}))
    resp = client.post(
        "/v1/cursor-hook",
        json={"hook_event_name": "beforeReadFile", "content": "no pii here", "model": "m"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"permission": "allow"}


def test_cursor_hook_blocks_on_pii_when_enabled(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, policy="monitor", cursor_block=True)
    client, _, _ = build(tmp_path, lambda p: ForwardResult(200, {}, {}), settings)
    resp = client.post(
        "/v1/cursor-hook",
        json={"hook_event_name": "beforeSubmitPrompt", "prompt": "ping John Smith", "model": "m"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["continue"] is False
    assert "PERSON_NAME" in body["user_message"]


def test_cursor_hook_block_allows_clean_prompt(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, policy="monitor", cursor_block=True)
    client, _, _ = build(tmp_path, lambda p: ForwardResult(200, {}, {}), settings)
    resp = client.post(
        "/v1/cursor-hook",
        json={"hook_event_name": "beforeSubmitPrompt", "prompt": "no pii at all", "model": "m"},
    )
    assert resp.json() == {"continue": True}


def test_cursor_hook_unknown_event_allows(tmp_path: Path) -> None:
    client, _, store = build(tmp_path, lambda p: ForwardResult(200, {}, {}))
    resp = client.post("/v1/cursor-hook", json={"hook_event_name": "sessionStart"})
    assert resp.status_code == 200
    assert resp.json() == {"continue": True}
    assert store.recent() == []  # nothing scanned, nothing recorded


def test_cursor_hook_after_agent_response_save_texts_all(tmp_path: Path) -> None:
    """afterAgentResponse always saves the assistant text when save_texts="all"."""
    settings = Settings(_env_file=None, save_texts="all", policy="monitor")
    client, _, store = build(tmp_path, lambda p: ForwardResult(200, {}, {}), settings)
    # Trigger a preceding prompt hook so the conversation exists.
    client.post(
        "/v1/cursor-hook",
        json={
            "hook_event_name": "beforeSubmitPrompt",
            "prompt": "fix the bug",
            "model": "composer-2.5",
            "conversation_id": "conv-abc",
        },
    )
    resp = client.post(
        "/v1/cursor-hook",
        json={
            "hook_event_name": "afterAgentResponse",
            "text": "Done! The bug is fixed.",
            "model": "composer-2.5",
            "conversation_id": "conv-abc",
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {}  # observe-only: empty response object

    # Latest row is the afterAgentResponse audit row.
    recent = store.recent()
    assert recent[0].wire_format == "cursor:afterAgentResponse"
    assert recent[0].source == "cursor"
    iid = recent[0].id
    assert iid is not None
    texts = store.texts(iid)
    assert len(texts) == 1
    assert texts[0].category == "assistant"
    assert texts[0].original == "Done! The bug is fixed."


def test_cursor_hook_after_agent_response_save_texts_anonymized_prompt_had_pii(
    tmp_path: Path,
) -> None:
    """Under save_texts="anonymized", the response is saved when the prompt had PII."""
    settings = Settings(_env_file=None, save_texts="anonymized", policy="monitor")
    client, _, store = build(tmp_path, lambda p: ForwardResult(200, {}, {}), settings)
    conv_id = "conv-pii-123"
    client.post(
        "/v1/cursor-hook",
        json={
            "hook_event_name": "beforeSubmitPrompt",
            "prompt": "ping John Smith",
            "model": "m",
            "conversation_id": conv_id,
        },
    )
    resp = client.post(
        "/v1/cursor-hook",
        json={
            "hook_event_name": "afterAgentResponse",
            "text": "Sure, I pinged them.",
            "model": "m",
            "conversation_id": conv_id,
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {}

    recent = store.recent()
    response_row = next(r for r in recent if r.wire_format == "cursor:afterAgentResponse")
    assert response_row.id is not None
    texts = store.texts(response_row.id)
    assert len(texts) == 1
    assert texts[0].category == "assistant"
    assert texts[0].original == "Sure, I pinged them."


def test_cursor_hook_after_agent_response_save_texts_anonymized_prompt_had_no_pii(
    tmp_path: Path,
) -> None:
    """Under save_texts="anonymized", the response is dropped when the prompt had no PII."""
    settings = Settings(_env_file=None, save_texts="anonymized", policy="monitor")
    client, _, store = build(tmp_path, lambda p: ForwardResult(200, {}, {}), settings)
    conv_id = "conv-clean-456"
    client.post(
        "/v1/cursor-hook",
        json={
            "hook_event_name": "beforeSubmitPrompt",
            "prompt": "just say hello",
            "model": "m",
            "conversation_id": conv_id,
        },
    )
    resp = client.post(
        "/v1/cursor-hook",
        json={
            "hook_event_name": "afterAgentResponse",
            "text": "Hello there!",
            "model": "m",
            "conversation_id": conv_id,
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {}

    recent = store.recent()
    response_row = next(r for r in recent if r.wire_format == "cursor:afterAgentResponse")
    assert response_row.id is not None
    assert store.texts(response_row.id) == []  # no text saved


def test_cursor_hook_after_agent_response_never_blocked(tmp_path: Path) -> None:
    """afterAgentResponse is never denied even when cursor_block=True and PII found."""
    settings = Settings(_env_file=None, policy="monitor", cursor_block=True)
    client, _, _ = build(tmp_path, lambda p: ForwardResult(200, {}, {}), settings)
    resp = client.post(
        "/v1/cursor-hook",
        json={
            "hook_event_name": "afterAgentResponse",
            "text": "The answer for John Smith is 42.",
            "model": "m",
            "conversation_id": "conv-block-test",
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {}  # empty object, NOT a deny


def test_cursor_hook_after_agent_response_no_conv_id_save_texts_all(tmp_path: Path) -> None:
    """Without a conversation_id, save_texts="all" still stores the response."""
    settings = Settings(_env_file=None, save_texts="all", policy="monitor")
    client, _, store = build(tmp_path, lambda p: ForwardResult(200, {}, {}), settings)
    resp = client.post(
        "/v1/cursor-hook",
        json={
            "hook_event_name": "afterAgentResponse",
            "text": "Here is the result.",
            "model": "m",
        },
    )
    assert resp.status_code == 200
    iid = store.recent()[0].id
    assert iid is not None
    texts = store.texts(iid)
    assert len(texts) == 1
    assert texts[0].category == "assistant"


def test_cursor_hook_after_agent_response_no_conv_id_save_texts_anonymized(
    tmp_path: Path,
) -> None:
    """Without a conversation_id under save_texts="anonymized", the response is not saved
    (cannot correlate with a preceding prompt, so falls back to safe-drop)."""
    settings = Settings(_env_file=None, save_texts="anonymized", policy="monitor")
    client, _, store = build(tmp_path, lambda p: ForwardResult(200, {}, {}), settings)
    resp = client.post(
        "/v1/cursor-hook",
        json={
            "hook_event_name": "afterAgentResponse",
            "text": "Here is the result.",
            "model": "m",
        },
    )
    assert resp.status_code == 200
    iid = store.recent()[0].id
    assert iid is not None
    assert store.texts(iid) == []  # no correlation possible → not saved


def test_openai_responses_function_call_arguments_not_saved(tmp_path: Path) -> None:
    """function_call arguments (LLM-authored) are anonymized but not saved."""
    settings = Settings(_env_file=None, save_texts="all", policy="pseudonymize")
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
