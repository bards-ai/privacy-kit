"""Conversation grouping: key derivation, store queries, proxy wiring, web API."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("sqlmodel")

from privacy_kit.gateway.proxy.transform import conversation_key
from privacy_kit.gateway.store import AuditStore

# --- conversation_key -------------------------------------------------------


def test_conversation_key_stable_across_turns_openai_chat() -> None:
    """Same opening user message → same key even as history grows each turn."""
    turn1 = {"messages": [{"role": "user", "content": "Hello there"}]}
    turn2 = {
        "messages": [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "A follow-up"},
        ]
    }
    k1 = conversation_key("openai_chat", turn1)
    k2 = conversation_key("openai_chat", turn2)
    assert k1 is not None
    assert k1 == k2


def test_conversation_key_differs_for_different_openings() -> None:
    a = conversation_key("openai_chat", {"messages": [{"role": "user", "content": "topic A"}]})
    b = conversation_key("openai_chat", {"messages": [{"role": "user", "content": "topic B"}]})
    assert a != b


def test_conversation_key_reads_first_text_block() -> None:
    """Content-as-block-list is supported (first text block wins)."""
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "block hello"}]},
        ]
    }
    assert conversation_key("openai_chat", body) == conversation_key(
        "openai_chat", {"messages": [{"role": "user", "content": "block hello"}]}
    )


def _claude_code_body(session_id: str, text: str) -> dict:
    """A Messages API body shaped like real Claude Code traffic: the session id
    rides in metadata.user_id as a JSON-encoded string."""
    return {
        "metadata": {
            "user_id": json.dumps(
                {"device_id": "d" * 8, "account_uuid": "a" * 8, "session_id": session_id}
            )
        },
        "messages": [{"role": "user", "content": text}],
    }


def test_conversation_key_anthropic_prefers_metadata_session_id() -> None:
    """Claude Code sends its transcript session id in metadata.user_id (verified
    against real traffic); it must win over the first-message-hash fallback and
    stay stable as history grows."""
    k1 = conversation_key("anthropic", _claude_code_body("sess-1", "hello"))
    k2 = conversation_key("anthropic", _claude_code_body("sess-1", "a different opening"))
    assert k1 == "sess-1"
    assert k1 == k2


def test_conversation_key_anthropic_new_session_id_after_clear() -> None:
    """/clear starts a new session id → new conversation, even when the visible
    opening text (injected reminders, re-typed prompt) is identical."""
    before = conversation_key("anthropic", _claude_code_body("sess-1", "hi"))
    after = conversation_key("anthropic", _claude_code_body("sess-2", "hi"))
    assert before != after


def test_conversation_key_anthropic_legacy_session_id_format() -> None:
    """Older Claude Code sent metadata.user_id as a plain string ending in
    _session_<uuid>; the uuid is still extracted."""
    body = {
        "metadata": {
            "user_id": "user_ab12_account_cd34_session_123e4567-e89b-12d3-a456-426614174000"
        },
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert conversation_key("anthropic", body) == "123e4567-e89b-12d3-a456-426614174000"


def test_conversation_key_anthropic_none_without_session_id() -> None:
    """The anthropic wire never falls back to first-message hashing: a shared
    opening prompt (or /clear's near-constant injected preamble) would merge
    unrelated conversations, and ungrouped beats wrongly grouped. A stable
    per-user metadata.user_id (the documented API meaning) must not be used
    either — it would merge all of a user's conversations."""
    bare = {"messages": [{"role": "user", "content": "hi"}]}
    plain = {"metadata": {"user_id": "customer-42"}, **bare}
    no_session = {"metadata": {"user_id": json.dumps({"device_id": "d1"})}, **bare}
    assert conversation_key("anthropic", bare) is None
    assert conversation_key("anthropic", plain) is None
    assert conversation_key("anthropic", no_session) is None


def test_conversation_key_fallback_skips_injected_blocks() -> None:
    """The OpenAI hash fallback must key on the first *typed* text: a session
    that opens with harness-injected wrappers (near-constant per project)
    would otherwise collapse every such session into one conversation."""
    def injected_body(prompt: str) -> dict:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": "<user_instructions>project instructions</user_instructions>",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<environment_context>cwd, sandbox</environment_context>"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
        }

    typed_only = {"messages": [{"role": "user", "content": "fix the bug"}]}
    assert conversation_key("openai_chat", injected_body("fix the bug")) == conversation_key(
        "openai_chat", typed_only
    )
    assert conversation_key("openai_chat", injected_body("topic A")) != conversation_key(
        "openai_chat", injected_body("topic B")
    )


def test_conversation_key_openai_chat_and_responses() -> None:
    chat = {"messages": [{"role": "user", "content": "hi"}]}
    resp_str = {"input": "hi"}
    resp_list = {"input": [{"role": "user", "content": "hi"}]}
    # Each format hashes the same opening text to the same key.
    assert conversation_key("openai_chat", chat) is not None
    assert conversation_key("openai_responses", resp_str) == conversation_key("openai_chat", chat)
    assert conversation_key("openai_responses", resp_list) == conversation_key(
        "openai_responses", resp_str
    )


def test_conversation_key_none_when_no_user_message() -> None:
    assert conversation_key("anthropic", {"messages": []}) is None
    assert conversation_key("anthropic", {}) is None
    assert conversation_key("openai_responses", {"input": []}) is None


def test_conversation_key_cursor_uses_explicit_id() -> None:
    assert conversation_key("cursor", {"conversation_id": "abc-123"}) == "abc-123"
    assert conversation_key("cursor", {}) is None


def test_conversation_key_openai_responses_prefers_prompt_cache_key() -> None:
    """Codex sends a stable prompt_cache_key per conversation (verified against
    real Codex CLI traffic); it must win over the first-message-hash fallback
    and stay stable as history grows."""
    turn1 = {"input": [{"role": "user", "content": "hi"}], "prompt_cache_key": "sess-a"}
    turn2 = {
        "input": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "a completely different follow-up"},
        ],
        "prompt_cache_key": "sess-a",
    }
    assert conversation_key("openai_responses", turn1) == "sess-a"
    assert conversation_key("openai_responses", turn1) == conversation_key(
        "openai_responses", turn2
    )


def test_conversation_key_openai_responses_new_key_after_clear() -> None:
    """A fresh prompt_cache_key (e.g. after Codex's /clear) must yield a
    different conversation id, even for an identical opening message."""
    body_a = {"input": [{"role": "user", "content": "hi"}], "prompt_cache_key": "sess-a"}
    body_b = {"input": [{"role": "user", "content": "hi"}], "prompt_cache_key": "sess-b"}
    before = conversation_key("openai_responses", body_a)
    after = conversation_key("openai_responses", body_b)
    assert before != after


def test_conversation_key_openai_responses_falls_back_to_client_metadata() -> None:
    """When prompt_cache_key is absent, use Codex's client_metadata thread/session
    id — never the per-turn turn_id, which changes every request."""
    body = {
        "input": [{"role": "user", "content": "hi"}],
        "client_metadata": {"thread_id": "thr-1", "turn_id": "differs-every-turn"},
    }
    assert conversation_key("openai_responses", body) == "thr-1"


def test_conversation_key_openai_responses_falls_back_to_hash_without_ids() -> None:
    """With neither field present, fall back to the first-message hash."""
    body = {"input": [{"role": "user", "content": "hi"}]}
    chat_equivalent = {"messages": [{"role": "user", "content": "hi"}]}
    assert conversation_key("openai_responses", body) == conversation_key(
        "openai_chat", chat_equivalent
    )


# --- store: list_conversations / get_conversation ---------------------------


def _backdate(store: AuditStore, iid: int, when: datetime) -> None:
    """Force a known created_at so ordering assertions are deterministic."""
    from sqlmodel import Session

    from privacy_kit.gateway.store.models import Interaction

    with Session(store.engine) as session:
        row = session.get(Interaction, iid)
        assert row is not None
        row.created_at = when
        session.add(row)
        session.commit()


def test_list_and_get_conversation(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # Conversation "conv-a": two main turns + one background (safety) turn.
    a1 = store.record(
        source="claude-code",
        wire_format="anthropic",
        model="claude-opus-4-8",
        entity_counts={"PERSON_NAME": 2},
        conversation_id="conv-a",
    )
    a2 = store.record(
        source="claude-code",
        wire_format="anthropic",
        model="claude-opus-4-8",
        entity_counts={"EMAIL_ADDRESS": 1},
        conversation_id="conv-a",
    )
    a3 = store.record(
        source="claude-code",
        wire_format="anthropic",
        model="claude-haiku-4-5",
        entity_counts={"PERSON_NAME": 1},
        kind="safety",
        conversation_id="conv-a",
    )
    # A different conversation, and an ungrouped (conversation_id=None) row.
    b1 = store.record(
        source="codex",
        wire_format="openai_responses",
        model="gpt-5",
        entity_counts={"PHONE_NUMBER": 1},
        conversation_id="conv-b",
    )
    store.record(
        source="cursor",
        wire_format="openai_chat",
        model="gpt-5",
        entity_counts={"PERSON_NAME": 1},
    )

    _backdate(store, a1, base)
    _backdate(store, a2, base + timedelta(minutes=5))
    _backdate(store, a3, base + timedelta(minutes=2))
    _backdate(store, b1, base + timedelta(hours=1))

    rows, total = store.list_conversations()
    # Only the two grouped conversations; the None-id row is excluded.
    assert total == 2
    by_id = {r["conversation_id"]: r for r in rows}
    assert set(by_id) == {"conv-a", "conv-b"}

    a = by_id["conv-a"]
    assert a["turn_count"] == 3
    assert a["entity_total"] == 4  # 2 + 1 + 1
    assert a["entity_counts"] == {"PERSON_NAME": 3, "EMAIL_ADDRESS": 1}
    assert a["sources"] == ["claude-code"]
    assert a["models"] == ["claude-haiku-4-5", "claude-opus-4-8"]
    assert a["background_count"] == 1
    # SQLite round-trips created_at as naive UTC, so compare on the wall clock.
    assert a["first_seen"].startswith("2026-01-01T00:00:00")
    assert a["last_seen"].startswith("2026-01-01T00:05:00")

    # Default sort is last_seen desc → conv-b (an hour later) comes first.
    assert rows[0]["conversation_id"] == "conv-b"

    # get_conversation returns the turns oldest-first (a1, a3, a2 by created_at).
    turns = store.get_conversation("conv-a")
    assert turns is not None
    assert [t.id for t in turns] == [a1, a3, a2]
    assert store.get_conversation("does-not-exist") is None


def test_list_conversations_min_entities_filter(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite")
    store.record(
        source="s",
        wire_format="anthropic",
        model="m",
        entity_counts={"PERSON_NAME": 5},
        conversation_id="rich",
    )
    store.record(
        source="s",
        wire_format="anthropic",
        model="m",
        entity_counts={},
        conversation_id="empty",
    )
    rows, total = store.list_conversations(min_entities=1)
    assert total == 1
    assert rows[0]["conversation_id"] == "rich"


def test_list_conversations_source_filter_includes_imported(tmp_path: Path) -> None:
    """Filtering by a live source surfaces both live and imported conversations;
    filtering by the "-import" value stays exact."""
    store = AuditStore(tmp_path / "audit.sqlite")
    store.record(
        source="claude-code",
        wire_format="anthropic",
        model="m",
        entity_counts={"PERSON_NAME": 1},
        conversation_id="live",
    )
    store.record(
        source="claude-code-import",
        wire_format="anthropic",
        model="m",
        policy="imported",
        entity_counts={"PERSON_NAME": 1},
        conversation_id="imported",
    )
    store.record(
        source="codex",
        wire_format="openai_responses",
        model="m",
        entity_counts={"PERSON_NAME": 1},
        conversation_id="other",
    )

    # Base source covers its imported rows too.
    _rows, total = store.list_conversations(sources=["claude-code"])
    assert total == 2
    ids = {r["conversation_id"] for r in _rows}
    assert ids == {"live", "imported"}

    # The "-import" value still selects only the imported conversation.
    rows, total = store.list_conversations(sources=["claude-code-import"])
    assert total == 1
    assert rows[0]["conversation_id"] == "imported"
