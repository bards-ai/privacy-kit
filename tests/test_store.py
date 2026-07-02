"""Audit store tests. Model-free; uses a temp SQLite file per test."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("sqlmodel")

from privacy_kit.core.types import Span
from privacy_kit.core.vault import anonymize
from privacy_kit.gateway.store import AuditStore, Interaction


class StubDetector:
    def __init__(self, spans: list[Span]) -> None:
        self._spans = spans

    def detect(self, text: str) -> list[Span]:
        return list(self._spans)


def make_store(tmp_path: Path) -> AuditStore:
    return AuditStore(tmp_path / "audit.sqlite")


def test_record_and_summarize(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.record(
        source="claude-code",
        wire_format="anthropic",
        model="claude-opus-4-8",
        entity_counts={"PERSON_NAME": 2, "EMAIL_ADDRESS": 1},
        input_tokens=42,
    )
    store.record(
        source="codex",
        wire_format="openai_responses",
        model="gpt-5",
        entity_counts={"PERSON_NAME": 1},
    )
    summary = store.summary()
    assert summary["interactions"] == 2
    assert summary["entities_total"] == 4
    assert summary["entities_by_type"] == {"PERSON_NAME": 3, "EMAIL_ADDRESS": 1}


def test_entity_total_and_counts_persisted(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    iid = store.record(
        source="cursor",
        wire_format="openai_chat",
        model="gpt-5",
        entity_counts={"PHONE_NUMBER": 1, "PERSON_NAME": 1},
    )
    assert iid == 1
    rows = store.recent()
    assert len(rows) == 1
    assert rows[0].entity_total == 2
    assert rows[0].entity_counts == {"PHONE_NUMBER": 1, "PERSON_NAME": 1}


def test_zero_counts_are_dropped(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.record(
        source="otel",
        wire_format="otel",
        model="n/a",
        entity_counts={"PERSON_NAME": 0, "EMAIL_ADDRESS": 2},
    )
    assert store.summary()["entities_by_type"] == {"EMAIL_ADDRESS": 2}


def test_record_texts_persisted(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    iid = store.record(
        source="claude-code",
        wire_format="anthropic",
        model="claude-opus-4-8",
        entity_counts={"PERSON_NAME": 1},
        texts=[("I'm Jan Nowak", "I'm [PERSON_NAME_1]"), ("plain", "plain")],
    )
    rows = store.texts(iid)
    assert [(r.seq, r.original, r.anonymized) for r in rows] == [
        (0, "I'm Jan Nowak", "I'm [PERSON_NAME_1]"),
        (1, "plain", "plain"),
    ]
    assert all(r.interaction_id == iid for r in rows)


def test_record_without_texts_stores_none(tmp_path: Path) -> None:
    """Metadata-only callers (the OTel sink) keep the texts table empty."""
    store = make_store(tmp_path)
    iid = store.record(
        source="otel",
        wire_format="otel",
        model="logs",
        entity_counts={"PERSON_NAME": 1},
    )
    assert store.texts(iid) == []


def test_no_raw_pii_anywhere_in_the_database(tmp_path: Path) -> None:
    """Metadata-only recording (e.g. the OTel sink) stores no raw PII."""
    text = "I'm Anna Kowalska, email anna.k@example.com."
    spans = [
        Span(text.index("Anna Kowalska"), text.index("Anna Kowalska") + 13, "PERSON_NAME", 0.99),
        Span(
            text.index("anna.k@example.com"),
            text.index("anna.k@example.com") + 18,
            "EMAIL_ADDRESS",
            0.99,
        ),
    ]
    _clean, vault = anonymize(text, StubDetector(spans))

    db = tmp_path / "audit.sqlite"
    store = AuditStore(db)
    store.record(
        source="claude-code",
        wire_format="anthropic",
        model="claude-opus-4-8",
        entity_counts=vault.type_counts,
    )

    # Dump every text value stored in every table and assert no raw PII leaked.
    secrets = ["Anna", "Kowalska", "anna.k@example.com", "example.com"]
    conn = sqlite3.connect(db)
    haystack: list[str] = []
    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        for row in conn.execute(f"SELECT * FROM {table}").fetchall():
            haystack.extend(str(cell) for cell in row)
    conn.close()
    blob = "\n".join(haystack)
    for secret in secrets:
        assert secret not in blob, f"raw PII {secret!r} leaked into the DB"
    # Sanity: the metadata we *do* want is present.
    assert "PERSON_NAME" in blob
    assert "EMAIL_ADDRESS" in blob


def test_raw_pii_lands_only_in_the_texts_table(tmp_path: Path) -> None:
    """With texts saved, raw values live in interactiontext and nowhere else."""
    text = "I'm Anna Kowalska, email anna.k@example.com."
    spans = [
        Span(text.index("Anna Kowalska"), text.index("Anna Kowalska") + 13, "PERSON_NAME", 0.99),
        Span(
            text.index("anna.k@example.com"),
            text.index("anna.k@example.com") + 18,
            "EMAIL_ADDRESS",
            0.99,
        ),
    ]
    clean, vault = anonymize(text, StubDetector(spans))

    db = tmp_path / "audit.sqlite"
    store = AuditStore(db)
    store.record(
        source="claude-code",
        wire_format="anthropic",
        model="claude-opus-4-8",
        entity_counts=vault.type_counts,
        texts=[(text, clean)],
    )

    secrets = ["Anna Kowalska", "anna.k@example.com"]
    conn = sqlite3.connect(db)
    tables = [
        name
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]
    assert "interactiontext" in tables
    for table in tables:
        blob = "\n".join(
            str(cell) for row in conn.execute(f"SELECT * FROM {table}").fetchall() for cell in row
        )
        for secret in secrets:
            if table == "interactiontext":
                assert secret in blob
            else:
                assert secret not in blob, f"raw PII {secret!r} leaked into table {table!r}"
    conn.close()


def test_recent_orders_newest_first(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    for i in range(3):
        store.record(
            source=f"tool{i}",
            wire_format="anthropic",
            model="m",
            entity_counts={"PERSON_NAME": 1},
        )
    rows: list[Interaction] = store.recent(limit=2)
    assert len(rows) == 2
    assert rows[0].source == "tool2"


# --- turn_had_pii tests -------------------------------------------------------


def test_turn_had_pii_returns_true_when_prompt_detected_pii(tmp_path: Path) -> None:
    """Returns True if the most recent exchange (since the last assistant row) had PII."""
    store = make_store(tmp_path)
    conv_id = "conv-1"
    # Prompt interaction with PII.
    store.record(
        source="cursor",
        wire_format="cursor:beforeSubmitPrompt",
        model="m",
        entity_counts={"PERSON_NAME": 1},
        conversation_id=conv_id,
        texts=[("John Smith, fix this", "PERSON_NAME_1, fix this", "user")],
    )
    assert store.turn_had_pii(conv_id) is True


def test_turn_had_pii_returns_false_when_prompt_had_no_pii(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    conv_id = "conv-2"
    store.record(
        source="cursor",
        wire_format="cursor:beforeSubmitPrompt",
        model="m",
        entity_counts={},
        conversation_id=conv_id,
    )
    assert store.turn_had_pii(conv_id) is False


def test_turn_had_pii_returns_false_for_unknown_conversation(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.turn_had_pii("no-such-conv") is False


def test_turn_had_pii_stops_at_previous_assistant_row(tmp_path: Path) -> None:
    """The look-back stops at the last interaction that already has an assistant text row.

    Previous exchange's PII must NOT bleed into the current exchange's gate.
    """
    store = make_store(tmp_path)
    conv_id = "conv-3"
    # Exchange 1: prompt had PII, then response was recorded (with assistant text).
    store.record(
        source="cursor",
        wire_format="cursor:beforeSubmitPrompt",
        model="m",
        entity_counts={"PERSON_NAME": 1},
        conversation_id=conv_id,
        texts=[("John Smith", "PERSON_NAME_1", "user")],
    )
    # Simulate the assistant response row that terminates exchange 1.
    store.record(
        source="cursor",
        wire_format="cursor:afterAgentResponse",
        model="m",
        entity_counts={},
        conversation_id=conv_id,
        texts=[("Done.", "Done.", "assistant")],
    )
    # Exchange 2: clean prompt, no PII.
    store.record(
        source="cursor",
        wire_format="cursor:beforeSubmitPrompt",
        model="m",
        entity_counts={},
        conversation_id=conv_id,
    )
    # Exchange 2's prompt had no PII, so turn_had_pii for the *current* exchange must be False.
    assert store.turn_had_pii(conv_id) is False


def test_turn_had_pii_finds_pii_in_second_exchange(tmp_path: Path) -> None:
    """Multiple exchanges: correctly detects PII in the second exchange."""
    store = make_store(tmp_path)
    conv_id = "conv-4"
    # Exchange 1: clean prompt, response.
    store.record(
        source="cursor",
        wire_format="cursor:beforeSubmitPrompt",
        model="m",
        entity_counts={},
        conversation_id=conv_id,
    )
    store.record(
        source="cursor",
        wire_format="cursor:afterAgentResponse",
        model="m",
        entity_counts={},
        conversation_id=conv_id,
        texts=[("OK.", "OK.", "assistant")],
    )
    # Exchange 2: PII prompt.
    store.record(
        source="cursor",
        wire_format="cursor:beforeSubmitPrompt",
        model="m",
        entity_counts={"EMAIL_ADDRESS": 1},
        conversation_id=conv_id,
        texts=[("john@x.com", "[EMAIL_ADDRESS_1]", "user")],
    )
    assert store.turn_had_pii(conv_id) is True
