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


def test_no_raw_pii_anywhere_in_the_database(tmp_path: Path) -> None:
    """The core invariant: real PII values must never reach the DB."""
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
