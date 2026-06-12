"""PII preview UI tests. No network, no model: stub detector + temp store."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from privacy_kit.core.types import Span
from privacy_kit.gateway.proxy import create_app
from privacy_kit.gateway.store import AuditStore


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


PII = {"John Smith": "PERSON_NAME", "john@x.com": "EMAIL_ADDRESS"}


def build(tmp_path: Path) -> tuple[TestClient, AuditStore]:
    store = AuditStore(tmp_path / "audit.sqlite")
    app = create_app(detector=LiteralDetector(PII), store=store)
    return TestClient(app), store


def test_ui_page_is_served_inline(tmp_path: Path) -> None:
    client, _ = build(tmp_path)
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "privacy-kit" in resp.text
    # A privacy tool must not load assets from the network.
    assert "http://" not in resp.text.replace("http://{", "")
    assert "https://" not in resp.text
    assert "<script src" not in resp.text


def test_preview_returns_spans_counts_and_anonymized(tmp_path: Path) -> None:
    client, _ = build(tmp_path)
    text = "Reach John Smith at john@x.com."
    resp = client.post("/ui/api/preview", json={"text": text})
    assert resp.status_code == 200
    data = resp.json()

    labels = {(s["label"]) for s in data["spans"]}
    assert labels == {"PERSON_NAME", "EMAIL_ADDRESS"}
    for span in data["spans"]:
        assert text[span["start"] : span["end"]] in PII

    assert data["anonymized"] == "Reach [PERSON_NAME_1] at [EMAIL_ADDRESS_1]."
    assert data["counts"] == {"PERSON_NAME": 1, "EMAIL_ADDRESS": 1}


def test_preview_is_never_audited_or_persisted(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    client.post("/ui/api/preview", json={"text": "John Smith john@x.com"})
    assert store.summary()["interactions"] == 0
    # And the raw text is nowhere in the DB file.
    blob = (tmp_path / "audit.sqlite").read_bytes()
    assert b"John Smith" not in blob


def test_preview_rejects_bad_payloads(tmp_path: Path) -> None:
    client, _ = build(tmp_path)
    assert client.post("/ui/api/preview", json={"text": 42}).status_code == 400
    assert client.post("/ui/api/preview", json=["nope"]).status_code == 400
    assert (
        client.post(
            "/ui/api/preview", content=b"not json", headers={"content-type": "application/json"}
        ).status_code
        == 400
    )
    assert client.post("/ui/api/preview", json={"text": "x" * 50_001}).status_code == 413


def test_summary_and_recent_reflect_the_audit_store(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    store.record(
        source="claude-code",
        wire_format="anthropic",
        model="claude-opus-4-8",
        entity_counts={"PERSON_NAME": 2},
        input_tokens=5,
        output_tokens=7,
    )

    summary = client.get("/ui/api/summary").json()
    assert summary["interactions"] == 1
    assert summary["entities_by_type"] == {"PERSON_NAME": 2}

    recent = client.get("/ui/api/recent").json()["interactions"]
    assert len(recent) == 1
    assert recent[0]["source"] == "claude-code"
    assert recent[0]["entity_counts"] == {"PERSON_NAME": 2}
    assert recent[0]["input_tokens"] == 5


def test_texts_endpoint_returns_segments(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    store.record(
        source="claude-code",
        wire_format="anthropic",
        model="claude-opus-4-8",
        entity_counts={"PERSON_NAME": 1},
        texts=[("hi John Smith", "hi [PERSON_NAME_1]")],
    )

    resp = client.get("/ui/api/texts")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["texts"]) == 1
    assert data["texts"][0]["original"] == "hi John Smith"
    assert data["texts"][0]["anonymized"] == "hi [PERSON_NAME_1]"


def test_texts_endpoint_empty(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    store.record(
        source="claude-code",
        wire_format="anthropic",
        model="claude-opus-4-8",
        entity_counts={"PERSON_NAME": 1},
    )

    resp = client.get("/ui/api/texts")
    assert resp.status_code == 200
    data = resp.json()
    assert data["texts"] == []
