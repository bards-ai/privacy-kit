"""OTLP log sink tests. No network, no model: stub detector + temp store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from privacy_kit.core.types import Span
from privacy_kit.gateway.otel import scrub_otlp
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


def otlp_logs_payload(prompt: str) -> dict[str, Any]:
    """A minimal but realistic OTLP/JSON logs payload (Claude Code shape)."""
    return {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": "claude-code"}}]
                },
                "scopeLogs": [
                    {
                        "logRecords": [
                            {
                                "timeUnixNano": "1700000000000000000",
                                "body": {"stringValue": prompt},
                                "attributes": [
                                    {
                                        "key": "event.name",
                                        "value": {"stringValue": "user_prompt"},
                                    },
                                    {"key": "prompt", "value": {"stringValue": prompt}},
                                ],
                            }
                        ]
                    }
                ],
            }
        ]
    }


PII = {"John Smith": "PERSON_NAME", "john@x.com": "EMAIL_ADDRESS"}


def test_scrub_otlp_replaces_all_string_values_and_counts() -> None:
    payload = otlp_logs_payload("Hi, I'm John Smith (john@x.com)")
    counts = scrub_otlp(payload, LiteralDetector(PII))

    record = payload["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
    body = record["body"]["stringValue"]
    prompt_attr = record["attributes"][1]["value"]["stringValue"]

    assert "John Smith" not in body and "john@x.com" not in body
    assert "[PERSON_NAME_1]" in body and "[EMAIL_ADDRESS_1]" in body
    assert prompt_attr == body  # same vault -> same placeholders
    # Non-PII resource attribute is untouched.
    resource_attr = payload["resourceLogs"][0]["resource"]["attributes"][0]["value"]["stringValue"]
    assert resource_attr == "claude-code"
    assert counts == {"PERSON_NAME": 1, "EMAIL_ADDRESS": 1}


def test_logs_endpoint_scrubs_and_audits(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite")
    app = create_app(detector=LiteralDetector(PII), store=store)
    client = TestClient(app)

    resp = client.post(
        "/v1/logs",
        json=otlp_logs_payload("contact John Smith"),
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 200
    assert resp.json() == {}  # OTLP success response

    summary = store.summary()
    assert summary["interactions"] == 1
    assert summary["entities_by_type"] == {"PERSON_NAME": 1}
    assert store.recent()[0].source == "otel"
    assert store.recent()[0].wire_format == "otel"


def test_logs_endpoint_rejects_protobuf(tmp_path: Path) -> None:
    app = create_app(detector=LiteralDetector(PII), store=AuditStore(tmp_path / "a.sqlite"))
    client = TestClient(app)
    resp = client.post(
        "/v1/logs",
        content=b"\x00\x01",
        headers={"content-type": "application/x-protobuf"},
    )
    assert resp.status_code == 415


def test_no_pii_logs_record_nothing(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite")
    app = create_app(detector=LiteralDetector(PII), store=store)
    client = TestClient(app)
    resp = client.post(
        "/v1/logs",
        json=otlp_logs_payload("just a normal log line"),
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 200
    assert store.summary()["interactions"] == 0


def test_traces_endpoint_scrubs_but_does_not_audit(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite")
    app = create_app(detector=LiteralDetector(PII), store=store)
    client = TestClient(app)
    resp = client.post(
        "/v1/traces",
        json=otlp_logs_payload("trace from John Smith"),
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 200
    assert store.summary()["interactions"] == 0  # traces aren't audited
