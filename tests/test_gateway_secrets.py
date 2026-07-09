"""Gateway secret-detection e2e: the M1 acceptance story, model-free.

A tool_result carrying a fake ``.env`` file (the exact thing Claude Code sends
when it reads one) goes through the proxy with the REAL deterministic detectors
— no NER model, no download — and the secrets must be caught in both policies:
audited in ``monitor``, replaced before forwarding in ``pseudonymize``.
All secret values are fabricated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from privacy_kit.core.detectors import CompositeDetector
from privacy_kit.core.detectors_regex import ChecksumPiiDetector
from privacy_kit.core.detectors_secret import SecretDetector
from privacy_kit.gateway.config import Settings
from privacy_kit.gateway.proxy import ForwardResult, create_app
from privacy_kit.gateway.store import AuditStore


class CapturingForwarder:
    def __init__(self) -> None:
        self.last_payload: dict[str, Any] | None = None

    async def __call__(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> ForwardResult:
        self.last_payload = payload
        return ForwardResult(200, {"content": [{"type": "text", "text": "done"}]}, {})


AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
DB_PASSWORD = "S3cr3t.Pw!x9"
CARD = "4111 1111 1111 1111"

ENV_FILE = (
    "# .env\n"
    f"AWS_ACCESS_KEY_ID={AWS_KEY}\n"
    f"DATABASE_URL=postgresql://svc:{DB_PASSWORD}@db.internal:5432/app\n"
    f"TEST_CARD={CARD}\n"
)


def build(tmp_path: Path, policy: str) -> tuple[TestClient, CapturingForwarder, AuditStore]:
    detector = CompositeDetector([SecretDetector(), ChecksumPiiDetector()])
    store = AuditStore(tmp_path / "audit.sqlite")
    forwarder = CapturingForwarder()
    app = create_app(
        detector=detector,
        store=store,
        forwarder=forwarder,
        settings=Settings(_env_file=None, policy=policy, save_texts="all"),
    )
    return TestClient(app), forwarder, store


def request_body() -> dict[str, Any]:
    return {
        "model": "claude-opus-4-8",
        "messages": [
            {"role": "user", "content": "read my env file"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": [{"type": "text", "text": ENV_FILE}],
                    }
                ],
            },
        ],
    }


def test_pseudonymize_replaces_secrets_before_forwarding(tmp_path: Path) -> None:
    client, forwarder, _store = build(tmp_path, "pseudonymize")
    assert client.post("/v1/messages", json=request_body()).status_code == 200

    sent = str(forwarder.last_payload)
    for secret in (AWS_KEY, DB_PASSWORD, CARD):
        assert secret not in sent
    assert "[SECRET_AWS_ACCESS_KEY_1]" in sent
    assert "[SECRET_CONNECTION_STRING_1]" in sent
    assert "[PAYMENT_CARD_1]" in sent
    # Non-secret structure survives so the LLM can still reason about the file.
    assert "AWS_ACCESS_KEY_ID=" in sent
    assert "db.internal:5432/app" in sent


def test_monitor_forwards_original_but_audits_secrets(tmp_path: Path) -> None:
    client, forwarder, store = build(tmp_path, "monitor")
    assert client.post("/v1/messages", json=request_body()).status_code == 200

    # Monitor: the original goes upstream untouched...
    sent = str(forwarder.last_payload)
    for secret in (AWS_KEY, DB_PASSWORD, CARD):
        assert secret in sent

    # ...but every secret is on the audit record.
    row = store.recent()[0]
    assert row.policy == "monitor"
    assert row.entity_counts.get("SECRET_AWS_ACCESS_KEY") == 1
    assert row.entity_counts.get("SECRET_CONNECTION_STRING") == 1
    assert row.entity_counts.get("PAYMENT_CARD") == 1
    assert row.id is not None
    anonymized = " ".join(t.anonymized for t in store.texts(row.id))
    assert "[SECRET_AWS_ACCESS_KEY_1]" in anonymized
