"""Codex via /v1/responses: ChatGPT-subscription vs API-key routing.

Codex sends the model call to /v1/responses in both auth modes. A subscription
request carries `chatgpt-account-id` and must be forwarded to the ChatGPT
backend; an API-key request keeps the api.openai.com upstream. Same Responses
wire format either way. No network, no model: stub detector + fake forwarder,
mirroring test_proxy.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from privacy_kit.core.types import Span
from privacy_kit.gateway.config import Settings
from privacy_kit.gateway.proxy import ForwardResult, create_app
from privacy_kit.gateway.store import AuditStore

SECRET = "topsecret-name"
PLACEHOLDER = "[PERSON_NAME_1]"


class OneLiteralDetector:
    """Tags every occurrence of SECRET as PERSON_NAME."""

    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        start = 0
        while (i := text.find(SECRET, start)) >= 0:
            spans.append(Span(i, i + len(SECRET), "PERSON_NAME", 0.99))
            start = i + len(SECRET)
        return spans


class CapturingForwarder:
    """Records the upstream call and answers with a canned Responses body."""

    def __init__(self, result: ForwardResult) -> None:
        self._result = result
        self.last_url: str | None = None
        self.last_headers: dict[str, str] | None = None
        self.last_payload: dict[str, Any] | None = None

    async def __call__(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> ForwardResult:
        self.last_url = url
        self.last_headers = headers
        self.last_payload = payload
        return self._result


def _build(tmp_path: Path) -> tuple[TestClient, CapturingForwarder, AuditStore]:
    upstream_body = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": f"Hello {PLACEHOLDER}!"}],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    forwarder = CapturingForwarder(ForwardResult(200, upstream_body, {}))
    store = AuditStore(tmp_path / "audit.sqlite")
    app = create_app(
        detector=OneLiteralDetector(),
        store=store,
        forwarder=forwarder,
        settings=Settings(_env_file=None, policy="pseudonymize"),
    )
    return TestClient(app), forwarder, store


_REQUEST = {
    "model": "gpt-5",
    "instructions": f"You help {SECRET}.",
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": f"My name is {SECRET}."}],
        }
    ],
}


def test_subscription_request_routes_to_chatgpt_backend(tmp_path: Path) -> None:
    client, forwarder, store = _build(tmp_path)

    response = client.post(
        "/v1/responses",
        json=_REQUEST,
        headers={"authorization": "Bearer chatgpt-oauth-token", "chatgpt-account-id": "acc-1"},
    )

    # chatgpt-account-id present -> ChatGPT backend, path rewritten to /codex/responses.
    assert forwarder.last_url == "https://chatgpt.com/backend-api/codex/responses"

    # The login token and account header pass through untouched.
    assert forwarder.last_headers is not None
    assert forwarder.last_headers["authorization"] == "Bearer chatgpt-oauth-token"
    assert forwarder.last_headers["chatgpt-account-id"] == "acc-1"

    # The secret never left: pseudonymized in instructions and input alike.
    assert forwarder.last_payload is not None
    sent = str(forwarder.last_payload)
    assert SECRET not in sent
    assert PLACEHOLDER in sent

    # The response is rehydrated for the client.
    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["text"] == f"Hello {SECRET}!"

    # Audited as codex traffic on the Responses wire format, metadata only.
    rows = store.recent(limit=1)
    assert len(rows) == 1
    assert rows[0].source == "codex"
    assert rows[0].wire_format == "openai_responses"
    assert rows[0].entity_total == 1  # one distinct value, seen in instructions + input


def test_api_key_request_keeps_openai_upstream(tmp_path: Path) -> None:
    client, forwarder, _ = _build(tmp_path)

    response = client.post(
        "/v1/responses",
        json=_REQUEST,
        headers={"authorization": "Bearer sk-realapikey"},  # no chatgpt-account-id
    )

    assert response.status_code == 200
    # No account header -> default OpenAI upstream, original path unchanged.
    assert forwarder.last_url == "https://api.openai.com/v1/responses"
    assert forwarder.last_payload is not None
    assert SECRET not in str(forwarder.last_payload)


def test_zstd_streaming_frame_body_is_decoded(tmp_path: Path) -> None:
    """Codex zstd-compresses bodies as streaming frames with no content-size header."""
    zstandard = pytest.importorskip("zstandard")
    client, forwarder, _ = _build(tmp_path)

    payload = {"model": "gpt-5", "input": f"My name is {SECRET}."}
    compressor = zstandard.ZstdCompressor().compressobj()  # streaming: no content size
    raw = compressor.compress(json.dumps(payload).encode()) + compressor.flush()

    response = client.post(
        "/v1/responses",
        content=raw,
        headers={
            "content-type": "application/json",
            "content-encoding": "zstd",
            "chatgpt-account-id": "acc-1",
        },
    )

    assert response.status_code == 200
    assert forwarder.last_payload is not None
    assert forwarder.last_payload["input"] == f"My name is {PLACEHOLDER}."
