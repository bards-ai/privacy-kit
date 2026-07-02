"""Log-safety audit for the gateway.

The hard invariant: raw PII must never reach logs, stdout, or stderr on the
proxy path. These tests drive PII-carrying requests through the app (happy
path, streaming, and a crashing forwarder) while capturing every logging
record and the captured std streams, then assert the raw values are absent.

``privacy-kit scan`` is the one intentional exception — the local user asked
to see their own text — and is excluded by design (see ruff T20 note in
pyproject).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from privacy_kit.core.types import Span
from privacy_kit.gateway.proxy import ForwardResult, create_app
from privacy_kit.gateway.store import AuditStore

RAW_PII = ("Anna Kowalska", "anna.kowalska@example.com")


class LiteralDetector:
    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for literal, etype in (
            ("Anna Kowalska", "PERSON_NAME"),
            ("anna.kowalska@example.com", "EMAIL_ADDRESS"),
        ):
            start = 0
            while (i := text.find(literal, start)) >= 0:
                spans.append(Span(i, i + len(literal), etype, 0.99))
                start = i + len(literal)
        return spans


def pii_request(client: TestClient) -> Any:
    return client.post(
        "/v1/messages",
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "I'm Anna Kowalska <anna.kowalska@example.com>"}
            ],
        },
    )


def assert_no_pii(*blobs: str) -> None:
    for blob in blobs:
        for raw in RAW_PII:
            assert raw not in blob


def test_proxy_round_trip_leaks_no_pii_to_logs_or_streams(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    async def ok_forwarder(
        url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> ForwardResult:
        return ForwardResult(200, {"content": [{"type": "text", "text": "hi"}]}, {})

    app = create_app(
        detector=LiteralDetector(), store=AuditStore(tmp_path / "a.sqlite"), forwarder=ok_forwarder
    )
    with caplog.at_level(logging.DEBUG):
        resp = pii_request(TestClient(app))
    assert resp.status_code == 200
    out = capsys.readouterr()
    assert_no_pii(caplog.text, out.out, out.err)


def test_crashing_forwarder_leaks_no_pii_to_logs_or_streams(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    async def boom(url: str, headers: dict[str, str], payload: dict[str, Any]) -> ForwardResult:
        raise RuntimeError("upstream exploded")

    app = create_app(
        detector=LiteralDetector(), store=AuditStore(tmp_path / "a.sqlite"), forwarder=boom
    )
    client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.DEBUG):
        resp = pii_request(client)
    assert resp.status_code == 500
    out = capsys.readouterr()
    assert_no_pii(caplog.text, out.out, out.err, resp.text)
