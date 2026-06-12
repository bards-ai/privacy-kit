"""End-to-end tests of the Claude Code path with the *real* model.

The full production pipeline — real detector, real ``anonymize`` /
``deanonymize``, real audit store — with only the network hop faked: the
"upstream" echoes the sanitized text it received back into its reply, so exact
rehydration proves the placeholders round-tripped. Gated behind
``PII_RUN_MODEL_TESTS=1`` (see ``make test-model``).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from privacy_kit.core.detectors import BardsAiOnnxDetector
from privacy_kit.gateway.proxy import ForwardResult, create_app
from privacy_kit.gateway.store import AuditStore

USER_TEXT = "Hi, I'm Anna Kowalska. Write to me at anna.kowalska@example.com, please."
RAW_PII = ("Anna Kowalska", "anna.kowalska@example.com")


class EchoUpstream:
    """A fake Anthropic upstream that quotes the (sanitized) user text back."""

    def __init__(self) -> None:
        self.seen_payload: dict[str, Any] | None = None

    async def __call__(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> ForwardResult:
        self.seen_payload = payload
        echoed = payload["messages"][0]["content"]
        return ForwardResult(
            200,
            {
                "content": [{"type": "text", "text": f"You said: {echoed}"}],
                "usage": {"input_tokens": 11, "output_tokens": 23},
            },
            {},
        )


def test_claude_code_path_end_to_end(detector: BardsAiOnnxDetector, tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite")
    upstream = EchoUpstream()
    client = TestClient(create_app(detector=detector, store=store, forwarder=upstream))

    resp = client.post(
        "/v1/messages",
        json={"model": "claude-opus-4-8", "messages": [{"role": "user", "content": USER_TEXT}]},
    )
    assert resp.status_code == 200

    # 1. Nothing sensitive crossed the (fake) wire.
    sent = json.dumps(upstream.seen_payload)
    for raw in RAW_PII:
        assert raw not in sent
    assert "[" in upstream.seen_payload["messages"][0]["content"]  # type: ignore[index]

    # 2. The client got the real values back, exactly.
    assert resp.json()["content"][0]["text"] == f"You said: {USER_TEXT}"

    # 3. Audited: metadata recorded; raw values appear only in the texts table
    #    (saved per PII_SAVE_TEXTS), never in any other table.
    row = store.recent()[0]
    assert row.source == "claude-code"
    assert row.entity_total >= 2
    assert (row.input_tokens, row.output_tokens) == (11, 23)
    assert row.id is not None
    saved = store.texts(row.id)
    assert saved and saved[0].original == USER_TEXT
    assert "[" in saved[0].anonymized and "Anna Kowalska" not in saved[0].anonymized
    conn = sqlite3.connect(tmp_path / "audit.sqlite")
    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        if table == "interactiontext":
            continue
        dump = "\n".join(
            str(cell) for r in conn.execute(f"SELECT * FROM {table}").fetchall() for cell in r
        )
        for raw in RAW_PII:
            assert raw not in dump, f"raw PII {raw!r} leaked into table {table!r}"
    conn.close()


class EchoStreamUpstream:
    """A fake streaming upstream: replays the sanitized text in tiny SSE deltas.

    Three-character deltas guarantee every placeholder is split across chunk
    boundaries, exercising the partial-placeholder buffering for real.
    """

    @asynccontextmanager
    async def open(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> AsyncIterator[tuple[int, dict[str, str], AsyncIterator[str]]]:
        text = payload["messages"][0]["content"]

        async def lines() -> AsyncIterator[str]:
            yield 'data: {"type":"message_start","message":{"usage":{"input_tokens":7}}}'
            yield ""
            for i in range(0, len(text), 3):
                delta = {"type": "text_delta", "text": text[i : i + 3]}
                obj = {"type": "content_block_delta", "index": 0, "delta": delta}
                yield "data: " + json.dumps(obj)
                yield ""
            yield "event: content_block_stop"
            yield 'data: {"type":"content_block_stop","index":0}'
            yield ""
            yield 'data: {"type":"message_delta","usage":{"output_tokens":13}}'
            yield ""

        yield 200, {"content-type": "text/event-stream"}, lines()


def test_claude_code_streaming_end_to_end(detector: BardsAiOnnxDetector, tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite")
    client = TestClient(
        create_app(detector=detector, store=store, stream_forwarder=EchoStreamUpstream())
    )

    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-8",
            "stream": True,
            "messages": [{"role": "user", "content": USER_TEXT}],
        },
    )
    assert resp.status_code == 200

    # Reassemble the streamed text the client would render.
    streamed = ""
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            obj = json.loads(line[len("data: ") :])
            if obj.get("type") == "content_block_delta":
                streamed += obj["delta"]["text"]
    assert streamed == USER_TEXT  # placeholders rehydrated across split deltas

    row = store.recent()[0]
    assert row.entity_total >= 2
    assert (row.input_tokens, row.output_tokens) == (7, 13)
