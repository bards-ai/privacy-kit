"""Importer tests: parsers against fixture JSONL, runner idempotency and
placeholder consistency. Model-free — uses a value-matching stub detector."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sqlmodel")

from sqlmodel import Session, col, select

from privacy_kit.core.types import Span
from privacy_kit.gateway.config import Settings
from privacy_kit.gateway.importer import claude_code, codex, run_import
from privacy_kit.gateway.store import AuditStore, Interaction


class ValueDetector:
    """Flags every occurrence of the configured literal values."""

    def __init__(self, values: dict[str, str]) -> None:
        self._values = values  # value -> label

    def detect(self, text: str) -> list[Span]:
        spans = []
        for value, label in self._values.items():
            start = 0
            while (idx := text.find(value, start)) != -1:
                spans.append(Span(start=idx, end=idx + len(value), label=label, score=1.0))
                start = idx + len(value)
        return spans


SESSION_ID = "2dd0ffca-0246-4d59-8100-779f7f90b198"


def _cc_line(**kw: Any) -> str:
    return json.dumps(kw)


def write_claude_session(root: Path, session_id: str = SESSION_ID) -> Path:
    project = root / "-home-user-proj"
    project.mkdir(parents=True, exist_ok=True)
    path = project / f"{session_id}.jsonl"
    lines = [
        _cc_line(type="mode", mode="normal", sessionId=session_id),
        # meta + command wrappers: all skipped
        _cc_line(
            type="user",
            isMeta=True,
            timestamp="2026-07-01T09:00:00.000Z",
            message={"role": "user", "content": "<local-command-caveat>x</local-command-caveat>"},
        ),
        _cc_line(
            type="user",
            timestamp="2026-07-01T09:00:01.000Z",
            message={"role": "user", "content": "<command-name>/clear</command-name>"},
        ),
        # sidechain: skipped
        _cc_line(
            type="user",
            isSidechain=True,
            timestamp="2026-07-01T09:00:02.000Z",
            message={"role": "user", "content": "subagent prompt"},
        ),
        # turn 1: prompt with PII, tool result, assistant reply
        _cc_line(
            type="user",
            timestamp="2026-07-01T09:01:00.000Z",
            message={"role": "user", "content": "email alice@example.com please"},
        ),
        _cc_line(
            type="assistant",
            timestamp="2026-07-01T09:01:05.000Z",
            message={
                "id": "msg_1",
                "model": "claude-opus-4-8",
                "role": "assistant",
                "usage": {"input_tokens": 10, "output_tokens": 20},
                "content": [
                    {"type": "thinking", "thinking": "secret reasoning"},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
        ),
        _cc_line(
            type="user",
            timestamp="2026-07-01T09:01:06.000Z",
            message={
                "role": "user",
                "content": [{"type": "tool_result", "content": "contact alice@example.com"}],
            },
        ),
        _cc_line(
            type="assistant",
            timestamp="2026-07-01T09:01:10.000Z",
            message={
                "id": "msg_2",
                "model": "claude-opus-4-8",
                "role": "assistant",
                "usage": {"input_tokens": 30, "output_tokens": 5},
                "content": [{"type": "text", "text": "Done, mailed alice@example.com"}],
            },
        ),
        # synthetic assistant: skipped
        _cc_line(
            type="assistant",
            timestamp="2026-07-01T09:01:11.000Z",
            message={"id": "msg_3", "model": "<synthetic>", "role": "assistant", "content": []},
        ),
        # turn 2: clean prompt
        _cc_line(
            type="user",
            timestamp="2026-07-01T09:02:00.000Z",
            message={"role": "user", "content": "thanks, now run the tests"},
        ),
        _cc_line(
            type="assistant",
            timestamp="2026-07-01T09:02:05.000Z",
            message={
                "id": "msg_4",
                "model": "claude-opus-4-8",
                "role": "assistant",
                "usage": {"input_tokens": 40, "output_tokens": 8},
                "content": [{"type": "text", "text": "All green."}],
            },
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


CODEX_ID = "019f2349-1146-7232-821e-b0dc85e50cd6"


def write_codex_session(root: Path, session_id: str = CODEX_ID) -> Path:
    day = root / "2026" / "07" / "02"
    day.mkdir(parents=True, exist_ok=True)
    path = day / f"rollout-2026-07-02T16-43-38-{session_id}.jsonl"
    lines = [
        json.dumps(
            {
                "timestamp": "2026-07-02T14:43:45.989Z",
                "type": "session_meta",
                "payload": {"id": session_id, "cwd": "/home/user/proj"},
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-07-02T14:43:46.000Z",
                "type": "turn_context",
                "payload": {"model": "gpt-5.4-mini"},
            }
        ),
        # developer response_item message: skipped
        json.dumps(
            {
                "timestamp": "2026-07-02T14:43:46.100Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "<user_instructions>x"}],
                },
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-07-02T14:44:00.000Z",
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "call bob@example.com"},
            }
        ),
        # user response_item duplicates the event_msg: skipped
        json.dumps(
            {
                "timestamp": "2026-07-02T14:44:00.100Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "call bob@example.com"}],
                },
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-07-02T14:44:01.000Z",
                "type": "response_item",
                "payload": {"type": "reasoning", "encrypted_content": "zzz"},
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-07-02T14:44:02.000Z",
                "type": "response_item",
                "payload": {"type": "function_call_output", "output": "bob@example.com found"},
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-07-02T14:44:03.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Calling bob@example.com now."}],
                },
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-07-02T14:44:04.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {"last_token_usage": {"input_tokens": 100, "output_tokens": 12}},
                },
            }
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- Parsers -----------------------------------------------------------------


def test_parse_claude_session(tmp_path: Path) -> None:
    path = write_claude_session(tmp_path)
    session = claude_code.parse_session(path)
    assert session is not None
    assert session.session_id == SESSION_ID
    assert session.source == "claude-code-import"
    assert len(session.turns) == 2

    turn1, turn2 = session.turns
    assert turn1.segments == [
        ("user", "email alice@example.com please"),
        ("tool", "contact alice@example.com"),
        ("assistant", "Done, mailed alice@example.com"),
    ]
    assert turn1.model == "claude-opus-4-8"
    assert turn1.timestamp == datetime(2026, 7, 1, 9, 1, tzinfo=timezone.utc)
    assert turn1.input_tokens == 30  # max across the exchange
    assert turn1.output_tokens == 25  # summed, deduped by message id
    assert turn2.segments == [
        ("user", "thanks, now run the tests"),
        ("assistant", "All green."),
    ]


def test_parse_claude_skips_empty_session(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    path = project / "empty.jsonl"
    path.write_text('{"type":"mode","mode":"normal"}\nnot json\n', encoding="utf-8")
    assert claude_code.parse_session(path) is None


def test_discover_claude_sessions_project_filter(tmp_path: Path) -> None:
    write_claude_session(tmp_path)
    assert claude_code.discover_sessions(tmp_path, project="user-proj")
    assert claude_code.discover_sessions(tmp_path, project="other") == []
    assert claude_code.discover_sessions(tmp_path / "missing") == []


def test_parse_until() -> None:
    from privacy_kit.gateway.importer.runner import parse_until

    end_of_day = parse_until("2026-06-01")
    assert end_of_day is not None
    assert (end_of_day.hour, end_of_day.minute, end_of_day.second) == (23, 59, 59)
    assert end_of_day.tzinfo is not None  # naive input is localized
    exact = parse_until("2026-06-01T12:00:00")
    assert exact is not None
    assert exact.hour == 12
    assert parse_until("not-a-date") is None


def test_parse_codex_session(tmp_path: Path) -> None:
    path = write_codex_session(tmp_path)
    session = codex.parse_session(path)
    assert session is not None
    assert session.session_id == CODEX_ID
    assert session.source == "codex-import"
    assert len(session.turns) == 1
    turn = session.turns[0]
    assert turn.segments == [
        ("user", "call bob@example.com"),
        ("tool", "bob@example.com found"),
        ("assistant", "Calling bob@example.com now."),
    ]
    assert turn.model == "gpt-5.4-mini"
    assert turn.input_tokens == 100
    assert turn.output_tokens == 12


# --- Preview info (titles for the import list) --------------------------------


def test_truncate_title() -> None:
    from privacy_kit.gateway.importer.base import TITLE_MAX_CHARS, truncate_title

    assert truncate_title("short prompt") == "short prompt"
    assert truncate_title("  first line \n second line ") == "first line"
    assert truncate_title("   \n\t ") is None
    capped = truncate_title("x" * 500)
    assert capped is not None
    assert len(capped) == TITLE_MAX_CHARS
    assert capped.endswith("…")


def test_claude_preview_info_skips_plumbing(tmp_path: Path) -> None:
    # meta, command-wrapper, and sidechain entries come first in the fixture;
    # the title must be the first real human prompt after them.
    path = write_claude_session(tmp_path)
    assert claude_code.preview_info(path) == ("email alice@example.com please", "-home-user-proj")


def test_claude_preview_info_no_prompt(tmp_path: Path) -> None:
    project = tmp_path / "-home-user-proj"
    project.mkdir(parents=True)
    path = project / "plumbing.jsonl"
    lines = [
        _cc_line(type="mode", mode="normal"),
        _cc_line(type="user", isMeta=True, message={"role": "user", "content": "meta text"}),
        _cc_line(
            type="user",
            message={"role": "user", "content": "<command-name>/clear</command-name>"},
        ),
        _cc_line(
            type="user",
            message={"role": "user", "content": [{"type": "tool_result", "content": "output"}]},
        ),
        "not json",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert claude_code.preview_info(path) == (None, "-home-user-proj")


def test_claude_preview_info_skips_injected_blocks(tmp_path: Path) -> None:
    # <ide_opened_file>/<system-reminder> blocks ride along inside real prompts;
    # they are imported but must not become the title.
    project = tmp_path / "-home-user-proj"
    project.mkdir(parents=True)
    path = project / "ide.jsonl"
    lines = [
        _cc_line(
            type="user",
            message={
                "role": "user",
                "content": [
                    {"type": "text", "text": "<ide_opened_file>README.md</ide_opened_file>"},
                    {"type": "text", "text": " <system-reminder>ctx</system-reminder>"},
                ],
            },
        ),
        _cc_line(
            type="user",
            message={
                "role": "user",
                "content": [
                    {"type": "text", "text": "<ide_opened_file>a.py</ide_opened_file>"},
                    {"type": "text", "text": "fix the bug"},
                ],
            },
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert claude_code.preview_info(path) == ("fix the bug", "-home-user-proj")


def test_codex_preview_info_prefers_event_msg(tmp_path: Path) -> None:
    # the developer/user response_item copies carry injected wrappers; the
    # event_msg user_message is the clean human text. Project is the meta cwd.
    path = write_codex_session(tmp_path)
    assert codex.preview_info(path) == ("call bob@example.com", "/home/user/proj")


def test_codex_preview_info_no_user_message(tmp_path: Path) -> None:
    day = tmp_path / "2026" / "07" / "02"
    day.mkdir(parents=True)
    path = day / f"rollout-2026-07-02T16-43-38-{CODEX_ID}.jsonl"
    meta = {"type": "session_meta", "payload": {"id": CODEX_ID, "cwd": "/home/user/proj"}}
    path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
    assert codex.preview_info(path) == (None, "/home/user/proj")


# --- Runner ------------------------------------------------------------------


def make_detector() -> ValueDetector:
    return ValueDetector({"alice@example.com": "EMAIL_ADDRESS", "bob@example.com": "EMAIL_ADDRESS"})


def test_run_import_end_to_end(tmp_path: Path) -> None:
    claude_root = tmp_path / "claude"
    codex_root = tmp_path / "codex"
    write_claude_session(claude_root)
    write_codex_session(codex_root)
    store = AuditStore(tmp_path / "audit.sqlite")

    job = run_import(
        store,
        make_detector(),
        settings=Settings(save_texts="all"),
        claude_root=claude_root,
        codex_root=codex_root,
    )
    assert job.state == "done"
    assert (job.found, job.imported, job.skipped, job.failed) == (2, 2, 0, 0)
    assert job.turns == 3

    with Session(store.engine) as session:
        rows = session.exec(select(Interaction).order_by(col(Interaction.created_at))).all()
    assert len(rows) == 3
    by_conv: dict[str | None, list[Interaction]] = {}
    for row in rows:
        by_conv.setdefault(row.conversation_id, []).append(row)
    assert set(by_conv) == {SESSION_ID, CODEX_ID}

    cc1, cc2 = by_conv[SESSION_ID]
    assert cc1.created_at.replace(tzinfo=timezone.utc) == datetime(
        2026, 7, 1, 9, 1, tzinfo=timezone.utc
    )
    assert cc1.entity_counts == {"EMAIL_ADDRESS": 1}  # distinct values, not occurrences
    assert cc1.policy == "imported"
    assert cc2.entity_counts == {}

    # placeholder consistency across the conversation's turns
    assert cc1.id is not None
    texts = store.texts(cc1.id)
    assert all("[EMAIL_ADDRESS_1]" in t.anonymized for t in texts)
    assert [t.category for t in texts] == ["user", "tool", "assistant"]


def test_run_import_is_idempotent(tmp_path: Path) -> None:
    claude_root = tmp_path / "claude"
    write_claude_session(claude_root)
    store = AuditStore(tmp_path / "audit.sqlite")

    first = run_import(
        store,
        make_detector(),
        settings=Settings(save_texts="all"),
        claude_root=claude_root,
        codex_root=tmp_path / "no-codex",
    )
    assert first.imported == 1
    second = run_import(
        store,
        make_detector(),
        settings=Settings(save_texts="all"),
        claude_root=claude_root,
        codex_root=tmp_path / "no-codex",
    )
    assert second.imported == 0
    assert second.skipped == 1
    with Session(store.engine) as session:
        count = len(session.exec(select(Interaction)).all())
    assert count == 2  # still just the first run's two turns


def test_run_import_save_texts_anonymized_keeps_only_pii_segments(tmp_path: Path) -> None:
    claude_root = tmp_path / "claude"
    write_claude_session(claude_root)
    store = AuditStore(tmp_path / "audit.sqlite")

    run_import(
        store,
        make_detector(),
        settings=Settings(save_texts="anonymized"),
        claude_root=claude_root,
        codex_root=tmp_path / "no-codex",
    )
    with Session(store.engine) as session:
        rows = session.exec(select(Interaction).order_by(col(Interaction.created_at))).all()
    pii_turn, clean_turn = rows
    assert pii_turn.id is not None and clean_turn.id is not None
    assert len(store.texts(pii_turn.id)) == 3  # all three segments contained the email
    assert store.texts(clean_turn.id) == []


def test_run_import_dry_run_writes_nothing(tmp_path: Path) -> None:
    claude_root = tmp_path / "claude"
    write_claude_session(claude_root)
    store = AuditStore(tmp_path / "audit.sqlite")

    job = run_import(
        store,
        make_detector(),
        dry_run=True,
        settings=Settings(save_texts="all"),
        claude_root=claude_root,
        codex_root=tmp_path / "no-codex",
    )
    assert job.imported == 1
    assert job.turns == 2
    with Session(store.engine) as session:
        assert session.exec(select(Interaction)).all() == []


def test_run_import_excludes_selected_sessions(tmp_path: Path) -> None:
    claude_root = tmp_path / "claude"
    codex_root = tmp_path / "codex"
    write_claude_session(claude_root)
    write_codex_session(codex_root)
    store = AuditStore(tmp_path / "audit.sqlite")

    job = run_import(
        store,
        make_detector(),
        exclude_session_ids={SESSION_ID},
        settings=Settings(save_texts="all"),
        claude_root=claude_root,
        codex_root=codex_root,
    )
    assert job.state == "done"
    assert (job.found, job.imported, job.skipped) == (1, 1, 0)
    with Session(store.engine) as session:
        rows = session.exec(select(Interaction)).all()
    assert {row.conversation_id for row in rows} == {CODEX_ID}


# --- API endpoints -----------------------------------------------------------


def _make_import_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, expose_plaintext: bool = True
) -> tuple[Any, AuditStore]:
    """TestClient over fixture history roots, plus the store it writes to."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from privacy_kit.gateway.importer import claude_code as cc_mod
    from privacy_kit.gateway.importer import codex as codex_mod
    from privacy_kit.gateway.proxy import create_app

    claude_root = tmp_path / "claude"
    codex_root = tmp_path / "codex"
    write_claude_session(claude_root)
    write_codex_session(codex_root)
    monkeypatch.setattr(cc_mod, "default_root", lambda: claude_root)
    monkeypatch.setattr(codex_mod, "default_root", lambda: codex_root)

    store = AuditStore(tmp_path / "audit.sqlite")
    app = create_app(
        detector=make_detector(),
        store=store,
        settings=Settings(_env_file=None, save_texts="all", expose_plaintext=expose_plaintext),
    )
    return TestClient(app), store


def _wait_done(client: Any) -> dict[str, Any]:
    for _ in range(100):
        status: dict[str, Any] = client.get("/api/v1/import/status").json()
        if status["state"] in ("done", "error"):
            return status
        time.sleep(0.05)
    return status


def test_import_api_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _store = _make_import_client(tmp_path, monkeypatch)

    preview = client.get("/api/v1/import/preview").json()
    assert preview["sources"]["claude-code"] == {"found": 1, "new": 1, "imported": 0}
    assert preview["sources"]["codex"] == {"found": 1, "new": 1, "imported": 0}

    assert client.post("/api/v1/import", json={"sources": ["nope"]}).status_code == 400

    resp = client.post("/api/v1/import", json={})
    assert resp.status_code == 202

    status = _wait_done(client)
    assert status["state"] == "done"
    assert status["imported"] == 2
    assert status["turns"] == 3

    preview = client.get("/api/v1/import/preview").json()
    assert preview["sources"]["claude-code"]["imported"] == 1


def test_import_api_filters_and_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, store = _make_import_client(tmp_path, monkeypatch)

    # preview: project narrows claude-code only; codex ignores it by design
    preview = client.get("/api/v1/import/preview", params={"project": "user-proj"}).json()
    assert preview["sources"]["claude-code"]["found"] == 1
    preview = client.get("/api/v1/import/preview", params={"project": "zzz"}).json()
    assert preview["sources"]["claude-code"]["found"] == 0
    assert preview["sources"]["codex"]["found"] == 1

    # preview: since/until filter by file mtime (fixtures were just written)
    preview = client.get("/api/v1/import/preview", params={"since": "2000-01-01"}).json()
    assert all(s["found"] == 1 for s in preview["sources"].values())
    preview = client.get("/api/v1/import/preview", params={"since": "2100-01-01"}).json()
    assert all(s["found"] == 0 for s in preview["sources"].values())
    preview = client.get("/api/v1/import/preview", params={"until": "2100-01-01"}).json()
    assert all(s["found"] == 1 for s in preview["sources"].values())
    preview = client.get("/api/v1/import/preview", params={"until": "2000-01-01"}).json()
    assert all(s["found"] == 0 for s in preview["sources"].values())
    # a date-only until covers its whole day, so "today" includes the fixtures
    today = datetime.now().astimezone().strftime("%Y-%m-%d")
    preview = client.get("/api/v1/import/preview", params={"until": today}).json()
    assert all(s["found"] == 1 for s in preview["sources"].values())

    for param in ("since", "until"):
        resp = client.get("/api/v1/import/preview", params={param: "not-a-date"})
        assert resp.status_code == 400
        assert "error" in resp.json()

    for bad in (
        {"since": "nope"},
        {"since": 123},
        {"until": "nope"},
        {"until": 123},
        {"project": 7},
        {"dry_run": "yes"},
    ):
        assert client.post("/api/v1/import", json=bad).status_code == 400

    # dry run: counts everything, writes nothing
    assert client.post("/api/v1/import", json={"dry_run": True}).status_code == 202
    status = _wait_done(client)
    assert status["state"] == "done"
    assert status["dry_run"] is True
    assert status["imported"] == 2
    assert status["turns"] == 3
    assert status["entities"] == 0
    with Session(store.engine) as session:
        assert session.exec(select(Interaction)).all() == []
    preview = client.get("/api/v1/import/preview").json()
    assert all(s["new"] == 1 for s in preview["sources"].values())

    # filters reach the runner and are echoed in the snapshot
    body = {
        "sources": ["claude-code"],
        "project": "zzz",
        "since": "2000-01-01",
        "until": "2100-12-31",
    }
    assert client.post("/api/v1/import", json=body).status_code == 202
    status = _wait_done(client)
    assert status["state"] == "done"
    assert (status["found"], status["imported"]) == (0, 0)
    assert status["project"] == "zzz"
    assert status["since"].startswith("2000-01-01")
    assert status["until"].startswith("2100-12-31T23:59:59")

    # real unfiltered run flips the preview split
    assert client.post("/api/v1/import", json={}).status_code == 202
    status = _wait_done(client)
    assert status["imported"] == 2
    preview = client.get("/api/v1/import/preview").json()
    assert all(s["imported"] == 1 for s in preview["sources"].values())


def test_import_preview_sessions_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _store = _make_import_client(tmp_path, monkeypatch)
    claude_path = tmp_path / "claude" / "-home-user-proj" / f"{SESSION_ID}.jsonl"
    codex_path = (
        tmp_path / "codex" / "2026" / "07" / "02" / f"rollout-2026-07-02T16-43-38-{CODEX_ID}.jsonl"
    )
    now = time.time()
    os.utime(claude_path, (now - 100, now - 100))
    os.utime(codex_path, (now, now))

    body = client.get("/api/v1/import/preview/sessions").json()
    assert body["total"] == 2
    assert body["titles_redacted"] is False
    newest, oldest = body["sessions"]  # newest-first by mtime
    assert newest["source"] == "codex"
    assert newest["id"] == CODEX_ID
    assert newest["title"] == "call bob@example.com"
    assert newest["project"] == "/home/user/proj"
    assert newest["imported"] is False
    assert datetime.fromisoformat(newest["modified_at"]).tzinfo is not None
    assert oldest == {
        "source": "claude-code",
        "id": SESSION_ID,
        "title": "email alice@example.com please",
        "project": "-home-user-proj",
        "modified_at": oldest["modified_at"],
        "imported": False,
    }

    # limit caps the page, not the total
    body = client.get("/api/v1/import/preview/sessions", params={"limit": 1}).json()
    assert body["total"] == 2
    assert [s["source"] for s in body["sessions"]] == ["codex"]

    # sources filter and validation
    body = client.get("/api/v1/import/preview/sessions", params={"sources": "codex"}).json()
    assert [s["source"] for s in body["sessions"]] == ["codex"]
    for bad in ({"sources": "nope"}, {"since": "not-a-date"}, {"until": "not-a-date"}):
        resp = client.get("/api/v1/import/preview/sessions", params=bad)
        assert resp.status_code == 400
        assert "error" in resp.json()

    # project narrows claude-code only (codex ignores it by design)
    body = client.get("/api/v1/import/preview/sessions", params={"project": "zzz"}).json()
    assert [s["source"] for s in body["sessions"]] == ["codex"]

    # importing flips the flag
    assert client.post("/api/v1/import", json={"sources": ["claude-code"]}).status_code == 202
    assert _wait_done(client)["state"] == "done"
    body = client.get("/api/v1/import/preview/sessions").json()
    flags = {s["source"]: s["imported"] for s in body["sessions"]}
    assert flags == {"claude-code": True, "codex": False}


def test_import_api_exclude_session_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _store = _make_import_client(tmp_path, monkeypatch)

    for bad in ({"exclude_session_ids": "x"}, {"exclude_session_ids": [1]}):
        assert client.post("/api/v1/import", json=bad).status_code == 400

    resp = client.post("/api/v1/import", json={"exclude_session_ids": [SESSION_ID]})
    assert resp.status_code == 202
    status = _wait_done(client)
    assert status["state"] == "done"
    assert (status["found"], status["imported"]) == (1, 1)

    # the excluded claude session stays new; the codex one was imported
    preview = client.get("/api/v1/import/preview").json()
    assert preview["sources"]["claude-code"] == {"found": 1, "new": 1, "imported": 0}
    assert preview["sources"]["codex"]["imported"] == 1
    sessions = client.get("/api/v1/import/preview/sessions").json()["sessions"]
    flags = {s["source"]: s["imported"] for s in sessions}
    assert flags == {"claude-code": False, "codex": True}


def test_import_preview_sessions_respects_expose_plaintext(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _store = _make_import_client(tmp_path, monkeypatch, expose_plaintext=False)

    body = client.get("/api/v1/import/preview/sessions").json()
    assert body["titles_redacted"] is True
    assert all(s["title"] is None for s in body["sessions"])
    by_source = {s["source"]: s for s in body["sessions"]}
    # The Claude project slug is path-derived; the Codex cwd would need a file
    # read, so it is withheld along with the title.
    assert by_source["claude-code"]["project"] == "-home-user-proj"
    assert by_source["codex"]["project"] is None
