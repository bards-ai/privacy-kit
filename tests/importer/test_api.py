"""Import API endpoint tests: preview counts, the session list, import runs
with filters and exclusions, request validation, and plaintext redaction."""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sqlmodel")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from importer_helpers import (
    CODEX_ID,
    SESSION_ID,
    make_detector,
    write_claude_session,
    write_codex_session,
)
from sqlmodel import Session, select

from privacy_kit.gateway.config import Settings
from privacy_kit.gateway.importer import claude_code, codex
from privacy_kit.gateway.proxy import create_app
from privacy_kit.gateway.store import AuditStore, Interaction


def make_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, expose_plaintext: bool = True
) -> tuple[TestClient, AuditStore]:
    """TestClient over fixture history roots, plus the store it writes to.

    The claude session is backdated so mtime ordering is deterministic:
    codex is always the newer of the two.
    """
    claude_root = tmp_path / "claude"
    codex_root = tmp_path / "codex"
    claude_path = write_claude_session(claude_root)
    codex_path = write_codex_session(codex_root)
    now = time.time()
    os.utime(claude_path, (now - 100, now - 100))
    os.utime(codex_path, (now, now))
    monkeypatch.setattr(claude_code, "default_root", lambda: claude_root)
    monkeypatch.setattr(codex, "default_root", lambda: codex_root)

    store = AuditStore(tmp_path / "audit.sqlite")
    app = create_app(
        detector=make_detector(),
        store=store,
        settings=Settings(_env_file=None, save_texts="all", expose_plaintext=expose_plaintext),
    )
    return TestClient(app), store


@pytest.fixture
def client_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, AuditStore]:
    return make_client(tmp_path, monkeypatch)


@pytest.fixture
def client(client_store: tuple[TestClient, AuditStore]) -> TestClient:
    return client_store[0]


@pytest.fixture
def store(client_store: tuple[TestClient, AuditStore]) -> AuditStore:
    return client_store[1]


def wait_done(client: TestClient) -> dict[str, Any]:
    for _ in range(100):
        status: dict[str, Any] = client.get("/api/v1/import/status").json()
        if status["state"] in ("done", "error"):
            return status
        time.sleep(0.05)
    return status


def run_import_and_wait(client: TestClient, body: dict[str, Any]) -> dict[str, Any]:
    assert client.post("/api/v1/import", json=body).status_code == 202
    status = wait_done(client)
    assert status["state"] == "done"
    return status


# --- Import runs ---------------------------------------------------------------


def test_import_flow(client: TestClient) -> None:
    preview = client.get("/api/v1/import/preview").json()
    assert preview["sources"]["claude-code"] == {"found": 1, "new": 1, "imported": 0}
    assert preview["sources"]["codex"] == {"found": 1, "new": 1, "imported": 0}

    status = run_import_and_wait(client, {})
    assert status["imported"] == 2
    assert status["turns"] == 3

    # the run flips the preview split for both sources
    preview = client.get("/api/v1/import/preview").json()
    assert all(s["imported"] == 1 for s in preview["sources"].values())


def test_import_dry_run_counts_without_writing(client: TestClient, store: AuditStore) -> None:
    status = run_import_and_wait(client, {"dry_run": True})
    assert status["dry_run"] is True
    assert status["imported"] == 2
    assert status["turns"] == 3
    assert status["entities"] == 0
    with Session(store.engine) as session:
        assert session.exec(select(Interaction)).all() == []
    preview = client.get("/api/v1/import/preview").json()
    assert all(s["new"] == 1 for s in preview["sources"].values())


def test_import_filters_reach_runner_and_echo_in_status(client: TestClient) -> None:
    body = {
        "sources": ["claude-code"],
        "project": "zzz",
        "since": "2000-01-01",
        "until": "2100-12-31",
    }
    status = run_import_and_wait(client, body)
    assert (status["found"], status["imported"]) == (0, 0)
    assert status["project"] == "zzz"
    assert status["since"].startswith("2000-01-01")
    assert status["until"].startswith("2100-12-31T23:59:59")


def test_import_exclude_session_ids(client: TestClient) -> None:
    status = run_import_and_wait(client, {"exclude_session_ids": [SESSION_ID]})
    assert (status["found"], status["imported"]) == (1, 1)

    # the excluded claude session stays new; the codex one was imported
    preview = client.get("/api/v1/import/preview").json()
    assert preview["sources"]["claude-code"] == {"found": 1, "new": 1, "imported": 0}
    assert preview["sources"]["codex"]["imported"] == 1
    sessions = client.get("/api/v1/import/preview/sessions").json()["sessions"]
    flags = {s["source"]: s["imported"] for s in sessions}
    assert flags == {"claude-code": False, "codex": True}


@pytest.mark.parametrize(
    "body",
    [
        {"sources": ["nope"]},
        {"since": "nope"},
        {"since": 123},
        {"until": "nope"},
        {"until": 123},
        {"project": 7},
        {"dry_run": "yes"},
        {"exclude_session_ids": "x"},
        {"exclude_session_ids": [1]},
    ],
)
def test_import_rejects_bad_body(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/api/v1/import", json=body).status_code == 400


# --- Preview -------------------------------------------------------------------


def test_preview_project_filter(client: TestClient) -> None:
    # project narrows claude-code only; codex ignores it by design
    preview = client.get("/api/v1/import/preview", params={"project": "user-proj"}).json()
    assert preview["sources"]["claude-code"]["found"] == 1
    preview = client.get("/api/v1/import/preview", params={"project": "zzz"}).json()
    assert preview["sources"]["claude-code"]["found"] == 0
    assert preview["sources"]["codex"]["found"] == 1


def test_preview_since_until_filter_by_mtime(client: TestClient) -> None:
    # the fixtures were just written, so their mtimes are "now"
    for params, found in (
        ({"since": "2000-01-01"}, 1),
        ({"since": "2100-01-01"}, 0),
        ({"until": "2100-01-01"}, 1),
        ({"until": "2000-01-01"}, 0),
    ):
        preview = client.get("/api/v1/import/preview", params=params).json()
        assert all(s["found"] == found for s in preview["sources"].values()), params
    # a date-only until covers its whole day, so "today" includes the fixtures
    today = datetime.now().astimezone().strftime("%Y-%m-%d")
    preview = client.get("/api/v1/import/preview", params={"until": today}).json()
    assert all(s["found"] == 1 for s in preview["sources"].values())


@pytest.mark.parametrize("params", [{"since": "not-a-date"}, {"until": "not-a-date"}])
def test_preview_rejects_bad_dates(client: TestClient, params: dict[str, str]) -> None:
    resp = client.get("/api/v1/import/preview", params=params)
    assert resp.status_code == 400
    assert "error" in resp.json()


# --- Preview session list --------------------------------------------------------


def test_preview_sessions_newest_first(client: TestClient) -> None:
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


def test_preview_sessions_limit_caps_page_not_total(client: TestClient) -> None:
    body = client.get("/api/v1/import/preview/sessions", params={"limit": 1}).json()
    assert body["total"] == 2
    assert [s["source"] for s in body["sessions"]] == ["codex"]


def test_preview_sessions_filters(client: TestClient) -> None:
    body = client.get("/api/v1/import/preview/sessions", params={"sources": "codex"}).json()
    assert [s["source"] for s in body["sessions"]] == ["codex"]
    # project narrows claude-code only (codex ignores it by design)
    body = client.get("/api/v1/import/preview/sessions", params={"project": "zzz"}).json()
    assert [s["source"] for s in body["sessions"]] == ["codex"]


@pytest.mark.parametrize(
    "params", [{"sources": "nope"}, {"since": "not-a-date"}, {"until": "not-a-date"}]
)
def test_preview_sessions_rejects_bad_params(client: TestClient, params: dict[str, str]) -> None:
    resp = client.get("/api/v1/import/preview/sessions", params=params)
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_preview_sessions_imported_flag_flips(client: TestClient) -> None:
    run_import_and_wait(client, {"sources": ["claude-code"]})
    body = client.get("/api/v1/import/preview/sessions").json()
    flags = {s["source"]: s["imported"] for s in body["sessions"]}
    assert flags == {"claude-code": True, "codex": False}


def test_preview_sessions_respects_expose_plaintext(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _store = make_client(tmp_path, monkeypatch, expose_plaintext=False)

    body = client.get("/api/v1/import/preview/sessions").json()
    assert body["titles_redacted"] is True
    assert all(s["title"] is None for s in body["sessions"])
    by_source = {s["source"]: s for s in body["sessions"]}
    # The Claude project slug is path-derived; the Codex cwd would need a file
    # read, so it is withheld along with the title.
    assert by_source["claude-code"]["project"] == "-home-user-proj"
    assert by_source["codex"]["project"] is None
