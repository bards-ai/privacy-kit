"""Parser tests: Claude Code / Codex session JSONL against fixture files, plus
the preview titles shown in the import list. Model-free."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("sqlmodel")

from importer_helpers import (
    CODEX_ID,
    SESSION_ID,
    cc_line,
    write_claude_session,
    write_codex_session,
)

from privacy_kit.gateway.importer import claude_code, codex
from privacy_kit.gateway.importer.base import TITLE_MAX_CHARS, truncate_title
from privacy_kit.gateway.importer.runner import parse_until


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
        cc_line(type="mode", mode="normal"),
        cc_line(type="user", isMeta=True, message={"role": "user", "content": "meta text"}),
        cc_line(
            type="user",
            message={"role": "user", "content": "<command-name>/clear</command-name>"},
        ),
        cc_line(
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
        cc_line(
            type="user",
            message={
                "role": "user",
                "content": [
                    {"type": "text", "text": "<ide_opened_file>README.md</ide_opened_file>"},
                    {"type": "text", "text": " <system-reminder>ctx</system-reminder>"},
                ],
            },
        ),
        cc_line(
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


def test_claude_default_root_honors_config_dir_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
    assert claude_code.default_root() == tmp_path / "cfg" / "projects"
    monkeypatch.delenv("CLAUDE_CONFIG_DIR")
    assert claude_code.default_root() == Path.home() / ".claude" / "projects"


def test_codex_default_root_honors_codex_home_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "cx"))
    assert codex.default_root() == tmp_path / "cx" / "sessions"
    monkeypatch.delenv("CODEX_HOME")
    assert codex.default_root() == Path.home() / ".codex" / "sessions"
