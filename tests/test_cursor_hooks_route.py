"""Cursor hooks: editing ~/.cursor/hooks.json via route.py and the CLI.

Cursor only routes its chat panel through a custom OpenAI base URL; Composer, the
agent loop, inline edit, and Tab stay on Cursor's backend. We register
``beforeSubmitPrompt`` and ``beforeReadFile`` hooks so privacy-kit can still
audit (and optionally block) PII on those surfaces. The edits must be idempotent
and must never clobber hooks the user defined.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("typer")

from typer.testing import CliRunner

from privacy_kit.gateway import route
from privacy_kit.gateway.cli import app

runner = CliRunner()

COMMAND = "/opt/venv/bin/privacy-kit hook cursor"


def _read(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def test_apply_registers_both_events(tmp_path: Path) -> None:
    hooks = tmp_path / "hooks.json"
    change = route.apply_cursor_hooks(COMMAND, path=hooks)

    data = _read(hooks)
    assert data["version"] == 1
    assert set(data["hooks"]) == set(route.CURSOR_HOOK_EVENTS)
    for event in route.CURSOR_HOOK_EVENTS:
        entries = data["hooks"][event]
        assert entries == [{"command": f"{COMMAND} {event}", "type": "command"}]
    assert change.events == route.CURSOR_HOOK_EVENTS
    assert change.path == hooks


def test_apply_is_idempotent(tmp_path: Path) -> None:
    hooks = tmp_path / "hooks.json"
    route.apply_cursor_hooks(COMMAND, path=hooks)
    route.apply_cursor_hooks(COMMAND, path=hooks)  # re-apply

    data = _read(hooks)
    for event in route.CURSOR_HOOK_EVENTS:
        # Exactly one privacy-kit entry per event — not duplicated.
        assert len(data["hooks"][event]) == 1


def test_apply_registers_after_agent_response_event(tmp_path: Path) -> None:
    hooks = tmp_path / "hooks.json"
    route.apply_cursor_hooks(COMMAND, path=hooks)

    data = _read(hooks)
    assert "afterAgentResponse" in data["hooks"]
    entry = data["hooks"]["afterAgentResponse"][0]
    assert entry == {"command": f"{COMMAND} afterAgentResponse", "type": "command"}


def test_apply_replaces_a_stale_privacy_kit_command(tmp_path: Path) -> None:
    hooks = tmp_path / "hooks.json"
    route.apply_cursor_hooks("/old/path/privacy-kit hook cursor", path=hooks)
    route.apply_cursor_hooks(COMMAND, path=hooks)

    data = _read(hooks)
    entry = data["hooks"]["beforeSubmitPrompt"][0]
    assert entry["command"] == f"{COMMAND} beforeSubmitPrompt"
    assert len(data["hooks"]["beforeSubmitPrompt"]) == 1


def test_apply_preserves_user_hooks(tmp_path: Path) -> None:
    hooks = tmp_path / "hooks.json"
    user_entry = {"command": "./format.sh", "type": "command"}
    hooks.write_text(
        json.dumps(
            {
                "version": 1,
                "hooks": {
                    "beforeSubmitPrompt": [user_entry],
                    "afterFileEdit": [user_entry],
                },
            }
        ),
        encoding="utf-8",
    )
    route.apply_cursor_hooks(COMMAND, path=hooks)

    data = _read(hooks)
    # The user's own beforeSubmitPrompt hook survives alongside ours.
    assert user_entry in data["hooks"]["beforeSubmitPrompt"]
    assert {"command": f"{COMMAND} beforeSubmitPrompt", "type": "command"} in data["hooks"][
        "beforeSubmitPrompt"
    ]
    # An unrelated event we don't touch is left exactly as it was.
    assert data["hooks"]["afterFileEdit"] == [user_entry]


def test_apply_rejects_invalid_json(tmp_path: Path) -> None:
    hooks = tmp_path / "hooks.json"
    hooks.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        route.apply_cursor_hooks(COMMAND, path=hooks)


def test_remove_deletes_only_our_entries(tmp_path: Path) -> None:
    hooks = tmp_path / "hooks.json"
    user_entry = {"command": "./format.sh", "type": "command"}
    hooks.write_text(
        json.dumps({"version": 1, "hooks": {"beforeSubmitPrompt": [user_entry]}}),
        encoding="utf-8",
    )
    route.apply_cursor_hooks(COMMAND, path=hooks)

    cleaned = route.remove_cursor_hooks(path=hooks)
    assert set(cleaned) == set(route.CURSOR_HOOK_EVENTS)

    data = _read(hooks)
    # Our entries gone; the user's hook stays; the emptied event we created is pruned.
    assert data["hooks"]["beforeSubmitPrompt"] == [user_entry]
    assert "beforeReadFile" not in data["hooks"]


def test_remove_is_idempotent_and_handles_missing_file(tmp_path: Path) -> None:
    hooks = tmp_path / "hooks.json"
    route.apply_cursor_hooks(COMMAND, path=hooks)
    assert route.remove_cursor_hooks(path=hooks)  # first removal does work
    assert route.remove_cursor_hooks(path=hooks) == []  # idempotent
    assert route.remove_cursor_hooks(path=tmp_path / "absent.json") == []


def test_remove_drops_empty_hooks_object(tmp_path: Path) -> None:
    hooks = tmp_path / "hooks.json"
    route.apply_cursor_hooks(COMMAND, path=hooks)
    route.remove_cursor_hooks(path=hooks)

    data = _read(hooks)
    assert "hooks" not in data  # nothing left, so the object is pruned


def test_cursor_hooks_path_scopes(tmp_path: Path) -> None:
    user = route.cursor_hooks_path("user")
    assert user == Path.home() / ".cursor" / "hooks.json"
    project = route.cursor_hooks_path("project", project_root=tmp_path)
    assert project == tmp_path / ".cursor" / "hooks.json"
    with pytest.raises(ValueError):
        route.cursor_hooks_path("global")


def test_cli_setup_cursor_apply_and_remove(tmp_path: Path) -> None:
    hooks = tmp_path / "hooks.json"

    result = runner.invoke(app, ["setup", "cursor", "--apply", "--settings-file", str(hooks)])
    assert result.exit_code == 0
    assert "Installed privacy-kit Cursor hooks" in result.stdout
    # Still prints the chat-panel pseudonymize instructions (layer 1).
    assert "Override OpenAI Base URL" in result.stdout
    data = _read(hooks)
    assert "beforeSubmitPrompt" in data["hooks"]
    assert "hook cursor beforeSubmitPrompt" in data["hooks"]["beforeSubmitPrompt"][0]["command"]

    result = runner.invoke(app, ["setup", "cursor", "--remove", "--settings-file", str(hooks)])
    assert result.exit_code == 0
    assert "Removed privacy-kit Cursor hooks" in result.stdout
    assert "hook cursor" not in hooks.read_text(encoding="utf-8")


def test_cli_setup_cursor_remove_on_absent_file_is_noop(tmp_path: Path) -> None:
    hooks = tmp_path / "hooks.json"
    result = runner.invoke(app, ["setup", "cursor", "--remove", "--settings-file", str(hooks)])
    assert result.exit_code == 0
    assert "nothing to do" in result.stdout
