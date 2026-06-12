"""Auto-routing tests: writing/restoring ANTHROPIC_BASE_URL in Claude Code settings."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("typer")

from typer.testing import CliRunner

from privacy_kit.gateway import route
from privacy_kit.gateway.cli import app
from privacy_kit.gateway.route import (
    apply_claude_code_route,
    remove_claude_code_route,
    revert_route,
)

runner = CliRunner()

BASE = "http://127.0.0.1:8787"


def read(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def test_apply_creates_settings_file_when_missing(tmp_path: Path) -> None:
    settings = tmp_path / ".claude" / "settings.json"
    change = apply_claude_code_route(BASE, settings)
    assert change.previous is None
    assert read(settings) == {"env": {"ANTHROPIC_BASE_URL": BASE}}


def test_apply_preserves_other_settings_and_records_previous(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "model": "opus",
                "env": {"ANTHROPIC_BASE_URL": "http://old:1", "FOO": "bar"},
            }
        )
    )
    change = apply_claude_code_route(BASE, settings)
    assert change.previous == "http://old:1"
    data = read(settings)
    assert data["model"] == "opus"  # untouched
    assert data["env"]["FOO"] == "bar"  # untouched
    assert data["env"]["ANTHROPIC_BASE_URL"] == BASE


def test_revert_restores_previous_value(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://old:1"}}))
    change = apply_claude_code_route(BASE, settings)
    assert revert_route(change)
    assert read(settings)["env"]["ANTHROPIC_BASE_URL"] == "http://old:1"


def test_revert_removes_key_when_there_was_none(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"model": "opus"}))
    change = apply_claude_code_route(BASE, settings)
    assert revert_route(change)
    assert read(settings) == {"model": "opus"}  # env block cleaned up entirely


def test_revert_never_clobbers_a_manual_user_edit(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    change = apply_claude_code_route(BASE, settings)
    # The user changes the value while the gateway is running...
    data = read(settings)
    data["env"]["ANTHROPIC_BASE_URL"] = "http://users-own-proxy:9"
    settings.write_text(json.dumps(data))
    # ...so shutdown must leave it alone.
    assert not revert_route(change)
    assert read(settings)["env"]["ANTHROPIC_BASE_URL"] == "http://users-own-proxy:9"


def test_apply_refuses_to_clobber_corrupt_json(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("{not json")
    with pytest.raises(ValueError):
        apply_claude_code_route(BASE, settings)
    assert settings.read_text() == "{not json"  # untouched


def test_remove_returns_removed_value_and_cleans_up(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    apply_claude_code_route(BASE, settings)
    assert remove_claude_code_route(settings) == BASE
    assert "env" not in read(settings)
    assert remove_claude_code_route(settings) is None  # idempotent


def test_cli_setup_apply_and_remove(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    result = runner.invoke(
        app, ["setup", "claude-code", "--apply", "--settings-file", str(settings)]
    )
    assert result.exit_code == 0
    assert "Wrote ANTHROPIC_BASE_URL" in result.stdout
    assert str(settings) in result.stdout  # we say exactly which file we override
    assert read(settings)["env"]["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:")

    result = runner.invoke(
        app, ["setup", "claude-code", "--remove", "--settings-file", str(settings)]
    )
    assert result.exit_code == 0
    assert "Removed ANTHROPIC_BASE_URL" in result.stdout
    assert "env" not in read(settings)


def test_cli_setup_apply_rejected_for_other_tools(tmp_path: Path) -> None:
    result = runner.invoke(app, ["setup", "cursor", "--apply"])
    assert result.exit_code == 1


def test_cli_setup_text_mentions_auto_apply() -> None:
    result = runner.invoke(app, ["setup", "claude-code"])
    assert result.exit_code == 0
    assert "--apply" in result.stdout
    assert "--route claude-code" in result.stdout


def test_serve_route_applies_and_restores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://old:1"}}))
    monkeypatch.setattr(route, "claude_settings_path", lambda: settings)

    seen_during_run: dict[str, Any] = {}

    def fake_run(app_obj: Any, host: str, port: int) -> None:
        seen_during_run["env"] = read(settings)["env"]["ANTHROPIC_BASE_URL"]
        seen_during_run["bind"] = (host, port)

    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = fake_run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    import privacy_kit.gateway.proxy as proxy_module

    monkeypatch.setattr(proxy_module, "build_default_app", lambda: object())

    result = runner.invoke(app, ["serve", "--route", "claude-code", "--port", "9123"])
    assert result.exit_code == 0
    # While the server "ran", Claude Code was routed at the gateway...
    assert seen_during_run["env"] == "http://127.0.0.1:9123"
    assert seen_during_run["bind"] == ("127.0.0.1", 9123)
    # ...we said which file we overrode...
    assert str(settings) in result.stdout
    # ...and on shutdown the previous value came back.
    assert read(settings)["env"]["ANTHROPIC_BASE_URL"] == "http://old:1"
    assert "Restored ANTHROPIC_BASE_URL" in result.stdout


def test_serve_route_rejects_unsupported_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(app, ["serve", "--route", "cursor"])
    assert result.exit_code == 1
