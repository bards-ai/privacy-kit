"""CLI tests. setup/report are model-free; scan needs the real model (env-gated)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("typer")

from typer.testing import CliRunner

from privacy_kit.gateway.cli import app
from privacy_kit.gateway.store import AuditStore

requires_model = pytest.mark.skipif(
    os.getenv("PII_RUN_MODEL_TESTS") != "1",
    reason="set PII_RUN_MODEL_TESTS=1 to download and run the model",
)

runner = CliRunner()


def test_setup_claude_code_prints_base_url() -> None:
    result = runner.invoke(app, ["setup", "claude-code", "--port", "9999"])
    assert result.exit_code == 0
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:9999" in result.stdout
    assert "OTEL_EXPORTER_OTLP_PROTOCOL=http/json" in result.stdout


def test_setup_codex_prints_provider_block() -> None:
    result = runner.invoke(app, ["setup", "codex"])
    assert result.exit_code == 0
    assert 'model_provider = "privacy-kit"' in result.stdout
    assert "supports_websockets = false" in result.stdout
    assert "/v1" in result.stdout


def test_setup_cursor_mentions_chat_only_limitation() -> None:
    result = runner.invoke(app, ["setup", "cursor"])
    assert result.exit_code == 0
    assert "Override OpenAI Base URL" in result.stdout
    assert "Composer" in result.stdout


def test_setup_unknown_tool_exits_nonzero() -> None:
    result = runner.invoke(app, ["setup", "vim"])
    assert result.exit_code == 1


def test_setup_cursor_rejects_bad_scope() -> None:
    result = runner.invoke(app, ["setup", "cursor", "--apply", "--scope", "nope"])
    assert result.exit_code == 1


def test_hook_fails_open_when_gateway_down() -> None:
    # Point at a port nothing is listening on; the hook must allow, not error.
    result = runner.invoke(
        app,
        ["hook", "cursor", "beforeSubmitPrompt"],
        input='{"prompt": "hi"}',
        env={"PII_HOST": "127.0.0.1", "PII_PORT": "9"},
    )
    assert result.exit_code == 0
    assert '"continue": true' in result.stdout


def test_hook_fails_open_for_before_read_file() -> None:
    result = runner.invoke(
        app,
        ["hook", "cursor", "beforeReadFile"],
        input='{"content": "hi"}',
        env={"PII_HOST": "127.0.0.1", "PII_PORT": "9"},
    )
    assert result.exit_code == 0
    assert '"permission": "allow"' in result.stdout


def test_hook_non_cursor_tool_allows_without_network() -> None:
    result = runner.invoke(app, ["hook", "other", "beforeSubmitPrompt"], input="{}")
    assert result.exit_code == 0
    assert '"continue": true' in result.stdout


def test_report_on_empty_db(tmp_path: Path) -> None:
    db = tmp_path / "audit.sqlite"
    AuditStore(db)  # create empty
    result = runner.invoke(app, ["report", "--db", str(db)])
    assert result.exit_code == 0
    assert "Interactions: 0" in result.stdout


def test_report_shows_recorded_entities(tmp_path: Path) -> None:
    db = tmp_path / "audit.sqlite"
    store = AuditStore(db)
    store.record(
        source="claude-code",
        wire_format="anthropic",
        model="claude-opus-4-8",
        entity_counts={"PERSON_NAME": 2, "EMAIL_ADDRESS": 1},
    )
    result = runner.invoke(app, ["report", "--db", str(db)])
    assert result.exit_code == 0
    assert "Interactions: 1" in result.stdout
    assert "PERSON_NAME" in result.stdout
    assert "claude-code" in result.stdout


def _no_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    from privacy_kit.gateway import clients

    monkeypatch.setattr(clients, "list_processes", lambda: [])


def _running(monkeypatch: pytest.MonkeyPatch, by_tool: dict[str, int]) -> list[str]:
    """Fake detection: `by_tool` maps tool name -> fake pid. Returns the call log."""
    from privacy_kit.gateway import clients

    calls: list[str] = []

    def fake_detect(tool: str, procs: object) -> list[clients.ClientProcess]:
        calls.append(tool)
        if tool in by_tool:
            return [clients.ClientProcess(pid=by_tool[tool], exe=tool, args=tool)]
        return []

    monkeypatch.setattr(clients, "list_processes", lambda: [])
    monkeypatch.setattr(clients, "detect", fake_detect)
    return calls


def _forbid_terminate(monkeypatch: pytest.MonkeyPatch) -> None:
    from privacy_kit.gateway import clients

    def boom(pids: list[int], grace: float = 8.0) -> list[int]:
        raise AssertionError("terminate must not be called")

    monkeypatch.setattr(clients, "terminate", boom)


def test_restart_clients_nothing_running(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_clients(monkeypatch)
    result = runner.invoke(app, ["restart-clients"])
    assert result.exit_code == 0
    assert "No running clients detected" in result.stdout


def test_restart_clients_unknown_tool_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_clients(monkeypatch)
    result = runner.invoke(app, ["restart-clients", "vim"])
    assert result.exit_code == 1


def test_restart_clients_defaults_to_all_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _running(monkeypatch, {})
    result = runner.invoke(app, ["restart-clients"])
    assert result.exit_code == 0
    assert calls == ["claude-code", "codex", "cursor"]


def test_restart_clients_warns_only_for_claude_code(monkeypatch: pytest.MonkeyPatch) -> None:
    _running(monkeypatch, {"claude-code": 111})
    _forbid_terminate(monkeypatch)
    result = runner.invoke(app, ["restart-clients", "claude-code"])
    assert result.exit_code == 0
    assert "claude --continue" in result.stdout


def test_restart_clients_cursor_non_interactive_skips_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from privacy_kit.gateway import clients

    _running(monkeypatch, {"cursor": 222})
    _forbid_terminate(monkeypatch)
    monkeypatch.setattr(clients, "stdin_is_interactive", lambda: False)
    result = runner.invoke(app, ["restart-clients", "cursor"])
    assert result.exit_code == 0
    assert "extensions" in result.stdout
    assert "Restart Cursor now?" not in result.stdout
    assert "Restart Cursor manually" in result.stdout


def test_restart_clients_cursor_answer_no_leaves_it_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from privacy_kit.gateway import clients

    _running(monkeypatch, {"cursor": 222})
    _forbid_terminate(monkeypatch)
    monkeypatch.setattr(clients, "stdin_is_interactive", lambda: True)
    result = runner.invoke(app, ["restart-clients", "cursor"], input="n\n")
    assert result.exit_code == 0
    assert "Restart Cursor now?" in result.stdout
    assert "Leaving Cursor running" in result.stdout


def test_restart_clients_cursor_answer_yes_kills_and_relaunches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from privacy_kit.gateway import clients

    _running(monkeypatch, {"cursor": 222})
    monkeypatch.setattr(clients, "stdin_is_interactive", lambda: True)
    killed: list[list[int]] = []
    launched: list[list[str]] = []

    def fake_terminate(pids: list[int], grace: float = 8.0) -> list[int]:
        killed.append(pids)
        return []

    monkeypatch.setattr(clients, "terminate", fake_terminate)
    monkeypatch.setattr(clients, "cursor_relaunch_argv", lambda procs: ["/usr/bin/cursor"])
    monkeypatch.setattr(clients, "relaunch_detached", lambda argv: launched.append(argv))
    result = runner.invoke(app, ["restart-clients", "cursor"], input="y\n")
    assert result.exit_code == 0
    assert killed == [[222]]
    assert launched == [["/usr/bin/cursor"]]
    assert "Restarted Cursor" in result.stdout


@requires_model
def test_scan_with_real_model(tmp_path: Path) -> None:
    sample = tmp_path / "note.txt"
    sample.write_text("Email John Smith at john.smith@example.com.", encoding="utf-8")
    result = runner.invoke(app, ["scan", str(sample)])
    assert result.exit_code == 0
    assert "PII span" in result.stdout
    assert "EMAIL_ADDRESS" in result.stdout or "PERSON_NAME" in result.stdout


@requires_model
def test_scan_anonymize_with_real_model(tmp_path: Path) -> None:
    sample = tmp_path / "note.txt"
    sample.write_text("Email John Smith at john.smith@example.com.", encoding="utf-8")
    result = runner.invoke(app, ["scan", str(sample), "--anonymize"])
    assert result.exit_code == 0
    assert "john.smith@example.com" not in result.stdout
    assert "[" in result.stdout  # contains a placeholder
