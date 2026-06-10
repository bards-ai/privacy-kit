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


def test_setup_codex_prints_openai_base_url() -> None:
    result = runner.invoke(app, ["setup", "codex"])
    assert result.exit_code == 0
    assert "OPENAI_BASE_URL=" in result.stdout
    assert "/v1" in result.stdout


def test_setup_cursor_mentions_chat_only_limitation() -> None:
    result = runner.invoke(app, ["setup", "cursor"])
    assert result.exit_code == 0
    assert "Override OpenAI Base URL" in result.stdout
    assert "Composer" in result.stdout


def test_setup_unknown_tool_exits_nonzero() -> None:
    result = runner.invoke(app, ["setup", "vim"])
    assert result.exit_code == 1


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
