"""Codex routing: editing ~/.codex/config.toml via route.py and the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tomlkit")
pytest.importorskip("typer")

import tomlkit
from typer.testing import CliRunner

from privacy_kit.gateway import route
from privacy_kit.gateway.cli import app

runner = CliRunner()

BASE = "http://127.0.0.1:8787"


def test_apply_creates_file_with_openai_base_url(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    change = route.apply_codex_route(BASE, cfg)

    doc = tomlkit.parse(cfg.read_text(encoding="utf-8"))
    assert doc["openai_base_url"] == f"{BASE}/v1"
    assert "chatgpt_base_url" not in doc  # one key routes both auth modes
    assert change.applied == {"openai_base_url": f"{BASE}/v1"}
    assert change.previous == {"openai_base_url": None}


def test_apply_inserts_into_root_table_not_after_a_trailing_table(tmp_path: Path) -> None:
    # Regression: appending a bare key after a [table] makes TOML read it as a
    # member of that table, which breaks Codex's config load (expected u32 ...).
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        'model = "gpt-5.4-mini"\n\n[tui.model_availability_nux]\n"gpt-5.5" = 4\n',
        encoding="utf-8",
    )
    route.apply_codex_route(BASE, cfg)

    doc = tomlkit.parse(cfg.read_text(encoding="utf-8"))
    # Parsed at the document root, NOT under [tui.model_availability_nux].
    assert doc["openai_base_url"] == f"{BASE}/v1"
    assert "openai_base_url" not in doc["tui"]["model_availability_nux"]
    assert doc["tui"]["model_availability_nux"]["gpt-5.5"] == 4


def test_apply_preserves_comments_and_unrelated_keys(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '# my codex tweaks\nmodel = "gpt-5-codex"\n',
        encoding="utf-8",
    )
    route.apply_codex_route(BASE, cfg)

    text = cfg.read_text(encoding="utf-8")
    assert "# my codex tweaks" in text
    doc = tomlkit.parse(text)
    assert doc["model"] == "gpt-5-codex"
    assert doc["openai_base_url"] == f"{BASE}/v1"


def test_apply_updates_existing_openai_base_url_in_place(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('openai_base_url = "https://old.test/v1"\nmodel = "gpt-5"\n', encoding="utf-8")
    change = route.apply_codex_route(BASE, cfg)

    doc = tomlkit.parse(cfg.read_text(encoding="utf-8"))
    assert doc["openai_base_url"] == f"{BASE}/v1"
    assert change.previous == {"openai_base_url": "https://old.test/v1"}


def test_apply_drops_stale_gateway_chatgpt_base_url(tmp_path: Path) -> None:
    # An earlier version set chatgpt_base_url to the gateway; clean it up.
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'chatgpt_base_url = "{BASE}/"\n', encoding="utf-8")
    change = route.apply_codex_route(BASE, cfg)

    doc = tomlkit.parse(cfg.read_text(encoding="utf-8"))
    assert "chatgpt_base_url" not in doc
    assert change.removed == {"chatgpt_base_url": f"{BASE}/"}


def test_apply_keeps_users_custom_chatgpt_base_url(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('chatgpt_base_url = "https://example.test/"\n', encoding="utf-8")
    change = route.apply_codex_route(BASE, cfg)

    doc = tomlkit.parse(cfg.read_text(encoding="utf-8"))
    assert doc["chatgpt_base_url"] == "https://example.test/"  # not a gateway value: hands off
    assert change.removed == {}


def test_apply_rejects_invalid_toml(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text("model = [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError):
        route.apply_codex_route(BASE, cfg)


def test_remove_deletes_gateway_keys_only(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('model = "gpt-5-codex"\n', encoding="utf-8")
    route.apply_codex_route(BASE, cfg)

    removed = route.remove_codex_route(cfg)
    assert removed == {"openai_base_url": f"{BASE}/v1"}

    doc = tomlkit.parse(cfg.read_text(encoding="utf-8"))
    assert doc["model"] == "gpt-5-codex"
    assert "openai_base_url" not in doc

    assert route.remove_codex_route(cfg) == {}  # idempotent
    assert route.remove_codex_route(tmp_path / "absent.toml") == {}


def test_cli_setup_codex_apply_and_remove(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"

    result = runner.invoke(
        app,
        ["setup", "codex", "--apply", "--port", "9999", "--settings-file", str(cfg)],
    )
    assert result.exit_code == 0
    assert "openai_base_url=http://127.0.0.1:9999/v1" in result.stdout
    doc = tomlkit.parse(cfg.read_text(encoding="utf-8"))
    assert doc["openai_base_url"] == "http://127.0.0.1:9999/v1"

    result = runner.invoke(app, ["setup", "codex", "--remove", "--settings-file", str(cfg)])
    assert result.exit_code == 0
    text = cfg.read_text(encoding="utf-8")
    assert "openai_base_url" not in text
