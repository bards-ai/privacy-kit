"""Codex routing: editing ~/.codex/config.toml via route.py and the CLI.

We route Codex through a dedicated ``[model_providers.privacy-kit]`` entry with
``supports_websockets = false`` so Codex 0.139+ never attempts the Responses
WebSocket transport (which the gateway does not speak and which would otherwise
print a "Falling back from WebSockets to HTTPS" warning every session).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("tomlkit")
pytest.importorskip("typer")

import tomlkit
from typer.testing import CliRunner

from privacy_kit.gateway import route
from privacy_kit.gateway.cli import app

runner = CliRunner()

BASE = "http://127.0.0.1:8787"


def _provider(doc: Any) -> Any:
    return doc["model_providers"]["privacy-kit"]


def test_apply_creates_provider_with_websockets_disabled(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    change = route.apply_codex_route(BASE, cfg)

    doc = tomlkit.parse(cfg.read_text(encoding="utf-8"))
    assert doc["model_provider"] == "privacy-kit"
    provider = _provider(doc)
    assert provider["base_url"] == f"{BASE}/v1"
    assert provider["wire_api"] == "responses"
    assert provider["requires_openai_auth"] is True
    assert provider["supports_websockets"] is False
    assert "openai_base_url" not in doc  # no bare key that would re-enable WS
    assert "chatgpt_base_url" not in doc

    assert change.applied == {
        "model_provider": "privacy-kit",
        "model_providers.privacy-kit.base_url": f"{BASE}/v1",
    }
    assert change.previous == {
        "model_provider": None,
        "model_providers.privacy-kit.base_url": None,
    }


def test_apply_inserts_model_provider_into_root_not_after_a_trailing_table(tmp_path: Path) -> None:
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
    assert doc["model_provider"] == "privacy-kit"
    assert "model_provider" not in doc["tui"]["model_availability_nux"]
    assert doc["tui"]["model_availability_nux"]["gpt-5.5"] == 4
    assert _provider(doc)["supports_websockets"] is False


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
    assert doc["model_provider"] == "privacy-kit"
    assert _provider(doc)["base_url"] == f"{BASE}/v1"


def test_apply_keeps_sibling_providers(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[model_providers.ollama]\nname = "ollama"\nbase_url = "http://localhost:11434/v1"\n',
        encoding="utf-8",
    )
    route.apply_codex_route(BASE, cfg)

    doc = tomlkit.parse(cfg.read_text(encoding="utf-8"))
    assert doc["model_providers"]["ollama"]["base_url"] == "http://localhost:11434/v1"
    assert _provider(doc)["supports_websockets"] is False


def test_apply_updates_existing_provider_in_place(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    route.apply_codex_route("http://127.0.0.1:9000", cfg)  # an earlier run
    change = route.apply_codex_route(BASE, cfg)

    doc = tomlkit.parse(cfg.read_text(encoding="utf-8"))
    assert _provider(doc)["base_url"] == f"{BASE}/v1"
    assert change.previous["model_providers.privacy-kit.base_url"] == "http://127.0.0.1:9000/v1"
    assert change.previous["model_provider"] == "privacy-kit"


def test_apply_drops_stale_gateway_openai_base_url(tmp_path: Path) -> None:
    # An earlier version routed via a bare openai_base_url; clean it up so it
    # can't re-enable the WebSocket transport on the built-in openai provider.
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'openai_base_url = "{BASE}/v1"\n', encoding="utf-8")
    change = route.apply_codex_route(BASE, cfg)

    doc = tomlkit.parse(cfg.read_text(encoding="utf-8"))
    assert "openai_base_url" not in doc
    assert change.removed == {"openai_base_url": f"{BASE}/v1"}


def test_apply_keeps_users_custom_openai_base_url(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('openai_base_url = "https://example.test/v1"\n', encoding="utf-8")
    change = route.apply_codex_route(BASE, cfg)

    doc = tomlkit.parse(cfg.read_text(encoding="utf-8"))
    assert doc["openai_base_url"] == "https://example.test/v1"  # not a gateway value
    assert change.removed == {}


def test_apply_drops_stale_gateway_chatgpt_base_url(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'chatgpt_base_url = "{BASE}/"\n', encoding="utf-8")
    change = route.apply_codex_route(BASE, cfg)

    doc = tomlkit.parse(cfg.read_text(encoding="utf-8"))
    assert "chatgpt_base_url" not in doc
    assert change.removed == {"chatgpt_base_url": f"{BASE}/"}


def test_apply_rejects_invalid_toml(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text("model = [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError):
        route.apply_codex_route(BASE, cfg)


def test_remove_deletes_only_what_apply_writes(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('model = "gpt-5-codex"\n', encoding="utf-8")
    route.apply_codex_route(BASE, cfg)

    removed = route.remove_codex_route(cfg)
    assert removed == {
        "model_provider": "privacy-kit",
        "model_providers.privacy-kit": f"{BASE}/v1",
    }

    doc = tomlkit.parse(cfg.read_text(encoding="utf-8"))
    assert doc["model"] == "gpt-5-codex"
    assert "model_provider" not in doc
    assert "model_providers" not in doc

    assert route.remove_codex_route(cfg) == {}  # idempotent
    assert route.remove_codex_route(tmp_path / "absent.toml") == {}


def test_remove_leaves_sibling_providers_and_user_model_provider_alone(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[model_providers.ollama]\nname = "ollama"\nbase_url = "http://localhost:11434/v1"\n',
        encoding="utf-8",
    )
    route.apply_codex_route(BASE, cfg)
    route.remove_codex_route(cfg)

    doc = tomlkit.parse(cfg.read_text(encoding="utf-8"))
    assert doc["model_providers"]["ollama"]["base_url"] == "http://localhost:11434/v1"
    assert "privacy-kit" not in doc["model_providers"]


def test_remove_leaves_a_users_own_model_provider_alone(tmp_path: Path) -> None:
    # remove must not delete a model_provider the user repointed elsewhere.
    cfg = tmp_path / "config.toml"
    route.apply_codex_route(BASE, cfg)
    doc = tomlkit.parse(cfg.read_text(encoding="utf-8"))
    doc["model_provider"] = "openai"
    cfg.write_text(tomlkit.dumps(doc), encoding="utf-8")

    removed = route.remove_codex_route(cfg)
    assert "model_provider" not in removed  # not pointing at us: hands off
    assert doc["model_provider"] == "openai"


def test_cli_setup_codex_apply_and_remove(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"

    result = runner.invoke(
        app,
        ["setup", "codex", "--apply", "--port", "9999", "--settings-file", str(cfg)],
    )
    assert result.exit_code == 0
    assert "model_provider=privacy-kit" in result.stdout
    doc = tomlkit.parse(cfg.read_text(encoding="utf-8"))
    assert doc["model_provider"] == "privacy-kit"
    assert doc["model_providers"]["privacy-kit"]["supports_websockets"] is False

    result = runner.invoke(app, ["setup", "codex", "--remove", "--settings-file", str(cfg)])
    assert result.exit_code == 0
    text = cfg.read_text(encoding="utf-8")
    assert "model_provider" not in text
    assert "privacy-kit" not in text
