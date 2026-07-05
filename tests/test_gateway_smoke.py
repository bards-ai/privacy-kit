"""Gateway smoke tests: settings defaults and the console-script shim."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydantic_settings")

from privacy_kit.gateway.config import get_settings


def test_settings_defaults() -> None:
    settings = get_settings()
    assert settings.model_id == "bardsai/eu-pii-anonimization-multilang"
    assert 0.0 <= settings.threshold <= 1.0
    assert settings.host == "127.0.0.1"
    assert settings.db_path == Path.home() / ".privacy-kit" / "audit.sqlite"
    # Log-only is the default: detect and log PII but forward prompts unchanged.
    assert settings.policy == "monitor"


def test_settings_share_pii_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    # The gateway reads the same PII_* knobs as the core library.
    monkeypatch.setenv("PII_THRESHOLD", "0.7")
    monkeypatch.setenv("PII_PORT", "9000")
    settings = get_settings()
    assert settings.threshold == 0.7
    assert settings.port == 9000


def test_policy_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # Opt into enforcement (placeholder substitution) via PII_POLICY.
    monkeypatch.setenv("PII_POLICY", "pseudonymize")
    assert get_settings().policy == "pseudonymize"


def test_console_script_entry_point_imports() -> None:
    from privacy_kit.cli import main

    assert callable(main)
