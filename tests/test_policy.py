"""Per-entity policy tests: resolver, apply_policy, and proxy enforcement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from privacy_kit.core.types import Span
from privacy_kit.core.vault import Vault
from privacy_kit.gateway.config import Settings
from privacy_kit.gateway.policy import PolicyResolver, apply_policy
from privacy_kit.gateway.proxy import ForwardResult, create_app
from privacy_kit.gateway.store import AuditStore


class LiteralDetector:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for literal, etype in self.mapping.items():
            start = 0
            while (i := text.find(literal, start)) >= 0:
                spans.append(Span(i, i + len(literal), etype, 0.99))
                start = i + len(literal)
        return spans


# --- Resolver ---


def test_resolver_default_comes_from_global_mode() -> None:
    monitor = PolicyResolver.from_settings(Settings(_env_file=None, policy="monitor"))
    assert monitor.action_for("PERSON_NAME") == "keep"
    enforce = PolicyResolver.from_settings(Settings(_env_file=None, policy="pseudonymize"))
    assert enforce.action_for("PERSON_NAME") == "pseudonymize"


def test_resolver_exact_beats_wildcard_and_longest_prefix_wins() -> None:
    resolver = PolicyResolver(
        "keep",
        {
            "SECRET_*": "block",
            "SECRET_JWT": "redact",  # exact beats the wildcard
            "SECRET_AWS_*": "pseudonymize",  # longer prefix beats shorter
        },
    )
    assert resolver.action_for("SECRET_JWT") == "redact"
    assert resolver.action_for("SECRET_AWS_ACCESS_KEY") == "pseudonymize"
    assert resolver.action_for("SECRET_SLACK_TOKEN") == "block"
    assert resolver.action_for("PERSON_NAME") == "keep"


def test_resolver_overrides_apply_in_monitor_mode() -> None:
    settings = Settings(_env_file=None, policy="monitor", policy_overrides={"SECRET_*": "block"})
    resolver = PolicyResolver.from_settings(settings)
    assert resolver.action_for("SECRET_GENERIC") == "block"
    assert resolver.action_for("EMAIL_ADDRESS") == "keep"


def test_policy_overrides_parse_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PII_POLICY_OVERRIDES", '{"SECRET_*": "block", "PERSON_NAME": "redact"}')
    settings = Settings(_env_file=None)
    assert settings.policy_overrides == {"SECRET_*": "block", "PERSON_NAME": "redact"}


# --- apply_policy ---


def test_apply_policy_mixed_actions_in_one_text() -> None:
    detector = LiteralDetector(
        {
            "John Smith": "PERSON_NAME",  # pseudonymize
            "john@x.com": "EMAIL_ADDRESS",  # keep
            "AKIA123": "SECRET_AWS_ACCESS_KEY",  # block
            "topsecretvalue": "SECRET_GENERIC",  # redact
        }
    )
    resolver = PolicyResolver(
        "keep",
        {
            "PERSON_NAME": "pseudonymize",
            "SECRET_AWS_ACCESS_KEY": "block",
            "SECRET_GENERIC": "redact",
        },
    )
    vault = Vault()
    blocked: dict[str, int] = {}
    text = "John Smith john@x.com AKIA123 topsecretvalue"
    out = apply_policy(text, detector, vault, resolver, blocked)

    assert out == "[PERSON_NAME_1] john@x.com AKIA123 [REDACTED]"
    assert blocked == {"SECRET_AWS_ACCESS_KEY": 1}
    # Only the pseudonymized value entered the vault (redact is one-way).
    assert vault.mapping == {"[PERSON_NAME_1]": "John Smith"}


def test_apply_policy_all_keep_returns_text_unchanged() -> None:
    detector = LiteralDetector({"John Smith": "PERSON_NAME"})
    vault = Vault()
    blocked: dict[str, int] = {}
    text = "I'm John Smith."
    assert apply_policy(text, detector, vault, PolicyResolver("keep", {}), blocked) == text
    assert not blocked and len(vault) == 0


# --- Proxy enforcement ---


PII = {
    "John Smith": "PERSON_NAME",
    "john@x.com": "EMAIL_ADDRESS",
    "AKIAIOSFODNN7EXAMPLE": "SECRET_AWS_ACCESS_KEY",
}


class CapturingForwarder:
    def __init__(self) -> None:
        self.calls = 0
        self.last_payload: dict[str, Any] | None = None

    async def __call__(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> ForwardResult:
        self.calls += 1
        self.last_payload = payload
        return ForwardResult(200, {"content": [{"type": "text", "text": "ok [PERSON_NAME_1]"}]}, {})


def build(tmp_path: Path, settings: Settings) -> tuple[TestClient, CapturingForwarder, AuditStore]:
    store = AuditStore(tmp_path / "audit.sqlite")
    forwarder = CapturingForwarder()
    app = create_app(
        detector=LiteralDetector(PII), store=store, forwarder=forwarder, settings=settings
    )
    return TestClient(app), forwarder, store


def post(client: TestClient, content: str) -> Any:
    return client.post(
        "/v1/messages",
        json={"model": "claude-opus-4-8", "messages": [{"role": "user", "content": content}]},
    )


def test_block_override_refuses_request_and_audits(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        policy="monitor",
        save_texts="all",
        policy_overrides={"SECRET_*": "block"},
    )
    client, forwarder, store = build(tmp_path, settings)

    resp = post(client, "deploy with key AKIAIOSFODNN7EXAMPLE for John Smith")
    assert resp.status_code == 403
    error = resp.json()["error"]
    assert error["type"] == "privacy_kit_blocked"
    assert error["entities"] == {"SECRET_AWS_ACCESS_KEY": 1}
    # Nothing reached the upstream.
    assert forwarder.calls == 0

    # The block is on the audit record, with the full detection.
    row = store.recent()[0]
    assert row.policy == "block"
    assert row.entity_counts.get("SECRET_AWS_ACCESS_KEY") == 1
    assert row.entity_counts.get("PERSON_NAME") == 1


def test_requests_without_blocked_types_still_flow(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, policy="monitor", policy_overrides={"SECRET_*": "block"})
    client, forwarder, _ = build(tmp_path, settings)
    resp = post(client, "hello from John Smith")
    assert resp.status_code == 200
    assert forwarder.calls == 1
    # Monitor default: the name went upstream unchanged.
    assert "John Smith" in str(forwarder.last_payload)


def test_monitor_with_pseudonymize_override_mixes_actions(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        policy="monitor",
        policy_overrides={"PERSON_NAME": "pseudonymize"},
    )
    client, forwarder, _ = build(tmp_path, settings)
    resp = post(client, "I'm John Smith, email john@x.com")
    assert resp.status_code == 200

    sent = str(forwarder.last_payload)
    assert "[PERSON_NAME_1]" in sent and "John Smith" not in sent
    assert "john@x.com" in sent  # default keep in monitor mode
    # Pseudonymized values still rehydrate in the response.
    assert "ok John Smith" in str(resp.json())


def test_redact_override_is_one_way(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        policy="pseudonymize",
        policy_overrides={"EMAIL_ADDRESS": "redact"},
    )
    client, forwarder, _ = build(tmp_path, settings)
    resp = post(client, "I'm John Smith, email john@x.com")
    assert resp.status_code == 200

    sent = str(forwarder.last_payload)
    assert "[REDACTED]" in sent and "john@x.com" not in sent
    assert "[PERSON_NAME_1]" in sent  # global pseudonymize still applies
    # The redacted value never entered the vault, so nothing rehydrates it.
    assert "john@x.com" not in str(resp.json())


def test_cursor_hook_denies_on_block_typed_entity(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, policy="monitor", policy_overrides={"SECRET_*": "block"})
    client, _, _ = build(tmp_path, settings)
    resp = client.post(
        "/v1/cursor-hook",
        json={
            "hook_event_name": "beforeReadFile",
            "content": "AWS_KEY=AKIAIOSFODNN7EXAMPLE",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["permission"] == "deny"
    assert "SECRET_AWS_ACCESS_KEY" in body.get("userMessage", str(body))


def test_cursor_hook_allows_non_blocked_pii_without_cursor_block(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, policy="monitor", policy_overrides={"SECRET_*": "block"})
    client, _, _ = build(tmp_path, settings)
    resp = client.post(
        "/v1/cursor-hook",
        json={"hook_event_name": "beforeReadFile", "content": "author: John Smith"},
    )
    assert resp.json().get("permission") == "allow"
