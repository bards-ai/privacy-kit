"""Dashboard JSON API (`/api/v1`) tests. No network, no model: stub detector +
temp store."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from privacy_kit.core.types import Span
from privacy_kit.gateway.config import Settings
from privacy_kit.gateway.proxy import create_app
from privacy_kit.gateway.store import AuditStore


class LiteralDetector:
    def __init__(self, mapping: dict[str, str], threshold: float = 0.5) -> None:
        self.mapping = mapping
        self.threshold = threshold

    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for literal, etype in self.mapping.items():
            start = 0
            while (i := text.find(literal, start)) >= 0:
                spans.append(Span(i, i + len(literal), etype, 0.99))
                start = i + len(literal)
        return spans


PII = {"John Smith": "PERSON_NAME", "john@x.com": "EMAIL_ADDRESS"}


def build(tmp_path: Path, settings: Settings | None = None) -> tuple[TestClient, AuditStore]:
    store = AuditStore(tmp_path / "audit.sqlite")
    app = create_app(
        detector=LiteralDetector(PII),
        store=store,
        settings=settings or Settings(_env_file=None),
    )
    return TestClient(app), store


def seed(store: AuditStore) -> None:
    store.record(
        source="claude-code",
        wire_format="anthropic",
        model="claude-opus-4-8",
        policy="monitor",
        entity_counts={"PERSON_NAME": 2, "EMAIL_ADDRESS": 1},
        input_tokens=10,
        output_tokens=20,
        texts=[("I'm John Smith, john@x.com", "I'm [PERSON_NAME_1], [EMAIL_ADDRESS_1]")],
    )
    store.record(
        source="codex",
        wire_format="openai_responses",
        model="gpt-5",
        policy="pseudonymize",
        entity_counts={"EMAIL_ADDRESS": 1},
        input_tokens=5,
        output_tokens=7,
    )


def _claude_id(client: TestClient) -> int:
    listing = client.get("/api/v1/interactions", params={"source": "claude-code"}).json()
    return int(listing["items"][0]["id"])


def test_summary_has_breakdowns_and_timeseries(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    seed(store)
    data = client.get("/api/v1/summary").json()
    assert data["interactions"] == 2
    assert data["entities_total"] == 4
    assert data["entities_by_type"] == {"PERSON_NAME": 2, "EMAIL_ADDRESS": 2}
    assert data["by_source"] == {"claude-code": 1, "codex": 1}
    assert data["by_policy"] == {"monitor": 1, "pseudonymize": 1}
    assert data["tokens"] == {"input": 15, "output": 27}
    assert len(data["timeseries"]) >= 1
    assert sum(p["interactions"] for p in data["timeseries"]) == 2


def seed_dated(store: AuditStore) -> None:
    store.record(
        source="claude-code",
        wire_format="anthropic",
        model="claude-opus-4-8",
        policy="monitor",
        entity_counts={"PERSON_NAME": 2, "EMAIL_ADDRESS": 1},
        input_tokens=10,
        output_tokens=20,
        created_at=datetime(2026, 1, 5, 12, 0),
    )
    store.record(
        source="codex",
        wire_format="openai_responses",
        model="gpt-5",
        policy="pseudonymize",
        entity_counts={"EMAIL_ADDRESS": 1},
        input_tokens=5,
        output_tokens=7,
        created_at=datetime(2026, 3, 10, 12, 0),
    )


def test_summary_respects_date_range(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    seed_dated(store)
    data = client.get(
        "/api/v1/summary", params={"date_from": "2026-03-01", "date_to": "2026-03-31"}
    ).json()
    assert data["interactions"] == 1
    assert data["by_source"] == {"codex": 1}
    assert data["entities_total"] == 1
    assert data["entities_by_type"] == {"EMAIL_ADDRESS": 1}
    assert data["tokens"] == {"input": 5, "output": 7}
    assert [p["date"] for p in data["timeseries"]] == ["2026-03-10"]


def test_summary_without_dates_and_invalid_dates_returns_all_time(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    seed_dated(store)
    all_time = client.get("/api/v1/summary").json()
    assert all_time["interactions"] == 2
    assert all_time["tokens"] == {"input": 15, "output": 27}
    assert client.get("/api/v1/summary", params={"date_from": "garbage"}).json() == all_time


def test_filters_lists_distinct_values(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    seed(store)
    data = client.get("/api/v1/filters").json()
    assert data["sources"] == ["claude-code", "codex"]
    assert set(data["models"]) == {"claude-opus-4-8", "gpt-5"}
    assert "PERSON_NAME" in data["entity_types"]
    assert set(data["policies"]) == {"monitor", "pseudonymize"}


def test_interactions_list_returns_every_column(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    seed(store)
    data = client.get("/api/v1/interactions").json()
    assert data["total"] == 2
    assert data["page"] == 1
    item = data["items"][0]
    for key in (
        "id",
        "created_at",
        "source",
        "wire_format",
        "model",
        "policy",
        "language",
        "input_tokens",
        "output_tokens",
        "entity_total",
        "entity_counts",
        "text_count",
        "detection_types",
    ):
        assert key in item, key


def test_interactions_filter_by_source(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    seed(store)
    data = client.get("/api/v1/interactions", params={"source": "codex"}).json()
    assert data["total"] == 1
    assert data["items"][0]["source"] == "codex"


def test_interactions_q_search_matches_saved_text(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    seed(store)
    data = client.get("/api/v1/interactions", params={"q": "John Smith"}).json()
    assert data["total"] == 1
    assert data["items"][0]["source"] == "claude-code"
    assert data["items"][0]["text_count"] == 1


def test_interactions_filter_by_entity_type(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    seed(store)
    data = client.get("/api/v1/interactions", params={"entity_type": "PERSON_NAME"}).json()
    assert data["total"] == 1
    assert data["items"][0]["source"] == "claude-code"


def test_interactions_pagination(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    seed(store)
    data = client.get("/api/v1/interactions", params={"page_size": 1}).json()
    assert data["total"] == 2
    assert data["total_pages"] == 2
    assert len(data["items"]) == 1


def test_interaction_detail_includes_detections_and_texts(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    seed(store)
    data = client.get(f"/api/v1/interactions/{_claude_id(client)}").json()
    assert data["interaction"]["source"] == "claude-code"
    assert data["interaction"]["policy"] == "monitor"
    assert {d["entity_type"] for d in data["detections"]} == {"PERSON_NAME", "EMAIL_ADDRESS"}
    assert data["texts_redacted"] is False
    assert data["texts"][0]["original"] == "I'm John Smith, john@x.com"
    assert "[PERSON_NAME_1]" in data["texts"][0]["anonymized"]


def test_interaction_detail_404(tmp_path: Path) -> None:
    client, _ = build(tmp_path)
    assert client.get("/api/v1/interactions/999").status_code == 404


def test_plaintext_is_redacted_when_disabled(tmp_path: Path) -> None:
    client, store = build(tmp_path, settings=Settings(_env_file=None, expose_plaintext=False))
    seed(store)
    data = client.get(f"/api/v1/interactions/{_claude_id(client)}").json()
    assert data["texts_redacted"] is True
    assert data["texts"][0]["original"] is None
    # The anonymized (placeholder) text is safe to show even when redacting.
    assert "[PERSON_NAME_1]" in data["texts"][0]["anonymized"]


def seed_conversation(store: AuditStore) -> None:
    """Two grouped turns under conv-x plus one ungrouped interaction."""
    store.record(
        source="claude-code",
        wire_format="anthropic",
        model="claude-opus-4-8",
        entity_counts={"PERSON_NAME": 1},
        input_tokens=10,
        output_tokens=20,
        conversation_id="conv-x",
        texts=[
            ("I'm John Smith", "I'm [PERSON_NAME_1]", "user"),
            ("Hi John Smith!", "Hi [PERSON_NAME_1]!", "assistant"),
        ],
    )
    store.record(
        source="claude-code",
        wire_format="anthropic",
        model="gpt-5-mini",
        entity_counts={"EMAIL_ADDRESS": 1},
        input_tokens=5,
        output_tokens=7,
        conversation_id="conv-x",
    )
    store.record(
        source="cursor",
        wire_format="openai_chat",
        model="gpt-5",
        entity_counts={"PHONE_NUMBER": 1},
    )


def test_conversations_list_groups_turns(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    seed_conversation(store)
    data = client.get("/api/v1/conversations").json()
    # Only the grouped conversation appears; the ungrouped row is excluded.
    assert data["total"] == 1
    item = data["items"][0]
    assert item["conversation_id"] == "conv-x"
    assert item["turn_count"] == 2
    assert item["entity_total"] == 2
    assert item["entity_counts"] == {"PERSON_NAME": 1, "EMAIL_ADDRESS": 1}


def test_conversation_detail_returns_ordered_turns(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    seed_conversation(store)
    data = client.get("/api/v1/conversations/conv-x").json()
    assert data["conversation_id"] == "conv-x"
    assert len(data["turns"]) == 2
    assert data["texts_redacted"] is False
    # First turn carries the saved prompt + agent response and its detection.
    first = data["turns"][0]
    cats = {t["category"]: t for t in first["texts"]}
    assert cats["user"]["original"] == "I'm John Smith"
    assert cats["assistant"]["original"] == "Hi John Smith!"
    assert {d["entity_type"] for d in first["detections"]} == {"PERSON_NAME"}


def test_conversation_detail_includes_summary(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    seed_conversation(store)
    summary = client.get("/api/v1/conversations/conv-x").json()["summary"]
    assert summary["turn_count"] == 2
    assert summary["entity_total"] == 2
    assert summary["entity_counts"] == {"PERSON_NAME": 1, "EMAIL_ADDRESS": 1}
    assert summary["sources"] == ["claude-code"]
    assert summary["models"] == ["claude-opus-4-8", "gpt-5-mini"]
    assert summary["input_tokens"] == 15  # 10 + 5
    assert summary["output_tokens"] == 27  # 20 + 7


def test_conversation_detail_redacts_plaintext_when_disabled(tmp_path: Path) -> None:
    client, store = build(tmp_path, settings=Settings(_env_file=None, expose_plaintext=False))
    seed_conversation(store)
    data = client.get("/api/v1/conversations/conv-x").json()
    assert data["texts_redacted"] is True
    # Both the prompt and the agent response have their originals redacted.
    for seg in data["turns"][0]["texts"]:
        assert seg["original"] is None
        assert "[PERSON_NAME_1]" in seg["anonymized"]


def test_conversation_detail_404(tmp_path: Path) -> None:
    client, _ = build(tmp_path)
    assert client.get("/api/v1/conversations/nope").status_code == 404


def test_delete_interaction(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    seed(store)
    iid = _claude_id(client)
    assert client.delete(f"/api/v1/interactions/{iid}").status_code == 200
    assert client.get("/api/v1/summary").json()["interactions"] == 1
    # Children are gone too — no orphans left behind.
    assert store.detections(iid) == []
    assert store.texts(iid) == []
    assert client.delete(f"/api/v1/interactions/{iid}").status_code == 404


def test_clear_requires_confirmation(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    seed(store)
    assert client.post("/api/v1/audit/clear", json={}).status_code == 400
    resp = client.post("/api/v1/audit/clear", json={"confirm": True})
    assert resp.status_code == 200
    assert resp.json()["cleared"] == 2
    assert client.get("/api/v1/summary").json()["interactions"] == 0


def test_preview_runs_and_is_not_persisted(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    resp = client.post("/api/v1/preview", json={"text": "Reach John Smith at john@x.com."})
    data = resp.json()
    assert data["counts"] == {"PERSON_NAME": 1, "EMAIL_ADDRESS": 1}
    assert data["anonymized"] == "Reach [PERSON_NAME_1] at [EMAIL_ADDRESS_1]."
    assert store.summary()["interactions"] == 0
    assert client.post("/api/v1/preview", json={"text": "x" * 50_001}).status_code == 413


def test_export_csv_has_header_and_attachment(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    seed(store)
    resp = client.get("/api/v1/export", params={"format": "csv"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert "filename=privacy-kit-interactions.csv" in resp.headers["content-disposition"]
    lines = resp.text.splitlines()
    assert lines[0].startswith("id,created_at,source,wire_format,kind,model,policy")
    assert "claude-code" in resp.text and "codex" in resp.text


def test_export_json_with_texts_respects_filter(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    seed(store)
    resp = client.get(
        "/api/v1/export",
        params={"format": "json", "include_texts": "true", "source": "claude-code"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["interactions"]) == 1
    item = data["interactions"][0]
    assert item["source"] == "claude-code"
    assert {d["entity_type"] for d in item["detections"]} == {"PERSON_NAME", "EMAIL_ADDRESS"}
    assert item["texts"][0]["original"] == "I'm John Smith, john@x.com"


def test_config_endpoint_exposes_runtime_settings(tmp_path: Path) -> None:
    client, _ = build(tmp_path)
    data = client.get("/api/v1/config").json()
    assert data["policy"] in {"monitor", "pseudonymize"}
    assert data["expose_plaintext"] is True
    assert "version" in data
    assert "model_id" in data


def test_config_patch_updates_policy(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, policy="monitor")
    store = AuditStore(tmp_path / "audit.sqlite")
    client = TestClient(create_app(detector=LiteralDetector(PII), store=store, settings=settings))

    res = client.patch("/api/v1/config", json={"policy": "pseudonymize"})
    assert res.status_code == 200
    assert res.json()["policy"] == "pseudonymize"
    # The same settings object the proxy reads per request is mutated, so the
    # change takes effect for subsequent traffic.
    assert settings.policy == "pseudonymize"
    assert client.get("/api/v1/config").json()["policy"] == "pseudonymize"


def test_config_patch_updates_save_texts(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, save_texts="all")
    store = AuditStore(tmp_path / "audit.sqlite")
    client = TestClient(create_app(detector=LiteralDetector(PII), store=store, settings=settings))

    res = client.patch("/api/v1/config", json={"save_texts": "anonymized"})
    assert res.status_code == 200
    assert res.json()["save_texts"] == "anonymized"
    assert settings.save_texts == "anonymized"


def test_config_patch_updates_threshold_on_settings_and_detector(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, threshold=0.5)
    detector = LiteralDetector(PII)
    store = AuditStore(tmp_path / "audit.sqlite")
    client = TestClient(create_app(detector=detector, store=store, settings=settings))

    res = client.patch("/api/v1/config", json={"threshold": 0.8})
    assert res.status_code == 200
    assert res.json()["threshold"] == 0.8
    assert settings.threshold == 0.8
    # The live detector is updated too, since threshold is read per detection.
    assert detector.threshold == 0.8


def test_config_patch_updates_multiple_settings_atomically(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, policy="monitor", save_texts="all")
    store = AuditStore(tmp_path / "audit.sqlite")
    client = TestClient(create_app(detector=LiteralDetector(PII), store=store, settings=settings))

    res = client.patch(
        "/api/v1/config", json={"policy": "pseudonymize", "save_texts": "anonymized"}
    )
    assert res.status_code == 200
    assert (settings.policy, settings.save_texts) == ("pseudonymize", "anonymized")


def test_config_patch_rejects_bad_values(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, policy="monitor")
    store = AuditStore(tmp_path / "audit.sqlite")
    client = TestClient(create_app(detector=LiteralDetector(PII), store=store, settings=settings))

    assert client.patch("/api/v1/config", json={"policy": "delete-everything"}).status_code == 400
    assert client.patch("/api/v1/config", json={"save_texts": "everything"}).status_code == 400
    assert client.patch("/api/v1/config", json={"threshold": 1.5}).status_code == 400
    assert client.patch("/api/v1/config", json={"threshold": "high"}).status_code == 400
    assert client.patch("/api/v1/config", json={"threshold": True}).status_code == 400
    # Unknown / read-only settings are rejected, and nothing is mutated.
    assert client.patch("/api/v1/config", json={"db_path": "/etc/passwd"}).status_code == 400
    assert client.patch("/api/v1/config", json={}).status_code == 400
    assert settings.policy == "monitor"


def test_texts_browser_lists_segments(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    seed(store)
    data = client.get("/api/v1/texts").json()
    assert data["redacted"] is False
    assert len(data["texts"]) == 1
    seg = data["texts"][0]
    assert seg["source"] == "claude-code"
    assert seg["original"] == "I'm John Smith, john@x.com"
    assert "[PERSON_NAME_1]" in seg["anonymized"]
    assert seg["category"] == "user"


def test_texts_browser_filters_by_category(tmp_path: Path) -> None:
    client, store = build(tmp_path)
    store.record(
        source="claude-code",
        wire_format="anthropic",
        model="m",
        entity_counts={"PERSON_NAME": 2},
        texts=[
            ("typed: Jane Doe", "typed: [PERSON_NAME_1]", "user"),
            ("file owner: John Smith", "file owner: [PERSON_NAME_2]", "tool"),
        ],
    )
    all_segs = client.get("/api/v1/texts").json()["texts"]
    assert {s["category"] for s in all_segs} == {"user", "tool"}

    tool_only = client.get("/api/v1/texts", params={"category": "tool"}).json()["texts"]
    assert [s["category"] for s in tool_only] == ["tool"]
    assert "John Smith" in tool_only[0]["original"]


def test_texts_browser_redacts_when_disabled(tmp_path: Path) -> None:
    client, store = build(tmp_path, settings=Settings(_env_file=None, expose_plaintext=False))
    seed(store)
    data = client.get("/api/v1/texts").json()
    assert data["redacted"] is True
    assert data["texts"][0]["original"] is None
