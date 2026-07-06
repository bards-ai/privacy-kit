"""run_import tests: end-to-end writes, idempotency, save_texts policies, dry
runs, per-session exclusion, and placeholder consistency across turns."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sqlmodel")

from importer_helpers import (
    CODEX_ID,
    SESSION_ID,
    make_detector,
    write_claude_session,
    write_codex_session,
)
from sqlmodel import Session, col, select

from privacy_kit.gateway.config import Settings
from privacy_kit.gateway.importer import ImportJob, run_import
from privacy_kit.gateway.store import AuditStore, Interaction


@pytest.fixture
def store(tmp_path: Path) -> AuditStore:
    return AuditStore(tmp_path / "audit.sqlite")


def do_import(
    store: AuditStore, tmp_path: Path, settings: Settings | None = None, **kw: Any
) -> ImportJob:
    """run_import over the session roots the write_* helpers populate."""
    return run_import(
        store,
        make_detector(),
        settings=settings or Settings(save_texts="all"),
        claude_root=tmp_path / "claude",
        codex_root=tmp_path / "codex",
        **kw,
    )


def interactions(store: AuditStore) -> list[Interaction]:
    with Session(store.engine) as session:
        return list(session.exec(select(Interaction).order_by(col(Interaction.created_at))).all())


def test_run_import_end_to_end(tmp_path: Path, store: AuditStore) -> None:
    write_claude_session(tmp_path / "claude")
    write_codex_session(tmp_path / "codex")

    job = do_import(store, tmp_path)
    assert job.state == "done"
    assert (job.found, job.imported, job.skipped, job.failed) == (2, 2, 0, 0)
    assert job.turns == 3

    rows = interactions(store)
    assert len(rows) == 3
    by_conv: dict[str | None, list[Interaction]] = {}
    for row in rows:
        by_conv.setdefault(row.conversation_id, []).append(row)
    assert set(by_conv) == {SESSION_ID, CODEX_ID}

    cc1, cc2 = by_conv[SESSION_ID]
    assert cc1.created_at.replace(tzinfo=timezone.utc) == datetime(
        2026, 7, 1, 9, 1, tzinfo=timezone.utc
    )
    assert cc1.entity_counts == {"EMAIL_ADDRESS": 1}  # distinct values, not occurrences
    assert cc1.policy == "imported"
    assert cc2.entity_counts == {}

    # placeholder consistency across the conversation's turns
    assert cc1.id is not None
    texts = store.texts(cc1.id)
    assert all("[EMAIL_ADDRESS_1]" in t.anonymized for t in texts)
    assert [t.category for t in texts] == ["user", "tool", "assistant"]


def test_run_import_is_idempotent(tmp_path: Path, store: AuditStore) -> None:
    write_claude_session(tmp_path / "claude")

    first = do_import(store, tmp_path)
    assert first.imported == 1
    second = do_import(store, tmp_path)
    assert second.imported == 0
    assert second.skipped == 1
    assert len(interactions(store)) == 2  # still just the first run's two turns


def test_run_import_save_texts_anonymized_keeps_only_pii_segments(
    tmp_path: Path, store: AuditStore
) -> None:
    write_claude_session(tmp_path / "claude")

    do_import(store, tmp_path, settings=Settings(save_texts="anonymized"))
    pii_turn, clean_turn = interactions(store)
    assert pii_turn.id is not None and clean_turn.id is not None
    assert len(store.texts(pii_turn.id)) == 3  # all three segments contained the email
    assert store.texts(clean_turn.id) == []


def test_run_import_dry_run_writes_nothing(tmp_path: Path, store: AuditStore) -> None:
    write_claude_session(tmp_path / "claude")

    job = do_import(store, tmp_path, dry_run=True)
    assert job.imported == 1
    assert job.turns == 2
    assert interactions(store) == []


def test_run_import_excludes_selected_sessions(tmp_path: Path, store: AuditStore) -> None:
    write_claude_session(tmp_path / "claude")
    write_codex_session(tmp_path / "codex")

    job = do_import(store, tmp_path, exclude_session_ids={SESSION_ID})
    assert job.state == "done"
    assert (job.found, job.imported, job.skipped) == (1, 1, 0)
    assert {row.conversation_id for row in interactions(store)} == {CODEX_ID}
