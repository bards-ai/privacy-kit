"""Batch driver: replay parsed history sessions through detection into the store.

Mirrors the live proxy's semantics: one :class:`Vault` per conversation so a
value keeps its placeholder across turns, per-turn entity counts over distinct
(type, value) pairs, and ``Settings.save_texts`` deciding which segments'
plaintext is kept ("anonymized" → only segments the detector changed; "all" →
everything). Detection runs once per segment — spans are applied to the shared
vault directly rather than re-running the model for counting.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Any

from privacy_kit.core.detectors import Detector
from privacy_kit.core.types import Span
from privacy_kit.core.vault import Vault
from privacy_kit.gateway.config import Settings, get_settings
from privacy_kit.gateway.importer import claude_code, codex
from privacy_kit.gateway.importer.base import ParsedSession
from privacy_kit.gateway.store.audit import AuditStore

logger = logging.getLogger("privacy_kit.importer")

SOURCES = ("claude-code", "codex")

# Segments beyond this are truncated before detection: transcripts can embed
# entire files, and unbounded inputs would stall the batch on a single blob.
MAX_SEGMENT_CHARS = 100_000


@dataclass
class ImportJob:
    """Mutable status of one import run; safe to serialize at any point."""

    state: str = "idle"  # idle | running | done | error
    sources: list[str] = field(default_factory=list)
    dry_run: bool = False
    since: datetime | None = None
    until: datetime | None = None
    project: str | None = None
    found: int = 0
    skipped: int = 0
    imported: int = 0
    failed: int = 0
    turns: int = 0
    entities: int = 0
    current: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "sources": list(self.sources),
                "dry_run": self.dry_run,
                "since": self.since.isoformat() if self.since else None,
                "until": self.until.isoformat() if self.until else None,
                "project": self.project,
                "found": self.found,
                "skipped": self.skipped,
                "imported": self.imported,
                "failed": self.failed,
                "turns": self.turns,
                "entities": self.entities,
                "current": self.current,
                "error": self.error,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            }


def _apply_spans(
    text: str, spans: list[Span], vault: Vault, turn_pairs: set[tuple[str, str]]
) -> str:
    """Placeholder the detected spans into ``text`` via the conversation vault,
    recording each distinct (type, value) into ``turn_pairs`` for per-turn
    counting. Same overlap/ordering rules as :func:`anonymize_into`."""
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    chosen: list[Span] = []
    last_end = -1
    for span in ordered:
        if span.start >= last_end and span.end > span.start:
            chosen.append(span)
            last_end = span.end
    for span in chosen:
        value = text[span.start : span.end]
        vault.placeholder_for(span.label, value)
        turn_pairs.add((span.label, value))
    out = text
    for span in reversed(chosen):
        placeholder = vault.placeholder_for(span.label, text[span.start : span.end])
        out = out[: span.start] + placeholder + out[span.end :]
    return out


def parse_until(value: str) -> datetime | None:
    """Inclusive upper-bound cutoff: a date-only value covers its whole day,
    naive values are local time. None on unparseable input."""
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if len(value) == 10:  # date-only: include the entire day
        dt = datetime.combine(dt.date(), time.max)
    return dt if dt.tzinfo else dt.astimezone()


def discover(
    sources: list[str],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    project: str | None = None,
    claude_root: Path | None = None,
    codex_root: Path | None = None,
) -> list[tuple[str, Path]]:
    """(source, path) pairs for every discovered session file."""
    out: list[tuple[str, Path]] = []
    if "claude-code" in sources:
        for path in claude_code.discover_sessions(
            claude_root, since=since, until=until, project=project
        ):
            out.append(("claude-code", path))
    if "codex" in sources:
        for path in codex.discover_sessions(codex_root, since=since, until=until, project=project):
            out.append(("codex", path))
    return out


def session_id_of(source: str, path: Path) -> str | None:
    """Cheap session id, without a full parse, for up-front dedupe."""
    if source == "claude-code":
        return path.stem
    # Codex rollout filenames end with the session UUID after the timestamp.
    name = path.stem  # rollout-YYYY-MM-DDTHH-MM-SS-<uuid>
    parts = name.split("-")
    if len(parts) >= 5:
        return "-".join(parts[-5:])
    return None


def session_preview(source: str, path: Path) -> tuple[str | None, str | None]:
    """(title, project) for the preview list; reads the file only as far as
    the first human prompt. The title is user text — never log it."""
    if source == "claude-code":
        return claude_code.preview_info(path)
    return codex.preview_info(path)


def _import_session(
    store: AuditStore, detector: Detector, session: ParsedSession, settings: Settings
) -> tuple[int, int]:
    """Detect, anonymize, and record every turn of one session; returns
    (turns recorded, entities found)."""
    conv_vault = Vault()
    turns = 0
    entities = 0
    for turn in session.turns:
        turn_pairs: set[tuple[str, str]] = set()
        processed: list[tuple[str, str, str]] = []
        for category, text in turn.segments:
            text = text[:MAX_SEGMENT_CHARS]
            spans = detector.detect(text)
            anonymized = _apply_spans(text, spans, conv_vault, turn_pairs)
            processed.append((text, anonymized, category))

        entity_counts: dict[str, int] = {}
        for label, _value in turn_pairs:
            entity_counts[label] = entity_counts.get(label, 0) + 1

        if settings.save_texts == "anonymized":
            kept = [seg for seg in processed if seg[0] != seg[1]]
        else:
            kept = processed

        store.record(
            source=session.source,
            wire_format=session.wire_format,
            model=turn.model or "unknown",
            entity_counts=entity_counts,
            kind="main",
            policy="imported",
            input_tokens=turn.input_tokens,
            output_tokens=turn.output_tokens,
            conversation_id=session.session_id,
            created_at=turn.timestamp,
            texts=kept,
        )
        turns += 1
        entities += sum(entity_counts.values())
    return turns, entities


def run_import(
    store: AuditStore,
    detector: Detector,
    *,
    sources: list[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    project: str | None = None,
    exclude_session_ids: Collection[str] | None = None,
    dry_run: bool = False,
    settings: Settings | None = None,
    job: ImportJob | None = None,
    claude_root: Path | None = None,
    codex_root: Path | None = None,
) -> ImportJob:
    """Import history sessions into the audit store.

    Sessions whose id already exists in the store are skipped, which makes
    re-runs incremental. ``exclude_session_ids`` drops user-deselected sessions
    up front (ids without a filename-derived session id can't be excluded). A
    session that fails to parse or record is logged and counted, never fatal to
    the batch. ``dry_run`` discovers and dedupes only.
    """
    job = job or ImportJob()
    settings = settings or get_settings()
    claude_root = claude_root or settings.claude_root
    codex_root = codex_root or settings.codex_root
    sources = [s for s in (sources or list(SOURCES)) if s in SOURCES]
    job.state = "running"
    job.sources = sources
    job.dry_run = dry_run
    job.since = since
    job.until = until
    job.project = project
    job.started_at = datetime.now().astimezone()
    try:
        found = discover(
            sources,
            since=since,
            until=until,
            project=project,
            claude_root=claude_root,
            codex_root=codex_root,
        )
        if exclude_session_ids:
            excluded = set(exclude_session_ids)
            found = [
                (src, p)
                for src, p in found
                if (sid := session_id_of(src, p)) is None or sid not in excluded
            ]
        job.found = len(found)

        known_ids = [sid for src, p in found if (sid := session_id_of(src, p))]
        existing = store.existing_conversation_ids(known_ids)

        parsers = {"claude-code": claude_code.parse_session, "codex": codex.parse_session}
        for source, path in found:
            quick_id = session_id_of(source, path)
            if quick_id and quick_id in existing:
                with job._lock:
                    job.skipped += 1
                continue
            with job._lock:
                job.current = str(path)
            try:
                session = parsers[source](path)
                if session is None or session.session_id in existing:
                    with job._lock:
                        job.skipped += 1
                    continue
                if dry_run:
                    with job._lock:
                        job.imported += 1
                        job.turns += len(session.turns)
                    continue
                turns, entities = _import_session(store, detector, session, settings)
                existing.add(session.session_id)
                with job._lock:
                    job.imported += 1
                    job.turns += turns
                    job.entities += entities
            except Exception:
                logger.exception("failed to import %s", path)
                with job._lock:
                    job.failed += 1
        job.state = "done"
    except Exception as exc:  # discovery/dedupe failure — surface it to the caller
        logger.exception("import run failed")
        with job._lock:
            job.state = "error"
            job.error = str(exc)
    finally:
        with job._lock:
            job.current = None
            job.finished_at = datetime.now().astimezone()
    return job
