"""JSON API for the web dashboard.

A versioned, read-plus-management API under ``/api/v1`` consumed by the bundled
Next.js dashboard. It serves filtered, paginated views over the audit store, the
per-interaction detail (metadata + detections + before/after text), aggregate
stats for charts, a live in-memory preview, and the management actions (delete an
interaction, clear the log, export). The proxy and its audit-write path are
untouched; this module only reads and, for the explicit management endpoints,
deletes.

Raw ``original`` text lives only in ``interactiontext``. The detail/export
endpoints that can return it are gated behind ``Settings.expose_plaintext`` so a
hosted deployment can redact originals; the list and summary endpoints never
return raw text.
"""

from __future__ import annotations

import csv
import io
import json
import threading
from datetime import datetime
from typing import Any, get_args

from fastapi import Depends, FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from privacy_kit import __version__
from privacy_kit.core.detectors import Detector
from privacy_kit.core.vault import Vault, anonymize_into
from privacy_kit.gateway.config import Settings, get_settings
from privacy_kit.gateway.importer import runner as importer_runner
from privacy_kit.gateway.importer.runner import ImportJob
from privacy_kit.gateway.store import AuditStore
from privacy_kit.gateway.store.models import Detection, Interaction, InteractionText

_MAX_PREVIEW_CHARS = 50_000

# Allowed values for the runtime-modifiable Literal settings, kept in sync with
# the ``Settings`` annotations so the API can't drift from the config.
_ALLOWED_POLICIES: frozenset[str] = frozenset(get_args(Settings.model_fields["policy"].annotation))
_ALLOWED_SAVE_TEXTS: frozenset[str] = frozenset(
    get_args(Settings.model_fields["save_texts"].annotation)
)

# The settings the dashboard may change at runtime. Everything here is
# operator-level (deployment-wide), never per-end-user, so this set doubles as
# the authorization scope for the future multi-user admin gate: when auth lands,
# the PATCH endpoint below is the single place to require admin. Settings absent
# here (model_id, db_path, host, port, cors_origins) can't change without a
# restart and stay env-only.
_EDITABLE_SETTINGS: frozenset[str] = frozenset({"policy", "save_texts", "threshold"})


def run_preview(detector: Detector, text: str) -> dict[str, Any]:
    """Live, in-memory PII detection for the preview tool. Nothing is persisted."""
    vault = Vault()
    spans = detector.detect(text)
    anonymized = anonymize_into(text, detector, vault)
    return {
        "spans": [
            {"start": s.start, "end": s.end, "label": s.label, "score": s.score} for s in spans
        ],
        "anonymized": anonymized,
        "counts": vault.type_counts,
    }


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _localize(dt: datetime) -> datetime:
    """Naive datetimes are interpreted as local time, like the CLI's --since."""
    return dt if dt.tzinfo else dt.astimezone()


def _import_window(since: str | None, until: str | None) -> tuple[datetime | None, datetime | None]:
    """Parse the import endpoints' since/until params; raises ``ValueError``
    with the user-facing message on bad input."""
    since_dt = _parse_dt(since)
    if since and since_dt is None:
        raise ValueError("since must be an ISO date or datetime (e.g. 2026-06-01)")
    if since_dt is not None:
        since_dt = _localize(since_dt)
    until_dt = importer_runner.parse_until(until) if until else None
    if until and until_dt is None:
        raise ValueError("until must be an ISO date or datetime (e.g. 2026-06-01)")
    return since_dt, until_dt


def _interaction_dict(row: Interaction) -> dict[str, Any]:
    return {
        "id": row.id,
        "created_at": row.created_at.isoformat(),
        "source": row.source,
        "wire_format": row.wire_format,
        "kind": row.kind,
        "model": row.model,
        "policy": row.policy,
        "language": row.language,
        "input_tokens": row.input_tokens,
        "output_tokens": row.output_tokens,
        "entity_total": row.entity_total,
        "entity_counts": row.entity_counts,
    }


def _detection_dict(row: Detection) -> dict[str, Any]:
    return {"id": row.id, "entity_type": row.entity_type, "count": row.count}


def _text_dict(row: InteractionText, *, expose_plaintext: bool) -> dict[str, Any]:
    return {
        "id": row.id,
        "seq": row.seq,
        "category": row.category,
        "original": row.original if expose_plaintext else None,
        "anonymized": row.anonymized,
    }


def _list_filters(
    source: list[str] = Query(default=[]),
    wire_format: list[str] = Query(default=[]),
    kind: list[str] = Query(default=[]),
    model: list[str] = Query(default=[]),
    policy: list[str] = Query(default=[]),
    language: list[str] = Query(default=[]),
    entity_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    min_entities: int | None = None,
    q: str | None = None,
) -> dict[str, Any]:
    """Shared filter query-params for the list and export endpoints."""
    return {
        "sources": source or None,
        "wire_formats": wire_format or None,
        "kinds": kind or None,
        "models": model or None,
        "policies": policy or None,
        "languages": language or None,
        "entity_type": entity_type,
        "date_from": _parse_dt(date_from),
        "date_to": _parse_dt(date_to),
        "min_entities": min_entities,
        "q": q,
    }


def register_webapi_routes(
    app: FastAPI, *, detector: Detector, store: AuditStore, settings: Settings | None = None
) -> None:
    """Mount the dashboard JSON API (`/api/v1`) onto ``app``."""
    settings = settings or get_settings()
    cfg = settings

    @app.get("/api/v1/summary")
    async def api_summary() -> JSONResponse:
        return JSONResponse(store.dashboard_summary())

    @app.get("/api/v1/filters")
    async def api_filters() -> JSONResponse:
        return JSONResponse(store.distinct_values())

    def _config_payload() -> dict[str, Any]:
        # Non-secret runtime configuration for the settings/about page.
        return {
            "version": __version__,
            "policy": cfg.policy,
            "save_texts": cfg.save_texts,
            "expose_plaintext": cfg.expose_plaintext,
            "model_id": cfg.model_id,
            "threshold": cfg.threshold,
            "db_path": str(cfg.db_path),
            "host": cfg.host,
            "port": cfg.port,
            "anthropic_upstream": cfg.anthropic_upstream,
            "openai_upstream": cfg.openai_upstream,
            "chatgpt_upstream": cfg.chatgpt_upstream,
            "otel_downstream": cfg.otel_downstream,
        }

    @app.get("/api/v1/config")
    async def api_config() -> JSONResponse:
        return JSONResponse(_config_payload())

    def _normalize_setting(field: str, value: Any) -> Any:
        """Validate one incoming setting, returning the value to apply. Raises
        ``ValueError`` with a user-facing message on bad input."""
        if field == "policy":
            if value not in _ALLOWED_POLICIES:
                raise ValueError(f"policy must be one of {sorted(_ALLOWED_POLICIES)}")
            return value
        if field == "save_texts":
            if value not in _ALLOWED_SAVE_TEXTS:
                raise ValueError(f"save_texts must be one of {sorted(_ALLOWED_SAVE_TEXTS)}")
            return value
        if field == "threshold":
            # bool is an int subclass; reject it so true/false can't pass as 0/1.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("threshold must be a number between 0.0 and 1.0")
            if not 0.0 <= value <= 1.0:
                raise ValueError("threshold must be between 0.0 and 1.0")
            return float(value)
        raise ValueError(f"unknown setting {field!r}")

    def _apply_setting(field: str, value: Any) -> None:
        setattr(cfg, field, value)
        # threshold is baked into the detector at construction, so update the
        # live instance too; it's read per detection (BardsAiOnnxDetector reads
        # self.threshold when filtering spans).
        if field == "threshold":
            detector.threshold = value  # type: ignore[attr-defined]

    @app.patch("/api/v1/config")
    async def api_update_config(request: Request) -> JSONResponse:
        # Runtime-modifiable, operator-level settings (see _EDITABLE_SETTINGS).
        # The proxy and detector read these per request off the shared objects
        # mutated here, so changes take effect immediately for subsequent
        # traffic. Changes are in-memory only and reset to the PII_* env/.env
        # values on restart. This endpoint is the single choke point for a future
        # multi-user admin-auth gate.
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "expected a JSON body"}, status_code=400)
        editable = sorted(_EDITABLE_SETTINGS)
        if not isinstance(body, dict) or not body:
            return JSONResponse(
                {"error": f"expected a JSON object with one or more of {editable}"},
                status_code=400,
            )
        unknown = set(body) - _EDITABLE_SETTINGS
        if unknown:
            return JSONResponse(
                {"error": f"unknown or read-only setting(s): {sorted(unknown)}"},
                status_code=400,
            )
        # Validate everything before applying anything, so a bad value can't
        # leave a partial update.
        try:
            normalized = {field: _normalize_setting(field, value) for field, value in body.items()}
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        for field, value in normalized.items():
            _apply_setting(field, value)
        return JSONResponse(_config_payload())

    # --- History import (Claude Code / Codex transcripts → audit log) --------
    # One job at a time; state lives in this closure. Detection runs in a
    # daemon thread (onnxruntime releases the GIL), the UI polls /status.
    import_state: dict[str, Any] = {"job": None, "thread": None}
    import_lock = threading.Lock()

    def _job_running() -> bool:
        thread = import_state["thread"]
        return thread is not None and thread.is_alive()

    @app.get("/api/v1/import/preview")
    async def api_import_preview(
        since: str | None = None, until: str | None = None, project: str | None = None
    ) -> JSONResponse:
        """Discovered history sessions per source, split new vs already imported."""
        try:
            since_dt, until_dt = _import_window(since, until)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        project_filter = project or None

        def build() -> dict[str, Any]:
            found = importer_runner.discover(
                list(importer_runner.SOURCES),
                since=since_dt,
                until=until_dt,
                project=project_filter,
            )
            ids = {(src, path): importer_runner.session_id_of(src, path) for src, path in found}
            existing = store.existing_conversation_ids(
                [sid for sid in ids.values() if sid is not None]
            )
            per_source: dict[str, dict[str, int]] = {
                src: {"found": 0, "new": 0, "imported": 0} for src in importer_runner.SOURCES
            }
            for (src, _path), sid in ids.items():
                stats = per_source[src]
                stats["found"] += 1
                if sid is not None and sid in existing:
                    stats["imported"] += 1
                else:
                    stats["new"] += 1
            return {"sources": per_source}

        return JSONResponse(await run_in_threadpool(build))

    @app.get("/api/v1/import/preview/sessions")
    async def api_import_preview_sessions(
        since: str | None = None,
        until: str | None = None,
        project: str | None = None,
        sources: str | None = None,
        limit: int = 200,
    ) -> JSONResponse:
        """Newest-first list of the sessions the filters would cover, capped at
        ``limit``. Titles are the first human prompt of each session — raw user
        text, so they are gated by ``expose_plaintext`` like every other
        plaintext-returning endpoint (and never logged)."""
        try:
            since_dt, until_dt = _import_window(since, until)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if sources:
            wanted = [s for s in sources.split(",") if s]
            if not wanted or any(s not in importer_runner.SOURCES for s in wanted):
                return JSONResponse(
                    {"error": f"sources must be a subset of {list(importer_runner.SOURCES)}"},
                    status_code=400,
                )
        else:
            wanted = list(importer_runner.SOURCES)
        project_filter = project or None
        cap = max(1, min(limit, 500))

        def build() -> dict[str, Any]:
            found = importer_runner.discover(
                wanted, since=since_dt, until=until_dt, project=project_filter
            )
            found.sort(key=lambda item: item[1].stat().st_mtime, reverse=True)
            page = [
                (src, path, importer_runner.session_id_of(src, path)) for src, path in found[:cap]
            ]
            existing = store.existing_conversation_ids(
                [sid for _src, _path, sid in page if sid is not None]
            )
            sessions: list[dict[str, Any]] = []
            for src, path, sid in page:
                if cfg.expose_plaintext:
                    title, proj = importer_runner.session_preview(src, path)
                else:
                    # No file reads at all when plaintext is off; the Claude
                    # project slug comes from the path, the Codex cwd doesn't.
                    title = None
                    proj = path.parent.name if src == "claude-code" else None
                sessions.append(
                    {
                        "source": src,
                        "id": sid,
                        "title": title,
                        "project": proj,
                        "modified_at": datetime.fromtimestamp(path.stat().st_mtime)
                        .astimezone()
                        .isoformat(),
                        "imported": sid is not None and sid in existing,
                    }
                )
            return {
                "total": len(found),
                "titles_redacted": not cfg.expose_plaintext,
                "sessions": sessions,
            }

        return JSONResponse(await run_in_threadpool(build))

    @app.post("/api/v1/import")
    async def api_import_start(request: Request) -> JSONResponse:
        """Start a background import; 409 while one is already running."""
        try:
            body = await request.json()
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        sources = body.get("sources") or list(importer_runner.SOURCES)
        if not isinstance(sources, list) or any(s not in importer_runner.SOURCES for s in sources):
            return JSONResponse(
                {"error": f"sources must be a subset of {list(importer_runner.SOURCES)}"},
                status_code=400,
            )
        since_raw = body.get("since")
        if since_raw is not None and not isinstance(since_raw, str):
            return JSONResponse({"error": "since must be a string"}, status_code=400)
        since = _parse_dt(since_raw)
        if since_raw and since is None:
            return JSONResponse(
                {"error": "since must be an ISO date or datetime (e.g. 2026-06-01)"},
                status_code=400,
            )
        if since is not None:
            since = _localize(since)
        until_raw = body.get("until")
        if until_raw is not None and not isinstance(until_raw, str):
            return JSONResponse({"error": "until must be a string"}, status_code=400)
        until = importer_runner.parse_until(until_raw) if until_raw else None
        if until_raw and until is None:
            return JSONResponse(
                {"error": "until must be an ISO date or datetime (e.g. 2026-06-01)"},
                status_code=400,
            )
        project = body.get("project")
        if project is not None and not isinstance(project, str):
            return JSONResponse({"error": "project must be a string"}, status_code=400)
        project = project or None
        dry_run = body.get("dry_run", False)
        if not isinstance(dry_run, bool):
            return JSONResponse({"error": "dry_run must be a boolean"}, status_code=400)
        exclude_raw = body.get("exclude_session_ids")
        if exclude_raw is not None and (
            not isinstance(exclude_raw, list) or any(not isinstance(x, str) for x in exclude_raw)
        ):
            return JSONResponse(
                {"error": "exclude_session_ids must be a list of strings"}, status_code=400
            )
        exclude_session_ids = set(exclude_raw) if exclude_raw else None
        with import_lock:
            if _job_running():
                return JSONResponse({"error": "an import is already running"}, status_code=409)
            job = ImportJob()
            # Dry runs keep the app's detector (the CLI swaps in a null one);
            # run_import short-circuits before detection, so it is never called.
            thread = threading.Thread(
                target=importer_runner.run_import,
                args=(store, detector),
                kwargs={
                    "sources": sources,
                    "since": since,
                    "until": until,
                    "project": project,
                    "exclude_session_ids": exclude_session_ids,
                    "dry_run": dry_run,
                    "settings": cfg,
                    "job": job,
                },
                name="privacy-kit-import",
                daemon=True,
            )
            import_state["job"] = job
            import_state["thread"] = thread
            thread.start()
        return JSONResponse(job.snapshot(), status_code=202)

    @app.get("/api/v1/import/status")
    async def api_import_status() -> JSONResponse:
        job = import_state["job"]
        if job is None:
            return JSONResponse({"state": "idle"})
        return JSONResponse(job.snapshot())

    @app.get("/api/v1/texts")
    async def api_texts(
        limit: int = 200,
        category: str | None = None,
        filters: dict[str, Any] = Depends(_list_filters),
    ) -> JSONResponse:
        # Flatten saved before/after segments across interactions for the texts
        # browser. ``original`` is gated by expose_plaintext, like the detail view.
        # ``category`` filters segments by origin ("user" | "tool"); it lives on the
        # segment, not the interaction, so it's applied here rather than in the store.
        interactions = store.iter_interactions(limit=min(limit, 500), **filters)
        out = []
        for it in interactions:
            if it.id is None:
                continue
            for t in store.texts(it.id):
                if category and t.category != category:
                    continue
                out.append(
                    {
                        "interaction_id": it.id,
                        "when": it.created_at.isoformat(),
                        "source": it.source,
                        "model": it.model,
                        "seq": t.seq,
                        "category": t.category,
                        "original": t.original if cfg.expose_plaintext else None,
                        "anonymized": t.anonymized,
                    }
                )
        return JSONResponse({"texts": out, "redacted": not cfg.expose_plaintext})

    @app.get("/api/v1/interactions")
    async def api_interactions(
        page: int = 1,
        page_size: int = 50,
        sort: str = "created_at",
        order: str = "desc",
        filters: dict[str, Any] = Depends(_list_filters),
    ) -> JSONResponse:
        rows, total = store.query_interactions(
            page=page, page_size=page_size, sort=sort, order=order, **filters
        )
        ids = [r.id for r in rows if r.id is not None]
        text_counts = store.text_counts_for(ids)
        items = []
        for row in rows:
            item = _interaction_dict(row)
            item["text_count"] = text_counts.get(row.id or -1, 0)
            item["detection_types"] = sorted(row.entity_counts.keys())
            items.append(item)
        page_size_c = max(1, min(page_size, 200))
        total_pages = (total + page_size_c - 1) // page_size_c if total else 0
        return JSONResponse(
            {
                "items": items,
                "page": max(1, page),
                "page_size": page_size_c,
                "total": total,
                "total_pages": total_pages,
            }
        )

    @app.get("/api/v1/interactions/{interaction_id}")
    async def api_interaction_detail(interaction_id: int) -> JSONResponse:
        row = store.get_interaction(interaction_id)
        if row is None:
            return JSONResponse({"error": "interaction not found"}, status_code=404)
        texts = store.texts(interaction_id)
        return JSONResponse(
            {
                "interaction": _interaction_dict(row),
                "detections": [_detection_dict(d) for d in store.detections(interaction_id)],
                "texts": [_text_dict(t, expose_plaintext=cfg.expose_plaintext) for t in texts],
                "texts_redacted": not cfg.expose_plaintext,
            }
        )

    @app.delete("/api/v1/interactions/{interaction_id}")
    async def api_delete_interaction(interaction_id: int) -> JSONResponse:
        if not store.delete_interaction(interaction_id):
            return JSONResponse({"error": "interaction not found"}, status_code=404)
        return JSONResponse({"deleted": interaction_id})

    @app.get("/api/v1/conversations")
    async def api_conversations(
        page: int = 1,
        page_size: int = 50,
        sort: str = "last_seen",
        order: str = "desc",
        filters: dict[str, Any] = Depends(_list_filters),
    ) -> JSONResponse:
        rows, total = store.list_conversations(
            page=page, page_size=page_size, sort=sort, order=order, **filters
        )
        page_size_c = max(1, min(page_size, 200))
        total_pages = (total + page_size_c - 1) // page_size_c if total else 0
        return JSONResponse(
            {
                "items": rows,
                "page": max(1, page),
                "page_size": page_size_c,
                "total": total,
                "total_pages": total_pages,
            }
        )

    @app.get("/api/v1/conversations/{conversation_id}")
    async def api_conversation_detail(conversation_id: str) -> JSONResponse:
        interactions = store.get_conversation(conversation_id)
        if interactions is None:
            return JSONResponse({"error": "conversation not found"}, status_code=404)
        # Each turn mirrors the single-interaction detail shape, reusing the same
        # helpers so plaintext redaction (expose_plaintext) is applied identically.
        turns = []
        background_count = 0
        for row in interactions:
            if row.kind != "main":
                background_count += 1
            iid = row.id or -1
            turns.append(
                {
                    "interaction": _interaction_dict(row),
                    "detections": [_detection_dict(d) for d in store.detections(iid)],
                    "texts": [
                        _text_dict(t, expose_plaintext=cfg.expose_plaintext)
                        for t in store.texts(iid)
                    ],
                }
            )
        return JSONResponse(
            {
                "conversation_id": conversation_id,
                "summary": store.conversation_summary(conversation_id, interactions),
                "turns": turns,
                "background_count": background_count,
                "texts_redacted": not cfg.expose_plaintext,
            }
        )

    @app.post("/api/v1/audit/clear")
    async def api_clear(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not (isinstance(body, dict) and body.get("confirm") is True):
            return JSONResponse(
                {"error": 'send {"confirm": true} to clear the entire audit log'},
                status_code=400,
            )
        return JSONResponse({"cleared": store.clear_all()})

    @app.post("/api/v1/preview")
    async def api_preview(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "expected a JSON body"}, status_code=400)
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str):
            return JSONResponse({"error": "expected {'text': <string>}"}, status_code=400)
        if len(text) > _MAX_PREVIEW_CHARS:
            return JSONResponse(
                {"error": f"text too long (max {_MAX_PREVIEW_CHARS} characters)"},
                status_code=413,
            )
        # CPU-bound inference off the event loop; result goes to the caller only —
        # deliberately no audit row and no logging.
        return JSONResponse(await run_in_threadpool(run_preview, detector, text))

    @app.get("/api/v1/export")
    async def api_export(
        fmt: str = Query("json", alias="format"),
        include_texts: bool = False,
        filters: dict[str, Any] = Depends(_list_filters),
    ) -> StreamingResponse:
        rows = store.iter_interactions(limit=None, **filters)

        if fmt == "csv":

            def csv_stream() -> Any:
                buf = io.StringIO()
                writer = csv.writer(buf)

                def flush() -> str:
                    out = buf.getvalue()
                    buf.seek(0)
                    buf.truncate(0)
                    return out

                writer.writerow(
                    [
                        "id",
                        "created_at",
                        "source",
                        "wire_format",
                        "kind",
                        "model",
                        "policy",
                        "language",
                        "input_tokens",
                        "output_tokens",
                        "entity_total",
                        "entity_counts",
                    ]
                )
                yield flush()
                for row in rows:
                    writer.writerow(
                        [
                            row.id,
                            row.created_at.isoformat(),
                            row.source,
                            row.wire_format,
                            row.kind,
                            row.model,
                            row.policy,
                            row.language or "",
                            "" if row.input_tokens is None else row.input_tokens,
                            "" if row.output_tokens is None else row.output_tokens,
                            row.entity_total,
                            json.dumps(row.entity_counts, ensure_ascii=False),
                        ]
                    )
                    yield flush()

            return StreamingResponse(
                csv_stream(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=privacy-kit-interactions.csv"
                },
            )

        data = []
        for row in rows:
            item = _interaction_dict(row)
            assert row.id is not None
            item["detections"] = [_detection_dict(d) for d in store.detections(row.id)]
            if include_texts:
                item["texts"] = [
                    _text_dict(t, expose_plaintext=cfg.expose_plaintext)
                    for t in store.texts(row.id)
                ]
            data.append(item)
        payload = json.dumps({"interactions": data}, ensure_ascii=False, indent=2)
        return StreamingResponse(
            iter([payload]),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=privacy-kit-interactions.json"},
        )
