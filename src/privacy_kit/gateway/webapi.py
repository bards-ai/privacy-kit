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
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from privacy_kit import __version__
from privacy_kit.core.detectors import Detector
from privacy_kit.core.vault import Vault, anonymize_into
from privacy_kit.gateway.config import Settings, get_settings
from privacy_kit.gateway.store import AuditStore
from privacy_kit.gateway.store.models import Detection, Interaction, InteractionText

_MAX_PREVIEW_CHARS = 50_000


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


def _interaction_dict(row: Interaction) -> dict[str, Any]:
    return {
        "id": row.id,
        "created_at": row.created_at.isoformat(),
        "source": row.source,
        "wire_format": row.wire_format,
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
        "original": row.original if expose_plaintext else None,
        "anonymized": row.anonymized,
    }


def _list_filters(
    source: list[str] = Query(default=[]),
    wire_format: list[str] = Query(default=[]),
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

    @app.get("/api/v1/config")
    async def api_config() -> JSONResponse:
        # Non-secret runtime configuration for the settings/about page.
        return JSONResponse(
            {
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
        )

    @app.get("/api/v1/texts")
    async def api_texts(
        limit: int = 200, filters: dict[str, Any] = Depends(_list_filters)
    ) -> JSONResponse:
        # Flatten saved before/after segments across interactions for the texts
        # browser. ``original`` is gated by expose_plaintext, like the detail view.
        interactions = store.iter_interactions(limit=min(limit, 500), **filters)
        out = []
        for it in interactions:
            if it.id is None:
                continue
            for t in store.texts(it.id):
                out.append(
                    {
                        "interaction_id": it.id,
                        "when": it.created_at.isoformat(),
                        "source": it.source,
                        "model": it.model,
                        "seq": t.seq,
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
