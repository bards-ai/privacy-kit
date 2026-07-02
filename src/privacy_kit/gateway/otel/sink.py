"""OTLP/HTTP log sink.

Receives OpenTelemetry logs (and traces/metrics) over OTLP/HTTP **JSON**, scrubs
PII out of every string value, records what was found to the audit store, and
optionally re-exports the scrubbed payload to a downstream collector.

This is the path that protects telemetry pipelines: Claude Code et al. only
include prompt text in logs when ``OTEL_LOG_USER_PROMPTS=1``, and this sink makes
sure that text is pseudonymized before it is stored or forwarded anywhere.

Unlike the proxy, scrubbing here is **not** reversed — observability data keeps
the placeholders. Only OTLP/JSON is supported; point exporters at this sink with
``OTEL_EXPORTER_OTLP_PROTOCOL=http/json``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from privacy_kit.core.detectors import Detector
from privacy_kit.core.vault import Vault, anonymize_into
from privacy_kit.gateway.config import Settings
from privacy_kit.gateway.store import AuditStore

# Downstream re-export: (url, headers, payload) -> None
DownstreamForward = Callable[[str, dict[str, str], dict[str, Any]], Awaitable[None]]


def _scrub_in_place(node: Any, anon: Callable[[str], str]) -> None:
    """Recursively anonymize every OTLP ``stringValue`` in ``node``."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "stringValue" and isinstance(value, str):
                node[key] = anon(value)
            else:
                _scrub_in_place(value, anon)
    elif isinstance(node, list):
        for item in node:
            _scrub_in_place(item, anon)


def scrub_otlp(payload: dict[str, Any], detector: Detector) -> dict[str, int]:
    """Anonymize all string values in ``payload`` in place; return entity counts."""
    vault = Vault()
    _scrub_in_place(payload, lambda text: anonymize_into(text, detector, vault))
    return vault.type_counts


async def _httpx_forward(url: str, headers: dict[str, str], payload: dict[str, Any]) -> None:
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        await client.post(url, headers=headers, json=payload)


_DROP_HEADERS = {"host", "content-length", "accept-encoding", "connection"}


def register_otel_routes(
    app: FastAPI,
    *,
    detector: Detector,
    store: AuditStore,
    settings: Settings,
    forward: DownstreamForward | None = None,
) -> None:
    """Mount /v1/logs, /v1/traces, /v1/metrics OTLP-JSON endpoints onto ``app``."""
    downstream = forward or _httpx_forward

    async def handle(request: Request, signal: str, *, audit: bool) -> JSONResponse:
        content_type = request.headers.get("content-type", "")
        if "json" not in content_type:
            return JSONResponse(
                {
                    "error": "The privacy-kit OTLP sink accepts JSON only; set "
                    "OTEL_EXPORTER_OTLP_PROTOCOL=http/json."
                },
                status_code=415,
            )
        payload = await request.json()
        if not isinstance(payload, dict):
            return JSONResponse({"error": "expected a JSON object body"}, status_code=400)

        # CPU-bound model inference — keep it off the event loop.
        counts = await run_in_threadpool(scrub_otlp, payload, detector)
        if audit and counts:
            source = (
                request.headers.get("x-privacy-kit-source")
                or request.headers.get("x-sieve-source")
                or "otel"
            )
            with suppress(Exception):
                store.record(
                    source=source,
                    wire_format="otel",
                    model=signal,
                    entity_counts=counts,
                )
        if settings.otel_downstream:
            headers = {k: v for k, v in request.headers.items() if k.lower() not in _DROP_HEADERS}
            url = settings.otel_downstream.rstrip("/") + f"/v1/{signal}"
            with suppress(Exception):  # forwarding must not break ingestion
                await downstream(url, headers, payload)

        # OTLP ExportXxxServiceResponse is an empty object on full success.
        return JSONResponse({})

    @app.post("/v1/logs")
    async def otlp_logs(request: Request) -> JSONResponse:
        return await handle(request, "logs", audit=True)

    @app.post("/v1/traces")
    async def otlp_traces(request: Request) -> JSONResponse:
        return await handle(request, "traces", audit=False)

    @app.post("/v1/metrics")
    async def otlp_metrics(request: Request) -> JSONResponse:
        return await handle(request, "metrics", audit=False)
