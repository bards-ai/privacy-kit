"""The privacy-kit gateway proxy.

A FastAPI app that AI tools route through via their ``*_BASE_URL`` overrides. For
each request it anonymizes the prompt text, forwards the sanitized body to the
real upstream (passing the client's own auth through), rehydrates the response
with the real values, and writes a metadata-only audit row.

Streaming (SSE) responses are de-anonymized on the fly — see ``streaming.py``.
The upstream stream is opened before the client response is constructed, so an
upstream refusal (auth error, validation error) surfaces with its real status
code instead of a broken 200 stream.

``create_app`` takes the detector, store, and forwarder as dependencies so it can
be tested with stubs and no network. ``build_default_app`` wires the real ones.
"""

from __future__ import annotations

import gzip
import json
import zlib
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from privacy_kit.core.detectors import Detector
from privacy_kit.core.vault import Vault, anonymize_into, deanonymize
from privacy_kit.gateway.config import Settings, get_settings
from privacy_kit.gateway.otel import register_otel_routes
from privacy_kit.gateway.proxy.streaming import PlaceholderStreamDecoder, StreamUsage, rewrite_sse
from privacy_kit.gateway.proxy.transform import (
    REQUEST_TRANSFORMS,
    RESPONSE_TRANSFORMS,
    extract_tokens,
)
from privacy_kit.gateway.store import AuditStore

# ``content-encoding`` is dropped because we decompress the request body here and
# forward it re-serialized (uncompressed); leaving the header would mislabel it.
_DROP_HEADERS = {
    "host",
    "content-length",
    "content-encoding",
    "accept-encoding",
    "connection",
}

# Default source label per wire format when the client doesn't set
# x-privacy-kit-source.
_DEFAULT_SOURCE = {
    "anthropic": "claude-code",
    "openai_chat": "openai-chat",
    "openai_responses": "codex",
}


def _source_label(request: Request, default: str) -> str:
    """The client-declared source tool, with the legacy Sieve header honored."""
    return (
        request.headers.get("x-privacy-kit-source")
        or request.headers.get("x-sieve-source")
        or default
    )


def _decompress(raw: bytes, encoding: str) -> bytes:
    """Decode a request body per its ``Content-Encoding``.

    Returns ``raw`` unchanged for identity/empty encodings. Handles ``gzip`` and
    ``deflate`` from the stdlib and ``br``/``zstd`` when their optional libraries
    are installed. Raises ``ValueError`` on an unsupported or unavailable codec.
    """
    enc = encoding.strip().lower()
    if not enc or enc == "identity":
        return raw
    if enc == "gzip":
        return gzip.decompress(raw)
    if enc == "deflate":
        try:
            return zlib.decompress(raw)
        except zlib.error:
            return zlib.decompress(raw, -zlib.MAX_WBITS)  # raw deflate, no zlib header
    if enc == "br":
        try:
            import brotli  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ValueError("brotli-encoded request body but brotli is not installed") from exc
        decompressed: bytes = brotli.decompress(raw)
        return decompressed
    if enc == "zstd":
        try:
            import zstandard  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ValueError("zstd-encoded request body but zstandard is not installed") from exc
        unzstd: bytes = zstandard.ZstdDecompressor().decompress(raw)
        return unzstd
    raise ValueError(f"unsupported Content-Encoding: {enc!r}")


async def _read_json_body(request: Request) -> Any:
    """Read the request body, decompressing per ``Content-Encoding``, then parse JSON.

    Clients such as Claude Code gzip large request bodies. Starlette's
    ``request.json()`` reads the raw bytes without decompressing, so it fails on a
    compressed body; read and decode the body ourselves instead.
    """
    raw = await request.body()
    return json.loads(_decompress(raw, request.headers.get("content-encoding", "")))


@dataclass
class ForwardResult:
    """The upstream response, parsed."""

    status_code: int
    json: dict[str, Any] | None
    headers: dict[str, str]


class Forwarder(Protocol):
    """Sends the sanitized request upstream and returns the parsed response."""

    async def __call__(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> ForwardResult: ...


class HttpxForwarder:
    """Default forwarder: POSTs the payload upstream with httpx."""

    def __init__(self, timeout: float = 600.0) -> None:
        self._timeout = timeout

    async def __call__(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> ForwardResult:
        import httpx

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
        try:
            data = response.json()
        except ValueError:
            data = None
        return ForwardResult(response.status_code, data, dict(response.headers))


class StreamForwarder(Protocol):
    """Opens a streaming upstream connection yielding (status, headers, lines)."""

    def open(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> Any: ...  # AbstractAsyncContextManager[tuple[int, dict[str, str], AsyncIterator[str]]]


class HttpxStreamForwarder:
    """Default streaming forwarder: opens an httpx stream and yields its lines."""

    @asynccontextmanager
    async def open(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> AsyncIterator[tuple[int, dict[str, str], AsyncIterator[str]]]:
        import httpx

        async with (
            httpx.AsyncClient(timeout=None) as client,
            client.stream("POST", url, headers=headers, json=payload) as response,
        ):

            async def lines() -> AsyncIterator[str]:
                async for line in response.aiter_lines():
                    yield line

            yield response.status_code, dict(response.headers), lines()


def _passthrough_headers(headers: Any) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _DROP_HEADERS}


def _upstream_base(wire: str, settings: Settings) -> str:
    base = settings.anthropic_upstream if wire == "anthropic" else settings.openai_upstream
    return base.rstrip("/")


def create_app(
    *,
    detector: Detector,
    store: AuditStore,
    forwarder: Forwarder | None = None,
    stream_forwarder: StreamForwarder | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Build the gateway app from its dependencies."""
    from privacy_kit import __version__

    settings = settings or get_settings()
    forward: Forwarder = forwarder or HttpxForwarder()
    stream_open: StreamForwarder = stream_forwarder or HttpxStreamForwarder()
    app = FastAPI(title="privacy-kit gateway", version=__version__)

    def _audit(
        request: Request,
        wire: str,
        model: str,
        vault: Vault,
        in_tokens: int | None = None,
        out_tokens: int | None = None,
    ) -> None:
        source = _source_label(request, _DEFAULT_SOURCE[wire])
        with suppress(Exception):  # auditing must never break the proxy path
            store.record(
                source=source,
                wire_format=wire,
                model=model,
                entity_counts=vault.type_counts,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            )

    async def proxy(request: Request, wire: str) -> JSONResponse | StreamingResponse:
        try:
            body = await _read_json_body(request)
        except (ValueError, OSError):
            return JSONResponse({"error": "could not decode request body"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "expected a JSON object body"}, status_code=400)

        model = str(body.get("model", "unknown"))
        vault = Vault()

        def anon(text: str) -> str:
            return anonymize_into(text, detector, vault)

        # Model inference is CPU-bound; run it off the event loop so concurrent
        # requests (Claude Code fires several in parallel) don't stall each other.
        await run_in_threadpool(REQUEST_TRANSFORMS[wire], body, anon)
        url = _upstream_base(wire, settings) + request.url.path
        headers = _passthrough_headers(request.headers)

        if body.get("stream"):
            stack = AsyncExitStack()
            status, up_headers, lines = await stack.enter_async_context(
                stream_open.open(url, headers, body)
            )
            content_type = {k.lower(): v for k, v in up_headers.items()}.get("content-type", "")

            if status >= 400 or (content_type and "text/event-stream" not in content_type):
                # Upstream refused to stream (or answered non-SSE): collect the
                # body and pass it through under its real status code. Anything
                # it echoes can only be placeholders — it never saw raw PII.
                raw = "\n".join([line async for line in lines])
                await stack.aclose()
                try:
                    payload: Any = json.loads(raw) if raw.strip() else {}
                except ValueError:
                    payload = {"error": {"message": raw}}
                in_tokens: int | None = None
                out_tokens: int | None = None
                if isinstance(payload, dict) and 200 <= status < 300:
                    RESPONSE_TRANSFORMS[wire](payload, lambda t: deanonymize(t, vault))
                    in_tokens, out_tokens = extract_tokens(wire, payload)
                _audit(request, wire, model, vault, in_tokens, out_tokens)
                return JSONResponse(payload, status)

            usage = StreamUsage()
            decoder = PlaceholderStreamDecoder(vault)

            async def event_stream() -> AsyncIterator[str]:
                try:
                    async for chunk in rewrite_sse(lines, wire, decoder, usage):
                        yield chunk
                finally:
                    # Audit even on a half-finished stream — PII already left.
                    _audit(request, wire, model, vault, usage.input_tokens, usage.output_tokens)
                    await stack.aclose()

            return StreamingResponse(
                event_stream(), status_code=status, media_type="text/event-stream"
            )

        result = await forward(url, headers, body)
        tokens: tuple[int | None, int | None] = (None, None)
        if isinstance(result.json, dict) and 200 <= result.status_code < 300:
            RESPONSE_TRANSFORMS[wire](result.json, lambda t: deanonymize(t, vault))
            tokens = extract_tokens(wire, result.json)
        _audit(request, wire, model, vault, *tokens)
        return JSONResponse(result.json if result.json is not None else {}, result.status_code)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/messages", response_model=None)
    async def anthropic_messages(request: Request) -> JSONResponse | StreamingResponse:
        return await proxy(request, "anthropic")

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request) -> JSONResponse:
        # Anonymize and forward so the count reflects sanitized text; nothing to
        # rehydrate and no interaction to audit.
        try:
            body = await _read_json_body(request)
        except (ValueError, OSError):
            return JSONResponse({"error": "could not decode request body"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "expected a JSON object body"}, status_code=400)
        vault = Vault()
        await run_in_threadpool(
            REQUEST_TRANSFORMS["anthropic"], body, lambda t: anonymize_into(t, detector, vault)
        )
        url = _upstream_base("anthropic", settings) + request.url.path
        result = await forward(url, _passthrough_headers(request.headers), body)
        return JSONResponse(result.json if result.json is not None else {}, result.status_code)

    @app.post("/v1/chat/completions", response_model=None)
    async def openai_chat(request: Request) -> JSONResponse | StreamingResponse:
        return await proxy(request, "openai_chat")

    @app.post("/v1/responses", response_model=None)
    async def openai_responses(request: Request) -> JSONResponse | StreamingResponse:
        return await proxy(request, "openai_responses")

    register_otel_routes(app, detector=detector, store=store, settings=settings)

    return app


def build_default_app() -> FastAPI:
    """Production app: real on-device detector + audit store + httpx upstream."""
    from privacy_kit.core.detectors import BardsAiOnnxDetector

    settings = get_settings()
    detector = BardsAiOnnxDetector(model_id=settings.model_id, threshold=settings.threshold)
    detector.warmup()
    return create_app(detector=detector, store=AuditStore(), settings=settings)
