"""The privacy-kit gateway proxy.

A FastAPI app that AI tools route through via their ``*_BASE_URL`` overrides. For
each request it detects PII and writes an audit row (metadata plus user-authored
text and tool/file data segments filtered by ``Settings.save_texts``; system
prompts and other machine-authored text are never stored). What it forwards
depends on ``Settings.policy``: under the default ``monitor`` policy the prompt is
sent **unchanged** (real values reach the upstream — detection is logged only);
under ``pseudonymize`` the PII is replaced with placeholders before forwarding and
the response is rehydrated with the real values. The client's own auth passes
through either way.

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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from privacy_kit.core.detectors import Detector
from privacy_kit.core.vault import Vault, anonymize_into, deanonymize
from privacy_kit.gateway.config import Settings, get_settings
from privacy_kit.gateway.language import detect_language
from privacy_kit.gateway.otel import register_otel_routes
from privacy_kit.gateway.proxy.classify import classify_kind
from privacy_kit.gateway.proxy.streaming import PlaceholderStreamDecoder, StreamUsage, rewrite_sse
from privacy_kit.gateway.proxy.transform import (
    REQUEST_TRANSFORMS,
    RESPONSE_TRANSFORMS,
    Author,
    extract_tokens,
)
from privacy_kit.gateway.store import AuditStore
from privacy_kit.gateway.ui import register_ui_routes
from privacy_kit.gateway.webapi import register_webapi_routes

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

# Cursor hook events we handle, mapped to the body field holding scannable text
# and the kind of decision the event expects back (prompt => continue; file/tool
# access => permission). See https://cursor.com/docs/hooks.
_CURSOR_HOOKS = {
    "beforeSubmitPrompt": ("prompt", "continue"),
    "beforeReadFile": ("content", "permission"),
}


def _cursor_allow(kind: str) -> JSONResponse:
    """The allow decision for a Cursor hook of the given kind."""
    return JSONResponse({"continue": True} if kind == "continue" else {"permission": "allow"})


def _cursor_deny(kind: str, message: str) -> JSONResponse:
    """The deny decision for a Cursor hook of the given kind, with a reason."""
    if kind == "continue":
        return JSONResponse({"continue": False, "user_message": message})
    return JSONResponse({"permission": "deny", "user_message": message})


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
            import brotli  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ValueError("brotli-encoded request body but brotli is not installed") from exc
        decompressed: bytes = brotli.decompress(raw)
        return decompressed
    if enc == "zstd":
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ValueError("zstd-encoded request body but zstandard is not installed") from exc
        # decompressobj(), not decompress(): streaming clients (Codex) emit
        # frames without a content-size header, which the one-shot API rejects.
        unzstd: bytes = zstandard.ZstdDecompressor().decompressobj().decompress(raw)
        return unzstd
    raise ValueError(f"unsupported Content-Encoding: {enc!r}")


async def _read_json_body(request: Request) -> Any:
    """Read the request body, decompressing per ``Content-Encoding``, then parse JSON.

    Clients such as Claude Code gzip large request bodies. Starlette's
    ``request.json()`` reads the raw bytes without decompressing, so it fails on a
    compressed body; read and decode the body ourselves instead.
    """
    raw = await request.body()
    encoding = request.headers.get("content-encoding", "")
    try:
        decoded = _decompress(raw, encoding)
    except ValueError:
        raise  # controlled messages from _decompress; safe to surface
    except Exception as exc:
        # Corrupt compressed data. The library's message may quote payload
        # bytes, so replace it rather than propagate it.
        raise ValueError(f"could not decompress {encoding!r} request body") from exc
    try:
        return json.loads(decoded)
    except UnicodeDecodeError as exc:
        raise ValueError("request body is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        # exc.msg is the bare reason ("Expecting value", …) — no payload text.
        raise ValueError(f"invalid JSON in request body: {exc.msg}") from exc


def _decode_error(request: Request, exc: Exception) -> JSONResponse:
    """A 400 that says why, plus the request metadata needed to debug routing.

    Goes only to the local client, which owns the original text; the reason
    strings are controlled by ``_read_json_body`` and carry no payload bytes.
    """
    return JSONResponse(
        {
            "error": f"could not decode request body: {exc}",
            "content_type": request.headers.get("content-type", ""),
            "content_encoding": request.headers.get("content-encoding", ""),
        },
        status_code=400,
    )


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


# Codex signed in with a ChatGPT account sends this header on every request; an
# API-key session never does. It is how we tell a subscription model call (which
# must reach chatgpt.com's backend) from an API-key one (api.openai.com).
_CHATGPT_ACCOUNT_HEADER = "chatgpt-account-id"


def _responses_route(request: Request, settings: Settings) -> tuple[str | None, str | None]:
    """Pick (upstream, path) for an OpenAI Responses request by auth signal.

    A ChatGPT-subscription Codex call carries ``chatgpt-account-id`` and must go
    to the ChatGPT backend's Codex path; api.openai.com's ``/v1/responses``
    rejects the ChatGPT OAuth token. Returns ``(None, None)`` for an API-key
    call so the default upstream and the request's own path are used.
    """
    if request.headers.get(_CHATGPT_ACCOUNT_HEADER):
        return settings.chatgpt_upstream, "/codex/responses"
    return None, None


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

    # Same-origin in production (the dashboard proxies API calls server-side), so
    # CORS stays off unless explicitly configured for host development.
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _audit(
        request: Request,
        wire: str,
        model: str,
        vault: Vault,
        in_tokens: int | None = None,
        out_tokens: int | None = None,
        texts: list[tuple[str, str]] | None = None,
        kind: str = "main",
    ) -> None:
        source = _source_label(request, _DEFAULT_SOURCE[wire])
        with suppress(Exception):  # auditing must never break the proxy path
            pairs = texts or []
            # Detect language from the original user text (unfiltered, so it
            # still works when only anonymized segments are saved).
            language = detect_language(" ".join(o for o, _ in pairs if o))
            if settings.save_texts == "anonymized":
                pairs = [(o, a) for o, a in pairs if o != a]
            else:  # "all"
                pairs = [(o, a) for o, a in pairs if o]
            store.record(
                source=source,
                wire_format=wire,
                kind=kind,
                model=model,
                policy=settings.policy,
                entity_counts=vault.type_counts,
                language=language,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                texts=pairs,
            )

    async def proxy(
        request: Request,
        wire: str,
        upstream: str | None = None,
        path: str | None = None,
    ) -> JSONResponse | StreamingResponse:
        try:
            body = await _read_json_body(request)
        except (ValueError, OSError) as exc:
            return _decode_error(request, exc)
        if not isinstance(body, dict):
            return JSONResponse({"error": "expected a JSON object body"}, status_code=400)

        model = str(body.get("model", "unknown"))
        # Classify the call's purpose on the original body, before anonymization
        # mutates it — keeps background chatter (safety/helper) separable from
        # the real conversation in the dashboard.
        kind = classify_kind(wire, body)
        vault = Vault()
        # Savable (original, anonymized) pairs: USER and TOOL segments that are
        # novel on this turn. Safe without a lock: the whole transform runs in
        # one threadpool worker per request and the await below orders all later
        # reads — revisit if segment anonymization is ever parallelized.
        captured: list[tuple[str, str]] = []

        # In "monitor" mode we still run detection (to populate the vault and log
        # what was present) but forward the original text — real values reach the
        # upstream. "pseudonymize" forwards the placeholdered text instead.
        forward_original = settings.policy == "monitor"

        def anon(text: str, author: Author, novel: bool) -> str:
            cleaned = anonymize_into(text, detector, vault)
            if novel and author in (Author.USER, Author.TOOL):
                captured.append((text, cleaned))
            return text if forward_original else cleaned

        # Model inference is CPU-bound; run it off the event loop so concurrent
        # requests (Claude Code fires several in parallel) don't stall each other.
        await run_in_threadpool(REQUEST_TRANSFORMS[wire], body, anon)
        base = upstream.rstrip("/") if upstream else _upstream_base(wire, settings)
        url = base + (path or request.url.path)
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
                _audit(
                    request, wire, model, vault, in_tokens, out_tokens, texts=captured, kind=kind
                )
                return JSONResponse(payload, status)

            usage = StreamUsage()
            decoder = PlaceholderStreamDecoder(vault)

            async def event_stream() -> AsyncIterator[str]:
                try:
                    async for chunk in rewrite_sse(lines, wire, decoder, usage):
                        yield chunk
                finally:
                    # Audit even on a half-finished stream — PII already left.
                    _audit(
                        request,
                        wire,
                        model,
                        vault,
                        usage.input_tokens,
                        usage.output_tokens,
                        texts=captured,
                        kind=kind,
                    )
                    await stack.aclose()

            return StreamingResponse(
                event_stream(), status_code=status, media_type="text/event-stream"
            )

        result = await forward(url, headers, body)
        tokens: tuple[int | None, int | None] = (None, None)
        if isinstance(result.json, dict) and 200 <= result.status_code < 300:
            RESPONSE_TRANSFORMS[wire](result.json, lambda t: deanonymize(t, vault))
            tokens = extract_tokens(wire, result.json)
        _audit(request, wire, model, vault, *tokens, texts=captured, kind=kind)
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
        except (ValueError, OSError) as exc:
            return _decode_error(request, exc)
        if not isinstance(body, dict):
            return JSONResponse({"error": "expected a JSON object body"}, status_code=400)
        vault = Vault()

        def _anon(t: str, _author: Author, _novel: bool) -> str:
            # Monitor mode forwards the original, so count its tokens; pseudonymize
            # counts the placeholdered text that actually goes upstream.
            return t if settings.policy == "monitor" else anonymize_into(t, detector, vault)

        await run_in_threadpool(REQUEST_TRANSFORMS["anthropic"], body, _anon)
        url = _upstream_base("anthropic", settings) + request.url.path
        result = await forward(url, _passthrough_headers(request.headers), body)
        return JSONResponse(result.json if result.json is not None else {}, result.status_code)

    @app.post("/v1/chat/completions", response_model=None)
    async def openai_chat(request: Request) -> JSONResponse | StreamingResponse:
        return await proxy(request, "openai_chat")

    @app.post("/v1/responses", response_model=None)
    async def openai_responses(request: Request) -> JSONResponse | StreamingResponse:
        # Codex routes the model call here for both auth modes. A ChatGPT-account
        # (subscription) session is detected by its header and forwarded to the
        # ChatGPT backend's Codex path; an API-key session keeps the default
        # api.openai.com upstream. Same Responses wire format either way; the
        # client's own auth token passes through untouched, like the Anthropic
        # OAuth path. Subscription routing is experimental.
        upstream, path = _responses_route(request, settings)
        return await proxy(request, "openai_responses", upstream=upstream, path=path)

    @app.post("/v1/cursor-hook")
    async def cursor_hook(request: Request) -> JSONResponse:
        """Audit (and optionally block) a Cursor hook event.

        Cursor surfaces that bypass the gateway base URL — Composer, the agent
        loop, inline edit, Tab — can't be pseudonymized, so the ``privacy-kit hook
        cursor`` command POSTs each ``beforeSubmitPrompt``/``beforeReadFile`` event
        here. We scan its text with the warm detector, record an audit row
        (``source="cursor"``), and return the hook decision: allow under the
        default monitor policy, or deny when ``PII_CURSOR_BLOCK`` is set and PII was
        found. Hooks cannot redact, so blocking is the only way to stop PII here.
        Anything unexpected fails open (allow) — privacy-kit must never wedge Cursor.
        """
        try:
            body = await _read_json_body(request)
        except (ValueError, OSError):
            return _cursor_allow("continue")
        if not isinstance(body, dict):
            return _cursor_allow("continue")

        event = str(body.get("hook_event_name", ""))
        field, kind = _CURSOR_HOOKS.get(event, ("", "continue"))
        text = body.get(field) if field else None
        if not isinstance(text, str) or not text:
            return _cursor_allow(kind)

        vault = Vault()
        cleaned = await run_in_threadpool(anonymize_into, text, detector, vault)
        with suppress(Exception):  # auditing must never break the hook path
            pairs = [(text, cleaned)]
            if settings.save_texts == "anonymized":
                pairs = [(o, a) for o, a in pairs if o != a]
            store.record(
                source="cursor",
                wire_format=f"cursor:{event}",
                model=str(body.get("model", "unknown")),
                policy=settings.policy,
                entity_counts=vault.type_counts,
                texts=pairs,
            )

        if settings.cursor_block and vault.type_counts:
            types = ", ".join(sorted(vault.type_counts))
            return _cursor_deny(
                kind,
                f"privacy-kit blocked this: detected PII ({types}) that would reach "
                "Cursor's backend unredacted.",
            )
        return _cursor_allow(kind)

    register_otel_routes(app, detector=detector, store=store, settings=settings)
    register_ui_routes(app, detector=detector, store=store)
    register_webapi_routes(app, detector=detector, store=store, settings=settings)

    return app


def build_default_app() -> FastAPI:
    """Production app: real on-device detector + audit store + httpx upstream."""
    from privacy_kit.core.detectors import BardsAiOnnxDetector

    settings = get_settings()
    detector = BardsAiOnnxDetector(model_id=settings.model_id, threshold=settings.threshold)
    detector.warmup()
    return create_app(detector=detector, store=AuditStore(), settings=settings)
