"""Observation proxy: capture exactly what Codex sends, forward it verbatim.

A debugging tool, NOT a privacy block: request and response bodies are written
in full to a local log file so we can learn the wire protocol of Codex's
ChatGPT-backend mode. Auth header values are redacted in the log; everything
else (including prompt text) is yours and stays on this machine. Delete the
log when done.

Usage:
    uv run python scripts/codex_probe.py [--port 8788] \
        [--upstream https://chatgpt.com/backend-api] [--log codex_probe.log]

Then point Codex at it in ~/.codex/config.toml:
    chatgpt_base_url = "http://127.0.0.1:8788/"
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response, StreamingResponse

REDACT_HEADERS = {"authorization", "cookie", "set-cookie", "openai-sentinel-token"}
SKIP_FORWARD_HEADERS = {"host", "content-length", "transfer-encoding", "connection"}


def _redact(headers: dict[str, str]) -> dict[str, str]:
    return {
        k: (f"<redacted len={len(v)} prefix={v[:12]!r}>" if k.lower() in REDACT_HEADERS else v)
        for k, v in headers.items()
    }


class Capture:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._n = 0

    def write(self, kind: str, **fields: Any) -> int:
        self._n += 1
        entry = {
            "n": self._n,
            "at": datetime.datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            **fields,
        }
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=repr) + "\n")
        return self._n


def _body_for_log(raw: bytes, headers: dict[str, str]) -> Any:
    encoding = headers.get("content-encoding", "").lower()
    data = raw
    try:
        if encoding == "gzip":
            import gzip

            data = gzip.decompress(raw)
        elif encoding == "zstd":
            import zstandard

            data = zstandard.ZstdDecompressor().decompressobj().decompress(raw)
        elif encoding == "br":
            import brotli

            data = brotli.decompress(raw)
    except Exception as exc:
        return {"_decode_error": str(exc), "_raw_bytes": len(raw)}
    try:
        return json.loads(data)
    except Exception:
        return {"_not_json": data[:2000].decode("utf-8", errors="replace"), "_bytes": len(data)}


def build_app(upstream: str, capture: Capture) -> FastAPI:
    app = FastAPI(title="codex probe")
    client = httpx.AsyncClient(timeout=None)

    @app.websocket("/{path:path}")
    async def ws_probe(websocket: WebSocket, path: str) -> None:
        capture.write(
            "websocket-attempt", path="/" + path, headers=_redact(dict(websocket.headers))
        )
        print(f"[probe] WS attempt on /{path} — logged, rejecting so Codex falls back to HTTPS")
        await websocket.close()  # never accepted -> client sees 403

    @app.api_route(
        "/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
    )
    async def relay(request: Request, path: str) -> Response:
        raw = await request.body()
        in_headers = dict(request.headers)
        n = capture.write(
            "request",
            method=request.method,
            path="/" + path,
            query=str(request.url.query),
            headers=_redact(in_headers),
            body=_body_for_log(raw, in_headers),
        )
        url = upstream.rstrip("/") + "/" + path
        if request.url.query:
            url += "?" + request.url.query
        fwd_headers = {k: v for k, v in in_headers.items() if k.lower() not in SKIP_FORWARD_HEADERS}

        upstream_request = client.build_request(
            request.method, url, headers=fwd_headers, content=raw
        )
        upstream_response = await client.send(upstream_request, stream=True)
        content_type = upstream_response.headers.get("content-type", "")
        status = upstream_response.status_code
        print(f"[probe] #{n} {request.method} /{path} -> {status} {content_type}")

        if "text/event-stream" in content_type:

            async def stream() -> Any:
                lines: list[str] = []
                try:
                    async for line in upstream_response.aiter_lines():
                        lines.append(line)
                        yield line + "\n"
                finally:
                    await upstream_response.aclose()
                    capture.write(
                        "response-sse",
                        request_n=n,
                        status=upstream_response.status_code,
                        headers=_redact(dict(upstream_response.headers)),
                        lines=lines,
                    )

            return StreamingResponse(
                stream(),
                status_code=upstream_response.status_code,
                media_type="text/event-stream",
            )

        body = await upstream_response.aread()
        await upstream_response.aclose()
        capture.write(
            "response",
            request_n=n,
            status=upstream_response.status_code,
            headers=_redact(dict(upstream_response.headers)),
            body=_body_for_log(body, dict(upstream_response.headers)),
        )
        out_headers = {
            k: v
            for k, v in upstream_response.headers.items()
            if k.lower() not in {"content-length", "content-encoding", "transfer-encoding"}
        }
        return Response(
            content=body, status_code=upstream_response.status_code, headers=out_headers
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--upstream", default="https://chatgpt.com/backend-api")
    parser.add_argument("--log", default="codex_probe.log")
    args = parser.parse_args()

    log_path = Path(args.log)
    print(f"[probe] forwarding verbatim to {args.upstream}")
    print(f"[probe] FULL traffic capture (auth redacted) -> {log_path.resolve()}")
    print(f'[probe] point Codex at it:  chatgpt_base_url = "http://127.0.0.1:{args.port}/"')
    uvicorn.run(build_app(args.upstream, Capture(log_path)), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
