"""Parse Codex rollout files (``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``).

The first line is ``session_meta`` (carries the session id). Human text comes
from ``event_msg``/``user_message`` — the ``response_item`` user messages also
contain injected ``<user_instructions>``/``<environment_context>`` wrappers, so
they are skipped in favor of the event stream. Assistant text is the
``response_item`` assistant message's ``output_text`` blocks (the duplicate
``event_msg``/``agent_message`` is ignored); tool output is
``function_call_output``. ``reasoning`` items are encrypted and skipped;
``function_call`` items are tool *arguments*, which this store never persists.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from privacy_kit.gateway.importer.base import (
    ParsedSession,
    ParsedTurn,
    parse_timestamp,
    truncate_title,
)

SOURCE = "codex-import"
WIRE_FORMAT = "openai_responses"


def default_root() -> Path:
    home = os.environ.get("CODEX_HOME")  # Codex's own relocation variable
    return (Path(home) if home else Path.home() / ".codex") / "sessions"


def discover_sessions(
    root: Path | None = None,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    project: str | None = None,  # accepted for interface parity; Codex files aren't per-project
) -> list[Path]:
    """Rollout files under ``root``, optionally filtered by mtime, oldest first."""
    base = root or default_root()
    if not base.is_dir():
        return []
    paths = []
    for path in base.rglob("rollout-*.jsonl"):
        if since is not None and path.stat().st_mtime < since.timestamp():
            continue
        if until is not None and path.stat().st_mtime > until.timestamp():
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: p.stat().st_mtime)


def preview_info(path: Path) -> tuple[str | None, str | None]:
    """(title, project) for the import preview list, in one partial scan.

    Project is ``session_meta``'s cwd; title is the first
    ``event_msg``/``user_message`` (the ``response_item`` user copies carry
    injected ``<user_instructions>``/``<environment_context>`` wrappers — same
    preference as ``parse_session``). The title is user text: callers must
    never log or persist it.
    """
    project: str | None = None
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                payload = entry.get("payload")
                if not isinstance(payload, dict):
                    continue
                etype = entry.get("type")
                if etype == "session_meta":
                    cwd = payload.get("cwd")
                    if isinstance(cwd, str) and cwd:
                        project = cwd
                elif etype == "event_msg" and payload.get("type") == "user_message":
                    text = payload.get("message")
                    if isinstance(text, str) and text.strip():
                        return truncate_title(text), project
    except OSError:
        pass
    return None, project


def parse_session(path: Path) -> ParsedSession | None:
    """Parse one rollout file; None when it has no id or no usable turns."""
    session_id: str | None = None
    model: str | None = None
    turns: list[ParsedTurn] = []
    current: ParsedTurn | None = None

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            etype = entry.get("type")
            payload = entry.get("payload")
            if not isinstance(payload, dict):
                continue

            if etype == "session_meta":
                sid = payload.get("id") or payload.get("session_id")
                if isinstance(sid, str) and sid:
                    session_id = sid
                continue

            if etype == "turn_context":
                m = payload.get("model")
                if isinstance(m, str) and m:
                    model = m
                continue

            timestamp = parse_timestamp(entry.get("timestamp"))

            if etype == "event_msg":
                ptype = payload.get("type")
                if ptype == "user_message":
                    text = payload.get("message")
                    if isinstance(text, str) and text.strip():
                        current = ParsedTurn(timestamp=timestamp, segments=[("user", text)])
                        turns.append(current)
                elif ptype == "token_count":
                    usage = (payload.get("info") or {}).get("last_token_usage")
                    if isinstance(usage, dict) and current is not None:
                        in_tok = usage.get("input_tokens") or 0
                        out_tok = usage.get("output_tokens") or 0
                        current.input_tokens = max(current.input_tokens or 0, int(in_tok))
                        current.output_tokens = (current.output_tokens or 0) + int(out_tok)
                continue

            if etype != "response_item":
                continue
            ptype = payload.get("type")
            if ptype == "message" and payload.get("role") == "assistant":
                if current is None:
                    current = ParsedTurn(timestamp=timestamp)
                    turns.append(current)
                for block in payload.get("content") or []:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        text = block.get("text", "")
                        if isinstance(text, str) and text.strip():
                            current.segments.append(("assistant", text))
            elif ptype == "function_call_output":
                output = payload.get("output")
                if isinstance(output, str) and output.strip():
                    if current is None:
                        current = ParsedTurn(timestamp=timestamp)
                        turns.append(current)
                    current.segments.append(("tool", output))

    turns = [t for t in turns if t.segments]
    if session_id is None or not turns:
        return None
    for turn in turns:
        turn.model = model
    return ParsedSession(
        session_id=session_id, source=SOURCE, wire_format=WIRE_FORMAT, path=path, turns=turns
    )
