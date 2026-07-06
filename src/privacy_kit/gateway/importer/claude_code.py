"""Parse Claude Code session transcripts (``~/.claude/projects/<slug>/*.jsonl``).

Each file is one session, one JSON object per line. Only ``type: user`` and
``type: assistant`` entries carry conversation text; everything else
(``system``, ``mode``, ``last-prompt``, ``attachment``, ``queue-operation``,
``file-history-snapshot``, …) is bookkeeping. Sidechain (subagent) and meta
entries are skipped, as are slash-command wrapper prompts and synthetic
assistant messages — the goal is the text a human typed and what the tools and
model returned, mirroring what the live proxy would have captured.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from privacy_kit.gateway.importer.base import (
    ParsedSession,
    ParsedTurn,
    parse_timestamp,
    truncate_title,
)

SOURCE = "claude-code-import"
WIRE_FORMAT = "anthropic"

# Prompts that are slash-command plumbing rather than human text.
_COMMAND_PREFIXES = ("<command-name>", "<local-command-caveat>", "<local-command-stdout>")

# Injected blocks that ride along inside a real prompt. They ARE imported
# (the live proxy forwards them too) but make useless preview titles.
_TITLE_SKIP_PREFIXES = ("<ide_opened_file>", "<system-reminder>")


def default_root() -> Path:
    return Path.home() / ".claude" / "projects"


def discover_sessions(
    root: Path | None = None,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    project: str | None = None,
) -> list[Path]:
    """Session files under ``root``, optionally filtered by project-slug
    substring and file mtime, oldest first."""
    base = root or default_root()
    if not base.is_dir():
        return []
    paths = []
    for path in base.glob("*/*.jsonl"):
        if project and project not in path.parent.name:
            continue
        if since is not None and path.stat().st_mtime < since.timestamp():
            continue
        if until is not None and path.stat().st_mtime > until.timestamp():
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: p.stat().st_mtime)


def _user_segments(content: object) -> tuple[list[tuple[str, str]], bool]:
    """Map a user entry's content to segments; second value is True when the
    entry contains a real human prompt (starts a new turn) rather than only
    tool results."""
    if isinstance(content, str):
        text = content.strip()
        if not text or text.startswith(_COMMAND_PREFIXES):
            return [], False
        return [("user", content)], True
    segments: list[tuple[str, str]] = []
    is_prompt = False
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text", "")
                if isinstance(text, str) and text.strip():
                    if text.strip().startswith(_COMMAND_PREFIXES):
                        continue
                    segments.append(("user", text))
                    is_prompt = True
            elif btype == "tool_result":
                inner = block.get("content")
                if isinstance(inner, str):
                    if inner.strip():
                        segments.append(("tool", inner))
                elif isinstance(inner, list):
                    for part in inner:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            if isinstance(text, str) and text.strip():
                                segments.append(("tool", text))
    return segments, is_prompt


def preview_info(path: Path) -> tuple[str | None, str | None]:
    """(title, project) for the import preview list, without a full parse.

    Title is the first real human prompt — same skip rules as parsing
    (meta/sidechain entries, command-wrapper prompts) via ``_user_segments``.
    The title is user text: callers must never log or persist it.
    """
    project = path.parent.name
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict) or entry.get("type") != "user":
                    continue
                if entry.get("isSidechain") or entry.get("isMeta"):
                    continue
                message = entry.get("message")
                if not isinstance(message, dict):
                    continue
                segments, is_prompt = _user_segments(message.get("content"))
                if not is_prompt:
                    continue
                for category, text in segments:
                    if category != "user":
                        continue
                    if text.lstrip().startswith(_TITLE_SKIP_PREFIXES):
                        continue
                    return truncate_title(text), project
    except OSError:
        pass
    return None, project


def parse_session(path: Path) -> ParsedSession | None:
    """Parse one session file; None when it yields no usable turns."""
    session_id = path.stem
    turns: list[ParsedTurn] = []
    current: ParsedTurn | None = None
    # Assistant entries can repeat (streamed snapshots share a message id), so
    # token usage is deduped by message id.
    seen_usage_ids: set[str] = set()

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            etype = entry.get("type")
            if etype not in ("user", "assistant"):
                continue
            if entry.get("isSidechain") or entry.get("isMeta"):
                continue
            message = entry.get("message")
            if not isinstance(message, dict):
                continue
            timestamp = parse_timestamp(entry.get("timestamp"))

            if etype == "user":
                segments, is_prompt = _user_segments(message.get("content"))
                if is_prompt:
                    current = ParsedTurn(timestamp=timestamp, segments=segments)
                    turns.append(current)
                elif segments:
                    if current is None:
                        current = ParsedTurn(timestamp=timestamp)
                        turns.append(current)
                    current.segments.extend(segments)
                continue

            # assistant
            if current is None:
                current = ParsedTurn(timestamp=timestamp)
                turns.append(current)
            model = message.get("model")
            if isinstance(model, str) and model and model != "<synthetic>":
                current.model = model
            elif model == "<synthetic>":
                continue
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if isinstance(text, str) and text.strip():
                        current.segments.append(("assistant", text))
            usage = message.get("usage")
            msg_id = message.get("id")
            if isinstance(usage, dict) and isinstance(msg_id, str) and msg_id not in seen_usage_ids:
                seen_usage_ids.add(msg_id)
                in_tok = usage.get("input_tokens") or 0
                out_tok = usage.get("output_tokens") or 0
                if in_tok or out_tok:
                    current.input_tokens = max(current.input_tokens or 0, int(in_tok))
                    current.output_tokens = (current.output_tokens or 0) + int(out_tok)

    turns = [t for t in turns if t.segments]
    if not turns:
        return None
    return ParsedSession(
        session_id=session_id, source=SOURCE, wire_format=WIRE_FORMAT, path=path, turns=turns
    )
