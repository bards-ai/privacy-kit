"""Neutral parsed shapes shared by all history parsers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ParsedTurn:
    """One user→assistant exchange: the human prompt, that exchange's tool
    outputs, and the assistant reply, as (category, text) segments in order.
    Categories match ``interactiontext.category``: "user" / "tool" / "assistant".
    """

    timestamp: datetime | None
    segments: list[tuple[str, str]] = field(default_factory=list)
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class ParsedSession:
    """One session transcript file, parsed into turns."""

    session_id: str
    source: str
    wire_format: str
    path: Path
    turns: list[ParsedTurn] = field(default_factory=list)


def parse_timestamp(value: object) -> datetime | None:
    """Parse an ISO-8601 timestamp (``...Z`` or offset form) to aware UTC."""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
