"""Automatic tool routing.

Exporting ``ANTHROPIC_BASE_URL`` by hand is easy to forget and impossible for a
child process to do into the parent shell. Claude Code, however, applies the
``env`` block of ``~/.claude/settings.json`` to every new session — so routing
it through the gateway is one JSON edit. This module owns that edit:

* :func:`apply_claude_code_route` writes the override and remembers what was
  there before;
* :func:`revert_route` restores the previous state, but only if the value is
  still the one we wrote (a manual user edit is never clobbered);
* :func:`remove_claude_code_route` is the explicit off switch.

Used by ``privacy-kit setup claude-code --apply/--remove`` (persistent) and
``privacy-kit serve --route claude-code`` (applied on startup, reverted on
shutdown).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROUTE_KEY = "ANTHROPIC_BASE_URL"


def claude_settings_path() -> Path:
    """Claude Code's user settings file (the ``env`` block is applied per session)."""
    return Path.home() / ".claude" / "settings.json"


@dataclass(frozen=True)
class RouteChange:
    """What was written where, with enough context to undo it safely."""

    path: Path
    applied: str
    previous: str | None  # value before we touched it; None = key was absent


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    data = json.loads(text)  # raises ValueError on corrupt JSON — never clobber it
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def apply_claude_code_route(base_url: str, settings_path: Path | None = None) -> RouteChange:
    """Write ``ANTHROPIC_BASE_URL`` into Claude Code's settings ``env`` block.

    Everything else in the file is preserved. Returns the change record needed
    to revert. Raises ``ValueError`` if the file exists but isn't valid JSON.
    """
    path = settings_path or claude_settings_path()
    data = _load(path)
    env = data.setdefault("env", {})
    if not isinstance(env, dict):
        raise ValueError(f"'env' in {path} is not a JSON object")
    previous = env.get(ROUTE_KEY)
    if previous is not None and not isinstance(previous, str):
        raise ValueError(f"'env.{ROUTE_KEY}' in {path} is not a string")
    env[ROUTE_KEY] = base_url
    _save(path, data)
    return RouteChange(path=path, applied=base_url, previous=previous)


def revert_route(change: RouteChange) -> bool:
    """Undo a previous :func:`apply_claude_code_route`.

    Restores the prior value (or removes the key if there was none). If the
    current value is no longer the one we wrote — the user edited it while the
    gateway was running — it is left alone. Returns True if anything changed.
    """
    try:
        data = _load(change.path)
    except ValueError:
        return False  # file was hand-edited into something unreadable; hands off
    env = data.get("env")
    if not isinstance(env, dict) or env.get(ROUTE_KEY) != change.applied:
        return False
    if change.previous is None:
        env.pop(ROUTE_KEY, None)
        if not env:
            data.pop("env", None)
    else:
        env[ROUTE_KEY] = change.previous
    _save(change.path, data)
    return True


def remove_claude_code_route(settings_path: Path | None = None) -> str | None:
    """Remove the override outright; returns the removed value, if any."""
    path = settings_path or claude_settings_path()
    data = _load(path)
    env = data.get("env")
    if not isinstance(env, dict) or ROUTE_KEY not in env:
        return None
    removed = env.pop(ROUTE_KEY)
    if not env:
        data.pop("env", None)
    _save(path, data)
    return str(removed)
