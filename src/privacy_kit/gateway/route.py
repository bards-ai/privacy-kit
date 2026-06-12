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


# --- Codex --------------------------------------------------------------------
#
# Codex reads ``~/.codex/config.toml``. The model call follows ``openai_base_url``
# in BOTH auth modes (ChatGPT subscription and API key) — captured traffic shows
# ``chatgpt_base_url`` governs only Codex's account/management endpoints, not the
# model call. So routing the gateway needs exactly one key: ``openai_base_url``.
# The gateway itself then forwards a subscription request (recognized by its
# ``chatgpt-account-id`` header) to the ChatGPT backend. tomlkit round-trips the
# file, so user comments and unrelated keys survive the edit.

CODEX_CHATGPT_KEY = "chatgpt_base_url"
CODEX_OPENAI_KEY = "openai_base_url"


def codex_config_path() -> Path:
    """Codex's user config file (read in both subscription and API-key mode)."""
    return Path.home() / ".codex" / "config.toml"


@dataclass(frozen=True)
class CodexRouteChange:
    """What was written to Codex's config, with enough context to undo it."""

    path: Path
    applied: dict[str, str]
    previous: dict[str, str | None]  # values before; None = key was absent
    removed: dict[str, str]  # stale gateway keys cleaned up (e.g. chatgpt_base_url)


def apply_codex_route(base_url: str, config_path: Path | None = None) -> CodexRouteChange:
    """Point Codex's ``openai_base_url`` at the gateway.

    Writes ``openai_base_url = {base}/v1`` into Codex's ``config.toml``,
    inserted into the document's root (never appended after a trailing
    ``[table]``, which TOML would read as a member of that table and break
    Codex's config load). Removes a stale ``chatgpt_base_url`` left pointing at
    this gateway by an earlier version; a user's own custom value is left alone.
    Comments and unrelated keys are preserved. Raises ``ValueError`` if the file
    exists but isn't valid TOML.
    """
    import tomlkit

    path = config_path or codex_config_path()
    doc = tomlkit.parse(path.read_text(encoding="utf-8")) if path.exists() else tomlkit.document()
    base = base_url.rstrip("/")
    value = base + "/v1"

    previous = {CODEX_OPENAI_KEY: str(doc[CODEX_OPENAI_KEY]) if CODEX_OPENAI_KEY in doc else None}

    removed: dict[str, str] = {}
    stale_chatgpt = base + "/"
    if CODEX_CHATGPT_KEY in doc and str(doc[CODEX_CHATGPT_KEY]) == stale_chatgpt:
        removed[CODEX_CHATGPT_KEY] = stale_chatgpt
        del doc[CODEX_CHATGPT_KEY]

    if CODEX_OPENAI_KEY in doc:
        doc[CODEX_OPENAI_KEY] = value  # in place: keeps its existing position
        text = tomlkit.dumps(doc)
    else:
        # New key: prepend it so it lands in the root table, above any [table].
        quoted = tomlkit.item(value).as_string()
        text = f"{CODEX_OPENAI_KEY} = {quoted}\n" + tomlkit.dumps(doc)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return CodexRouteChange(
        path=path, applied={CODEX_OPENAI_KEY: value}, previous=previous, removed=removed
    )


def remove_codex_route(config_path: Path | None = None) -> dict[str, str]:
    """Remove the gateway overrides outright; returns the removed key -> value map."""
    import tomlkit

    path = config_path or codex_config_path()
    if not path.exists():
        return {}
    doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    removed: dict[str, str] = {}
    for key in (CODEX_CHATGPT_KEY, CODEX_OPENAI_KEY):
        if key in doc:
            removed[key] = str(doc[key])
            del doc[key]
    if removed:
        path.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return removed
