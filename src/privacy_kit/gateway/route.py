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
# Codex reads ``~/.codex/config.toml``. We route it through the gateway with a
# dedicated provider entry rather than a bare ``openai_base_url`` for one reason:
# Codex 0.139+ first tries a Responses *WebSocket* transport against the active
# provider and only falls back to HTTP after the WS connection fails — emitting a
# noisy "Falling back from WebSockets to HTTPS transport" warning every session,
# and attempting a transport the gateway does not speak (and, for PII filtering,
# must not: a WS the proxy can't sanitize would defeat the point). A provider
# with ``supports_websockets = false`` turns that off at the source, so every
# request is plain HTTP the gateway can sanitize — verified via ``codex doctor``
# ("Responses WebSocket is not enabled for the active provider").
#
# ``requires_openai_auth = true`` preserves BOTH auth modes: Codex still attaches
# the ChatGPT-account header + OAuth token (subscription) or the API key, so the
# gateway's subscription-vs-API-key routing (keyed on ``chatgpt-account-id``) is
# unchanged — confirmed by capturing Codex's outbound request. tomlkit round-trips
# the file, so user comments and unrelated keys survive the edit.

CODEX_CHATGPT_KEY = "chatgpt_base_url"
CODEX_OPENAI_KEY = "openai_base_url"
CODEX_PROVIDER_KEY = "model_provider"
CODEX_PROVIDERS_KEY = "model_providers"
CODEX_PROVIDER_ID = "privacy-kit"


def codex_config_path() -> Path:
    """Codex's user config file (read in both subscription and API-key mode)."""
    return Path.home() / ".codex" / "config.toml"


@dataclass(frozen=True)
class CodexRouteChange:
    """What was written to Codex's config, with enough context to undo it."""

    path: Path
    applied: dict[str, str]
    previous: dict[str, str | None]  # values before; None = key was absent
    removed: dict[str, str]  # stale gateway keys cleaned up (e.g. openai_base_url)


def _codex_provider_table(base_url: str) -> Any:
    """Build the ``[model_providers.privacy-kit]`` table pointing at the gateway."""
    import tomlkit

    table = tomlkit.table()
    table["name"] = CODEX_PROVIDER_ID
    table["base_url"] = base_url
    table["wire_api"] = "responses"  # the only wire_api Codex still supports
    table["requires_openai_auth"] = True  # keep ChatGPT/API-key auth attached
    table["supports_websockets"] = False  # the whole point: no WS attempt, no warning
    return table


def apply_codex_route(base_url: str, config_path: Path | None = None) -> CodexRouteChange:
    """Route Codex through the gateway with a WebSocket-free provider entry.

    Writes a ``[model_providers.privacy-kit]`` table (``base_url = {base}/v1``,
    ``supports_websockets = false``) and selects it with ``model_provider``. The
    scalar ``model_provider`` is inserted into the document root (never appended
    after a trailing ``[table]``, which TOML would read as a member of that table
    and break Codex's config load). Removes the gateway values an earlier version
    wrote to the root (``openai_base_url``/``chatgpt_base_url``); a user's own
    custom values are left alone. Comments and unrelated keys are preserved.
    Raises ``ValueError`` if the file exists but isn't valid TOML.
    """
    import tomlkit

    path = config_path or codex_config_path()
    doc = tomlkit.parse(path.read_text(encoding="utf-8")) if path.exists() else tomlkit.document()
    base = base_url.rstrip("/")
    api_base = base + "/v1"

    provider_base_key = f"{CODEX_PROVIDERS_KEY}.{CODEX_PROVIDER_ID}.base_url"
    providers = doc.get(CODEX_PROVIDERS_KEY)
    prev_provider_base: str | None = None
    if isinstance(providers, dict) and CODEX_PROVIDER_ID in providers:
        existing = providers[CODEX_PROVIDER_ID]
        if isinstance(existing, dict) and "base_url" in existing:
            prev_provider_base = str(existing["base_url"])
    previous: dict[str, str | None] = {
        CODEX_PROVIDER_KEY: str(doc[CODEX_PROVIDER_KEY]) if CODEX_PROVIDER_KEY in doc else None,
        provider_base_key: prev_provider_base,
    }

    # Clean up gateway values an earlier version wrote to the document root; a
    # user's own values (pointing elsewhere) are not gateway values, so hands off.
    removed: dict[str, str] = {}
    if CODEX_OPENAI_KEY in doc and str(doc[CODEX_OPENAI_KEY]) == api_base:
        removed[CODEX_OPENAI_KEY] = api_base
        del doc[CODEX_OPENAI_KEY]
    stale_chatgpt = base + "/"
    if CODEX_CHATGPT_KEY in doc and str(doc[CODEX_CHATGPT_KEY]) == stale_chatgpt:
        removed[CODEX_CHATGPT_KEY] = stale_chatgpt
        del doc[CODEX_CHATGPT_KEY]

    # Write (or replace) our dedicated provider table. Reusing the existing
    # ``model_providers`` super-table keeps any sibling providers the user defined.
    if not isinstance(doc.get(CODEX_PROVIDERS_KEY), dict):
        doc[CODEX_PROVIDERS_KEY] = tomlkit.table(is_super_table=True)
    doc[CODEX_PROVIDERS_KEY][CODEX_PROVIDER_ID] = _codex_provider_table(api_base)

    if CODEX_PROVIDER_KEY in doc:
        doc[CODEX_PROVIDER_KEY] = CODEX_PROVIDER_ID  # in place: keeps its position
        text = tomlkit.dumps(doc)
    else:
        # New scalar: prepend it so it lands in the root table, above any [table].
        quoted = tomlkit.item(CODEX_PROVIDER_ID).as_string()
        rest = tomlkit.dumps(doc)
        sep = "\n\n" if rest.strip() else "\n"
        text = f"{CODEX_PROVIDER_KEY} = {quoted}{sep}{rest}"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return CodexRouteChange(
        path=path,
        applied={CODEX_PROVIDER_KEY: CODEX_PROVIDER_ID, provider_base_key: api_base},
        previous=previous,
        removed=removed,
    )


def remove_codex_route(config_path: Path | None = None) -> dict[str, str]:
    """Remove the gateway override outright; returns the removed key -> value map.

    Cleans only what the current :func:`apply_codex_route` writes — the
    ``model_provider`` selection (when it still points at us) and our
    ``[model_providers.privacy-kit]`` table. A user's own ``model_provider``,
    sibling providers, ``chatgpt_base_url`` and ``openai_base_url`` are left
    untouched. Comments and unrelated keys survive the edit.
    """
    import tomlkit

    path = config_path or codex_config_path()
    if not path.exists():
        return {}
    doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    removed: dict[str, str] = {}

    if CODEX_PROVIDER_KEY in doc and str(doc[CODEX_PROVIDER_KEY]) == CODEX_PROVIDER_ID:
        removed[CODEX_PROVIDER_KEY] = CODEX_PROVIDER_ID
        del doc[CODEX_PROVIDER_KEY]

    providers = doc.get(CODEX_PROVIDERS_KEY)
    if isinstance(providers, dict) and CODEX_PROVIDER_ID in providers:
        table = providers[CODEX_PROVIDER_ID]
        base = table.get("base_url") if isinstance(table, dict) else None
        removed[f"{CODEX_PROVIDERS_KEY}.{CODEX_PROVIDER_ID}"] = str(base) if base else ""
        del providers[CODEX_PROVIDER_ID]
        if not providers:  # drop an empty model_providers super-table
            del doc[CODEX_PROVIDERS_KEY]

    if removed:
        path.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return removed
