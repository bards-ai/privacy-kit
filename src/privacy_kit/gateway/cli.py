"""Command-line interface for the gateway.

``privacy-kit serve``   run the local gateway proxy + OTLP sink
``privacy-kit setup``   print the env/config to point a tool at the gateway
``privacy-kit report``  summarize the audit log
``privacy-kit scan``    one-off PII scan of a file or stdin
"""

from __future__ import annotations

import shlex
import shutil
import sys
from pathlib import Path

import typer

from privacy_kit import __version__

app = typer.Typer(
    name="privacy-kit",
    help="Local PII-filtering gateway for AI tools.",
    no_args_is_help=True,
    add_completion=False,
)

_SETUP_TOOLS = ("claude-code", "codex", "cursor")


@app.command()
def version() -> None:
    """Print the privacy-kit version."""
    typer.echo(__version__)


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind host (default from settings)"),
    port: int | None = typer.Option(None, help="Bind port (default from settings)"),
    route_tools: list[str] = typer.Option(
        [],
        "--route",
        help="Auto-route a tool through this gateway while it runs: writes the "
        "override into the tool's own config on startup and restores it on "
        "shutdown. Supported: claude-code.",
    ),
) -> None:
    """Run the local gateway proxy. Loads the on-device model on startup."""
    import uvicorn

    from privacy_kit.gateway import route
    from privacy_kit.gateway.config import get_settings
    from privacy_kit.gateway.proxy import build_default_app

    unsupported = [t for t in route_tools if t != "claude-code"]
    if unsupported:
        typer.secho(
            f"--route supports only claude-code for now (got: {', '.join(unsupported)}).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    settings = get_settings()
    bind_host = host or settings.host
    bind_port = port or settings.port

    changes: list[route.RouteChange] = []
    if "claude-code" in route_tools:
        base = f"http://{bind_host}:{bind_port}"
        try:
            change = route.apply_claude_code_route(base)
        except ValueError as exc:
            typer.secho(
                f"Could not apply the Claude Code route: {exc}", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(1) from exc
        changes.append(change)
        typer.secho(
            f"Routing Claude Code through this gateway: {route.ROUTE_KEY}={base}",
            fg=typer.colors.GREEN,
        )
        typer.echo(
            f"  Overrode {route.ROUTE_KEY} in {change.path} — new Claude Code sessions "
            "route through the gateway; the previous value is restored when this "
            "server shuts down."
        )

    if settings.policy == "monitor":
        typer.secho("Policy: monitor")
    else:
        typer.secho("Policy: pseudonymize")

    typer.echo("Loading PII model and starting the privacy-kit gateway…")
    typer.echo(f"PII preview UI: http://{bind_host}:{bind_port}/ui")
    try:
        uvicorn.run(build_default_app(), host=bind_host, port=bind_port)
    finally:
        for change in changes:
            if route.revert_route(change):
                typer.echo(f"Restored {route.ROUTE_KEY} in {change.path}.")


@app.command()
def setup(
    tool: str = typer.Argument(..., help=f"One of: {', '.join(_SETUP_TOOLS)}"),
    host: str | None = typer.Option(None, help="Gateway host (default from settings)"),
    port: int | None = typer.Option(None, help="Gateway port (default from settings)"),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Don't just print: write the routing into the tool's own config "
        "(claude-code edits ~/.claude/settings.json; codex edits ~/.codex/config.toml; "
        "cursor installs hooks in ~/.cursor/hooks.json).",
    ),
    remove: bool = typer.Option(
        False,
        "--remove",
        help="Remove a previously applied routing (claude-code, codex, and cursor).",
    ),
    scope: str = typer.Option(
        "user",
        "--scope",
        help="Cursor hooks scope: 'user' (~/.cursor/hooks.json) or 'project' "
        "(.cursor/hooks.json in the current directory). Ignored for other tools.",
    ),
    settings_file: Path | None = typer.Option(
        None, help="Override the tool settings file to edit (advanced/testing)."
    ),
) -> None:
    """Print — or apply — the routing of a tool's traffic through the gateway."""
    from privacy_kit.gateway import route
    from privacy_kit.gateway.config import get_settings

    if tool not in _SETUP_TOOLS:
        typer.secho(
            f"Unknown tool {tool!r}. Choose one of: {', '.join(_SETUP_TOOLS)}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)
    if apply and remove:
        typer.secho("--apply and --remove are mutually exclusive.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    settings = get_settings()
    base = f"http://{host or settings.host}:{port or settings.port}"

    if tool == "cursor":
        if scope not in ("user", "project"):
            typer.secho(
                f"Unknown --scope {scope!r} (use 'user' or 'project').",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)
        if apply:
            command = _cursor_hook_command()
            try:
                cursor_change = route.apply_cursor_hooks(command, scope=scope, path=settings_file)
            except ValueError as exc:
                typer.secho(f"Could not edit the hooks file: {exc}", fg=typer.colors.RED, err=True)
                raise typer.Exit(1) from exc
            typer.secho(
                f"Installed privacy-kit Cursor hooks in {cursor_change.path}.",
                fg=typer.colors.GREEN,
            )
            typer.echo(f"  Events: {', '.join(cursor_change.events)} (runs: {command} <event>)")
            typer.echo(
                "  These audit Composer/agent/inline-edit flows that bypass the gateway "
                "base URL. Keep `privacy-kit serve` running; set PII_CURSOR_BLOCK=1 to deny "
                "submissions that contain PII (hooks can't redact). Undo with: "
                f"privacy-kit setup cursor --remove --scope {scope}"
            )
            typer.echo("")
            typer.echo(_setup_text(tool, base))
            return
        if remove:
            cleaned = route.remove_cursor_hooks(scope=scope, path=settings_file)
            path = settings_file or route.cursor_hooks_path(scope)
            if not cleaned:
                typer.echo(f"No privacy-kit Cursor hooks found in {path}; nothing to do.")
            else:
                typer.secho(
                    f"Removed privacy-kit Cursor hooks ({', '.join(cleaned)}) from {path}.",
                    fg=typer.colors.GREEN,
                )
            return
        typer.echo(_setup_text(tool, base))
        return

    if remove:
        if tool == "codex":
            removed_keys = route.remove_codex_route(settings_file)
            codex_path = settings_file or route.codex_config_path()
            if not removed_keys:
                typer.echo(f"No gateway override found in {codex_path}; nothing to do.")
            else:
                for key, value in removed_keys.items():
                    typer.secho(f"Removed {key}={value} from {codex_path}.", fg=typer.colors.GREEN)
                typer.echo("New Codex sessions talk to OpenAI directly again.")
            return
        removed = route.remove_claude_code_route(settings_file)
        path = settings_file or route.claude_settings_path()
        if removed is None:
            typer.echo(f"No {route.ROUTE_KEY} override found in {path}; nothing to do.")
        else:
            typer.secho(f"Removed {route.ROUTE_KEY}={removed} from {path}.", fg=typer.colors.GREEN)
            typer.echo("New Claude Code sessions talk to Anthropic directly again.")
        return

    if apply:
        if tool == "codex":
            try:
                codex_change = route.apply_codex_route(base, settings_file)
            except ValueError as exc:
                typer.secho(f"Could not edit the config file: {exc}", fg=typer.colors.RED, err=True)
                raise typer.Exit(1) from exc
            for key, value in codex_change.applied.items():
                typer.secho(f"Wrote {key}={value} to {codex_change.path}.", fg=typer.colors.GREEN)
                prev = codex_change.previous[key]
                if prev is not None and prev != value:
                    typer.echo(f"  (replaced previous value: {prev})")
            for key, value in codex_change.removed.items():
                typer.echo(f"  Removed stale {key}={value} (left by an earlier version).")
            typer.echo(
                "New Codex sessions now route through the gateway — works for both "
                "ChatGPT-subscription and API-key logins. Keep `privacy-kit serve` "
                "running. Undo with: privacy-kit setup codex --remove"
            )
            return
        try:
            change = route.apply_claude_code_route(base, settings_file)
        except ValueError as exc:
            typer.secho(f"Could not edit the settings file: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from exc
        typer.secho(f"Wrote {route.ROUTE_KEY}={base} to {change.path}.", fg=typer.colors.GREEN)
        if change.previous is not None:
            typer.echo(f"  (replaced previous value: {change.previous})")
        typer.echo(
            "New Claude Code sessions now route through the gateway — keep "
            "`privacy-kit serve` running. Undo with: privacy-kit setup claude-code --remove"
        )
        return

    typer.echo(_setup_text(tool, base))


def _setup_text(tool: str, base: str) -> str:
    otel = (
        "# Optional: route telemetry logs through the gateway too\n"
        "export OTEL_LOGS_EXPORTER=otlp\n"
        "export OTEL_EXPORTER_OTLP_PROTOCOL=http/json\n"
        f"export OTEL_EXPORTER_OTLP_ENDPOINT={base}\n"
        "export OTEL_LOG_USER_PROMPTS=1"
    )
    if tool == "claude-code":
        return (
            f"# Route Claude Code through privacy-kit (run `privacy-kit serve` first).\n"
            f"# No manual export needed — apply it automatically (edits ~/.claude/settings.json):\n"
            f"#   privacy-kit setup claude-code --apply     # persistent (undo: --remove)\n"
            f"#   privacy-kit serve --route claude-code     # only while the gateway runs\n"
            f"# Or by hand:\n"
            f"export ANTHROPIC_BASE_URL={base}\n"
            f"\n"
            f"# Subscription (Max/Pro) — no API key needed. The gateway forwards your\n"
            f"# Claude login token and preserves the Claude Code system identifier so\n"
            f"# Anthropic still accepts the OAuth request. If your Claude Code version\n"
            f"# won't send the subscription token to a custom base URL, mint a\n"
            f"# subscription-backed token once and it's picked up automatically:\n"
            f"#   claude setup-token\n"
            f"\n"
            f"# Only if you authenticate with an API key instead of a subscription:\n"
            f"#   export ANTHROPIC_AUTH_TOKEN=<your Anthropic API key>\n"
            f"\n{otel}\n"
        )
    if tool == "codex":
        return (
            f"# Route Codex through privacy-kit (run `privacy-kit serve` first).\n"
            f"# Apply automatically (edits ~/.codex/config.toml):\n"
            f"#   privacy-kit setup codex --apply     # undo with: --remove\n"
            f"#\n"
            f"# This adds a dedicated provider rather than a bare openai_base_url:\n"
            f"# Codex 0.139+ tries a Responses WebSocket transport first and only\n"
            f"# falls back to HTTP after it fails (the 'Falling back from WebSockets'\n"
            f"# warning). supports_websockets = false skips that — every request is\n"
            f"# HTTP the gateway can sanitize. requires_openai_auth keeps both auth\n"
            f"# modes working: a ChatGPT-subscription request (free/Plus/Pro, no API\n"
            f"# key) is recognized by the gateway and forwarded to chatgpt.com with\n"
            f"# your login token untouched (experimental); an API key works too.\n"
            f"#\n"
            f"# Or by hand in ~/.codex/config.toml:\n"
            f'#   model_provider = "privacy-kit"\n'
            f"#   [model_providers.privacy-kit]\n"
            f'#   name = "privacy-kit"\n'
            f'#   base_url = "{base}/v1"\n'
            f'#   wire_api = "responses"\n'
            f"#   requires_openai_auth = true\n"
            f"#   supports_websockets = false\n"
            f"\n{otel}\n"
        )
    # cursor
    return (
        "# Cursor needs two layers — its surfaces sit on two backends.\n"
        "#\n"
        "# 1. Chat/plan panel — the ONE surface the gateway can pseudonymize.\n"
        "#    Settings → Models → Override OpenAI Base URL:\n"
        f"#      Override OpenAI Base URL:  {base}/v1\n"
        "#      OpenAI API Key:            <your OpenAI API key>\n"
        "#    Run the gateway with PII_POLICY=pseudonymize and its prompts are\n"
        "#    redacted and rehydrated, exactly like Claude Code / Codex.\n"
        "#\n"
        "# 2. Composer, the agent loop, inline edit (Cmd+K), Apply, and Tab stay on\n"
        "#    Cursor's own backend and bypass the base URL — they can't be redacted.\n"
        "#    Cover them with Cursor hooks (audit; optional block on PII):\n"
        "#      privacy-kit setup cursor --apply                  # ~/.cursor/hooks.json\n"
        "#      privacy-kit setup cursor --apply --scope project  # .cursor/hooks.json\n"
        "#      PII_CURSOR_BLOCK=1 privacy-kit serve              # deny prompts with PII\n"
    )


def _cursor_hook_command() -> str:
    """The base command Cursor should run for our hooks (event name appended).

    Resolves to an absolute path at apply time so it works regardless of the PATH
    Cursor spawns hooks with. Falls back to ``python -m privacy_kit`` when the
    console script isn't found on PATH.
    """
    exe = shutil.which("privacy-kit")
    if exe:
        return f"{shlex.quote(exe)} hook cursor"
    return f"{shlex.quote(sys.executable)} -m privacy_kit hook cursor"


@app.command()
def report(
    db: Path | None = typer.Option(None, help="Audit DB path (default from settings)"),
    limit: int = typer.Option(10, help="How many recent interactions to show"),
) -> None:
    """Summarize the audit log."""
    from privacy_kit.gateway.config import get_settings
    from privacy_kit.gateway.store import AuditStore

    db_path = db or get_settings().db_path
    store = AuditStore(db_path)
    summary = store.summary()

    typer.echo(f"privacy-kit audit — {db_path}")
    typer.echo(f"Interactions: {summary['interactions']}")
    typer.echo(f"PII entities: {summary['entities_total']}")
    by_type = summary["entities_by_type"]
    if by_type:
        typer.echo("By type:")
        for etype, count in sorted(by_type.items(), key=lambda kv: -kv[1]):
            typer.echo(f"  {etype:<28} {count}")

    recent = store.recent(limit=limit)
    if recent:
        typer.echo(f"Recent ({len(recent)}):")
        for row in recent:
            stamp = row.created_at.strftime("%Y-%m-%d %H:%M:%S")
            typer.echo(
                f"  {stamp}  {row.source:<14} {row.wire_format:<18} "
                f"{row.model:<22} entities={row.entity_total}"
            )


@app.command()
def scan(
    path: str = typer.Argument("-", help="File to scan, or - for stdin"),
    anonymize: bool = typer.Option(
        False, "--anonymize", help="Print the anonymized text instead of a span list"
    ),
) -> None:
    """One-off PII scan of a file or stdin (loads the on-device model)."""
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")

    from privacy_kit.core.detectors import build_detector

    detector = build_detector()

    if anonymize:
        from privacy_kit.core.vault import anonymize as anonymize_text

        clean, vault = anonymize_text(text, detector)
        typer.echo(clean)
        typer.secho(f"\n# {len(vault)} value(s) pseudonymized", fg=typer.colors.GREEN, err=True)
        return

    spans = detector.detect(text)
    if not spans:
        typer.secho("No PII detected.", fg=typer.colors.GREEN)
        return
    typer.echo(f"Detected {len(spans)} PII span(s):")
    for span in spans:
        snippet = span.text_of(text).replace("\n", " ")
        typer.echo(f"  {span.label:<28} {span.score:.2f}  {snippet!r}")


@app.command()
def hook(
    tool: str = typer.Argument(..., help="Hook source tool (only 'cursor')."),
    event: str = typer.Argument(..., help="Cursor hook event name, e.g. beforeSubmitPrompt."),
) -> None:
    """Cursor hook handler: scan one hook event via the running gateway.

    Reads the hook JSON from stdin, forwards it to the local gateway's
    /v1/cursor-hook, and prints the gateway's decision to stdout. Fails OPEN — if
    the gateway isn't running (or anything goes wrong), it allows the action so an
    absent privacy-kit never blocks Cursor. Installed by `setup cursor --apply`.
    """
    import json

    raw = sys.stdin.read()
    # The shape Cursor expects when the action is permitted, by event kind.
    if event == "beforeReadFile":
        allow = '{"permission": "allow"}'
    elif event == "afterAgentResponse":
        allow = "{}"
    else:
        allow = '{"continue": true}'
    if tool != "cursor":
        typer.echo(allow)
        return
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except ValueError:
        payload = {}
    if isinstance(payload, dict):
        payload.setdefault("hook_event_name", event)

    from privacy_kit.gateway.config import get_settings

    settings = get_settings()
    url = f"http://{settings.host}:{settings.port}/v1/cursor-hook"
    try:
        import httpx

        response = httpx.post(url, json=payload, timeout=15.0)
        typer.echo(response.text if response.status_code == 200 else allow)
    except Exception:
        typer.echo(allow)  # gateway down / unreachable: never wedge Cursor


if __name__ == "__main__":
    app()
