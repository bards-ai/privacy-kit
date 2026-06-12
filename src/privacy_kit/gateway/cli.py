"""Command-line interface for the gateway.

``privacy-kit serve``   run the local gateway proxy + OTLP sink
``privacy-kit setup``   print the env/config to point a tool at the gateway
``privacy-kit report``  summarize the audit log
``privacy-kit scan``    one-off PII scan of a file or stdin
"""

from __future__ import annotations

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
        "(claude-code edits ~/.claude/settings.json; codex edits ~/.codex/config.toml).",
    ),
    remove: bool = typer.Option(
        False,
        "--remove",
        help="Remove a previously applied routing (claude-code and codex).",
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
    if (apply or remove) and tool == "cursor":
        typer.secho(
            "--apply/--remove support claude-code and codex; Cursor is configured in "
            "its own Settings UI — use the printed instructions below.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    settings = get_settings()
    base = f"http://{host or settings.host}:{port or settings.port}"

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
            f"# One setting routes both auth modes — Codex's model call follows\n"
            f"# openai_base_url whether you're signed in with a ChatGPT account\n"
            f"# (free/Plus/Pro, no API key) or an API key. A subscription request is\n"
            f"# recognized by the gateway and forwarded to chatgpt.com with your\n"
            f"# login token untouched (experimental).\n"
            f"export OPENAI_BASE_URL={base}/v1\n"
            f'# or in ~/.codex/config.toml:  openai_base_url = "{base}/v1"\n'
            f"\n{otel}\n"
        )
    # cursor
    return (
        "# Route Cursor's chat panel through privacy-kit (Settings → Models):\n"
        f"#   Override OpenAI Base URL:  {base}/v1\n"
        "#   OpenAI API Key:            <your OpenAI API key>\n"
        "# Note: only Cursor's chat/plan panel honors this; Composer, inline\n"
        "# edit, and autocomplete are locked to Cursor's own backend.\n"
    )


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

    from privacy_kit.core.detectors import BardsAiOnnxDetector

    detector = BardsAiOnnxDetector()

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


if __name__ == "__main__":
    app()
