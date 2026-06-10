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
) -> None:
    """Run the local gateway proxy. Loads the on-device model on startup."""
    import uvicorn

    from privacy_kit.gateway.config import get_settings
    from privacy_kit.gateway.proxy import build_default_app

    settings = get_settings()
    typer.echo("Loading PII model and starting the privacy-kit gateway…")
    uvicorn.run(build_default_app(), host=host or settings.host, port=port or settings.port)


@app.command()
def setup(
    tool: str = typer.Argument(..., help=f"One of: {', '.join(_SETUP_TOOLS)}"),
    host: str | None = typer.Option(None, help="Gateway host (default from settings)"),
    port: int | None = typer.Option(None, help="Gateway port (default from settings)"),
) -> None:
    """Print how to route a tool's traffic through the gateway."""
    from privacy_kit.gateway.config import get_settings

    if tool not in _SETUP_TOOLS:
        typer.secho(
            f"Unknown tool {tool!r}. Choose one of: {', '.join(_SETUP_TOOLS)}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    settings = get_settings()
    base = f"http://{host or settings.host}:{port or settings.port}"
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
            f"# Route Claude Code through privacy-kit (run `privacy-kit serve` first):\n"
            f"export ANTHROPIC_BASE_URL={base}\n"
            f"export ANTHROPIC_AUTH_TOKEN=<your Anthropic API key>\n"
            f"\n{otel}\n"
        )
    if tool == "codex":
        return (
            f"# Route Codex through privacy-kit (run `privacy-kit serve` first):\n"
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
