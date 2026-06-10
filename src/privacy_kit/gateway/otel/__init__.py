"""OTLP log sink.

Receives OpenTelemetry logs over OTLP/HTTP (e.g. Claude Code telemetry with
``OTEL_LOG_USER_PROMPTS=1``), pseudonymizes PII in prompt attributes, stores to
the audit DB, and optionally re-exports sanitized logs downstream.
"""

from privacy_kit.gateway.otel.sink import register_otel_routes, scrub_otlp

__all__ = ["register_otel_routes", "scrub_otlp"]
