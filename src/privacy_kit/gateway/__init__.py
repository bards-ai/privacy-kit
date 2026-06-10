"""The privacy-kit gateway — a local PII-filtering proxy for AI tools.

For tools you can't import a library into (Claude Code, Codex, Cursor), the
gateway sits in the network path via their ``*_BASE_URL`` overrides: it
pseudonymizes prompt text with a :class:`~privacy_kit.core.vault.Vault` before
it leaves the machine, forwards the sanitized request to the real upstream,
rehydrates the response, and records a metadata-only audit row. An OTLP sink
scrubs telemetry the same way (one-way).

Requires the ``gateway`` extra: ``pip install 'privacy-kit[gateway]'``.
"""

from privacy_kit.gateway.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
