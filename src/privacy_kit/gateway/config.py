"""Central configuration for the gateway.

Loaded from environment variables (prefix ``PII_``) or a local ``.env`` file.
The shared knobs (``PII_MODEL_ID``, ``PII_MODEL_CACHE_DIR``, ``PII_THRESHOLD``)
use the same names the core library reads. Keep secrets out of source control —
``.env`` should be gitignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the gateway's detector, store, proxy, and OTLP sink."""

    model_config = SettingsConfigDict(env_prefix="PII_", env_file=".env", extra="ignore")

    # --- Detection ---
    model_id: str = "bardsai/eu-pii-anonimization-multilang"
    """HuggingFace model id for the on-device PII NER model."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Minimum per-entity confidence to treat a span as PII."""

    # --- Audit store ---
    db_path: Path = Path("privacy_kit.sqlite")
    """SQLite file for the audit log: metadata plus, per ``save_texts``,
    user-authored text and tool/file data segments (original + anonymized)
    in plaintext."""

    save_texts: Literal["anonymized", "all"] = "all"
    """Which eligible request text segments to save (original + anonymized,
    plaintext).  Only user-authored text and tool/file data are eligible; system
    prompts, instruction blocks, tool-call arguments, and assistant turns are
    never stored regardless of this setting.  Among eligible segments:
    "anonymized" = only those where PII was detected and replaced;
    "all" = every eligible segment."""

    # --- Policy ---
    policy: Literal["monitor", "pseudonymize"] = "monitor"
    """How the proxy treats detected PII before forwarding upstream.
    "monitor" (default): detect and log PII to the audit store, but forward the
    prompt unchanged — real values reach the upstream LLM. "pseudonymize": replace
    PII with reversible placeholders before forwarding, and rehydrate the response."""

    # --- Proxy ---
    host: str = "127.0.0.1"
    port: int = 8787
    anthropic_upstream: str = "https://api.anthropic.com"
    openai_upstream: str = "https://api.openai.com"
    chatgpt_upstream: str = "https://chatgpt.com/backend-api"
    """Upstream for Codex signed in with a ChatGPT account (no API key)."""

    # --- OTLP sink ---
    otel_downstream: str | None = None
    """If set, scrubbed OTLP payloads are re-exported to this collector base URL."""


def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
