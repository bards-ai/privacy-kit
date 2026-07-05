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
    detector: Literal["local", "null"] = "local"
    """Which PII detector to run. "local" loads the on-device model. "null"
    detects nothing — requests pass through and conversations are still saved,
    but no PII is found (nothing is pseudonymized, no entity counts recorded).
    Use "null" to exercise conversation saving without the model download; pair
    it with ``save_texts="all"`` (the default) so segments are still stored."""

    model_id: str = "bardsai/eu-pii-anonimization-multilang"
    """HuggingFace model id for the on-device PII NER model."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Minimum per-entity confidence to treat a span as PII."""

    # --- Audit store ---
    db_path: Path = Field(default_factory=lambda: Path.home() / ".privacy-kit" / "audit.sqlite")
    """SQLite file for the audit log: metadata plus, per ``save_texts``,
    user-authored text and tool/file data segments (original + anonymized)
    in plaintext. Defaults to a stable per-user location
    (``~/.privacy-kit/audit.sqlite``) so the gateway reuses one audit log
    regardless of the working directory it is launched from; the file is
    created ``0o600``. Override with ``PII_DB_PATH``."""

    save_texts: Literal["anonymized", "all"] = "all"
    """Which text segments to save (original + anonymized, plaintext).  System
    prompts, instruction blocks, and tool-call arguments are never stored
    regardless of this setting.  "anonymized" = only user/tool segments where
    PII was detected and replaced (the assistant turn is stored only when the
    turn had PII).  "all" = every user/tool segment plus the assistant turn,
    regardless of whether PII was found — this is what lets full conversations
    save when the detector is "null"."""

    # --- Policy ---
    policy: Literal["monitor", "pseudonymize"] = "monitor"
    """How the proxy treats detected PII before forwarding upstream.
    "monitor" (default): detect and log PII to the audit store, but forward the
    prompt unchanged — real values reach the upstream LLM. "pseudonymize": replace
    PII with reversible placeholders before forwarding, and rehydrate the response."""

    policy_overrides: dict[str, Literal["keep", "redact", "pseudonymize", "block"]] = {}
    """Per-entity-type actions layered over ``policy``. Keys are entity labels
    (``PERSON_NAME``) or prefix wildcards (``SECRET_*``); values: "keep" (forward
    the original, audit only), "redact" (one-way ``[REDACTED]``, never
    rehydrated), "pseudonymize" (reversible ``[TYPE_N]``), "block" (refuse the
    request with a 403 before anything is forwarded). Overrides apply in both
    global modes — ``PII_POLICY_OVERRIDES='{"SECRET_*": "block"}'`` stops
    credentials even in monitor mode. Exact label beats wildcard; the longest
    wildcard prefix wins."""

    # --- Cursor hooks ---
    cursor_block: bool = False
    """When a Cursor hook (beforeSubmitPrompt/beforeReadFile) detects PII, deny the
    action instead of only auditing it. Cursor hooks cannot rewrite text, so this is
    the only way to stop PII reaching Cursor's backend from the surfaces that bypass
    the gateway base URL (Composer, the agent loop, inline edit, Tab). Off by default
    — monitor only: detect, record, and always allow."""

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

    # --- Web dashboard API ---
    cors_origins: list[str] = []
    """Browser origins allowed to call the JSON API directly (CORS). Empty by
    default: the bundled dashboard proxies requests server-side, so the browser
    is same-origin and no CORS is needed. Set ``PII_CORS_ORIGINS`` (a JSON array,
    e.g. ``["http://localhost:3000"]``) only when serving the frontend from a
    different origin during host development."""

    expose_plaintext: bool = True
    """Whether the dashboard API may return the raw ``original`` text held in
    ``interactiontext``. True (default) preserves the local single-user
    experience. Set ``PII_EXPOSE_PLAINTEXT=false`` for a hosted/multi-user
    deployment so originals are redacted in API responses."""


def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
