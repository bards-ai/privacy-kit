"""Vendored secret-detection rules.

Adapted from the gitleaks ruleset (https://github.com/gitleaks/gitleaks,
MIT license — Copyright (c) 2019 Zachary Rice) as a curated, tested, in-tree
table: each entry records the upstream gitleaks rule id it derives from, so
refreshing against a newer gitleaks release is a mechanical diff
(``scripts/sync_secret_rules.py`` fetches the upstream TOML for comparison).
Rules privacy-kit adds itself carry the ``privacy-kit-`` id prefix.

These rules are deterministic (regex + optional Shannon-entropy floor): no
model, no network. They target the category the NER model structurally cannot
cover — format-defined, high-entropy credentials — which is exactly what AI
coding tools leak when they read ``.env`` files, configs, and keys.

Span labels use the ``SECRET_`` prefix so downstream policy can treat all
credentials as one class (e.g. block secrets while pseudonymizing names).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["SECRET_RULES", "SecretRule"]


@dataclass(frozen=True, slots=True)
class SecretRule:
    """One deterministic secret pattern.

    ``secret_group`` is the regex group holding the credential itself; group 0
    (the default) means the whole match. When a rule anchors on surrounding
    context (``password = "..."``), pointing ``secret_group`` at the value
    keeps the replacement surgical: the LLM still sees ``password = [SECRET_…]``.
    ``min_entropy`` (bits/char, Shannon) filters low-entropy false positives on
    the broad generic rules; 0.0 disables the check.
    """

    rule_id: str
    label: str
    pattern: re.Pattern[str]
    secret_group: int = 0
    min_entropy: float = 0.0


def _rule(
    rule_id: str,
    label: str,
    pattern: str,
    *,
    secret_group: int = 0,
    min_entropy: float = 0.0,
    flags: int = 0,
) -> SecretRule:
    return SecretRule(rule_id, label, re.compile(pattern, flags), secret_group, min_entropy)


SECRET_RULES: tuple[SecretRule, ...] = (
    # --- Cloud / platform keys (fixed prefixes, safe to match bare) ---
    _rule(
        "aws-access-token",
        "SECRET_AWS_ACCESS_KEY",
        r"\b(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}\b",
    ),
    _rule("gcp-api-key", "SECRET_GCP_API_KEY", r"\bAIza[0-9A-Za-z\-_]{35}\b"),
    _rule(
        "github-pat",
        "SECRET_GITHUB_TOKEN",
        r"\b(?:ghp|gho|ghu|ghs|ghr)_[0-9A-Za-z]{36}\b",
    ),
    _rule(
        "github-fine-grained-pat",
        "SECRET_GITHUB_TOKEN",
        r"\bgithub_pat_[0-9A-Za-z_]{82}\b",
    ),
    _rule("gitlab-pat", "SECRET_GITLAB_TOKEN", r"\bglpat-[0-9A-Za-z_\-]{20}\b"),
    _rule(
        "openai-api-key",
        "SECRET_OPENAI_API_KEY",
        r"\bsk-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}\b"
        r"|\bsk-(?:proj|svcacct|admin)-[A-Za-z0-9_\-]{40,}\b",
    ),
    _rule(
        "anthropic-api-key",
        "SECRET_ANTHROPIC_API_KEY",
        r"\bsk-ant-[a-z0-9]{2,10}-[A-Za-z0-9_\-]{80,}\b",
    ),
    _rule(
        "huggingface-access-token",
        "SECRET_HUGGINGFACE_TOKEN",
        r"\bhf_[A-Za-z]{34}\b",
    ),
    _rule("slack-token", "SECRET_SLACK_TOKEN", r"\bxox[baprs]-[0-9A-Za-z\-]{10,}\b"),
    _rule(
        "stripe-access-token",
        "SECRET_STRIPE_KEY",
        r"\b[sr]k_(?:test|live|prod)_[0-9A-Za-z]{10,99}\b",
    ),
    _rule(
        "sendgrid-api-token",
        "SECRET_SENDGRID_KEY",
        r"\bSG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}\b",
    ),
    _rule("npm-access-token", "SECRET_NPM_TOKEN", r"\bnpm_[0-9A-Za-z]{36}\b"),
    _rule(
        "pypi-upload-token",
        "SECRET_PYPI_TOKEN",
        r"\bpypi-AgEIcHlwaS5vcmc[A-Za-z0-9_\-]{50,}\b",
    ),
    # --- Structured credentials ---
    _rule(
        "jwt",
        "SECRET_JWT",
        r"\bey[A-Za-z0-9]{15,}\.ey[A-Za-z0-9/_\-]{15,}\.[A-Za-z0-9/_\-]{10,}={0,2}\b",
    ),
    _rule(
        "private-key",
        "SECRET_PRIVATE_KEY",
        r"-----BEGIN[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?-----"
        r"[A-Za-z0-9+/=\s]*?"
        r"-----END[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?-----",
    ),
    _rule(
        # privacy-kit addition: credentials embedded in connection URLs.
        "privacy-kit-connection-string",
        "SECRET_CONNECTION_STRING",
        r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqps?|mssql)://"
        r"[^\s:@/]+:([^\s@/]+)@",
        secret_group=1,
        flags=re.IGNORECASE,
    ),
    _rule(
        # privacy-kit addition: Authorization header values pasted into text.
        "privacy-kit-authorization-header",
        "SECRET_AUTH_HEADER",
        r"(?i)\bauthorization\s*[:=]\s*(?:bearer|basic|token)\s+([A-Za-z0-9._~+/=\-]{16,})",
        secret_group=1,
        min_entropy=3.0,
    ),
    # --- Generic assignment (gitleaks generic-api-key, narrowed) ---
    _rule(
        # The keyword may sit inside a larger identifier (AWS_SECRET_ACCESS_KEY,
        # STRIPE_API_KEY_LIVE), so it is framed by identifier affixes instead of
        # word boundaries. The entropy floor drops placeholder-ish values.
        "generic-api-key",
        "SECRET_GENERIC",
        r"(?i)\b[\w.-]{0,32}(?:api[_-]?key|apikey|secret|token|password|passwd|"
        r"credentials?)[\w.-]{0,16}"
        r"""\s*[:=]\s*['"]?([A-Za-z0-9+/_.=\-]{8,80})['"]?""",
        secret_group=1,
        min_entropy=3.5,
    ),
)
