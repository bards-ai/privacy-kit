"""Per-entity-type policy: which action each detected span gets.

The global ``PII_POLICY`` mode maps to a default action — ``monitor`` → keep
(forward unchanged, audit only), ``pseudonymize`` → reversible placeholder —
and ``PII_POLICY_OVERRIDES`` refines it per entity type. Keys are entity labels
(``PERSON_NAME``) or prefix wildcards (``SECRET_*``); values are actions:

- ``keep``: forward the original value (it is still detected and audited).
- ``redact``: one-way ``[REDACTED]`` — never rehydrated in the response.
- ``pseudonymize``: reversible ``[TYPE_N]`` placeholder via the request Vault.
- ``block``: refuse the whole request before anything reaches the upstream.

Overrides apply in both global modes, so ``{"SECRET_*": "block"}`` stops
credentials cold even when the gateway otherwise only monitors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from privacy_kit.core.detectors import Detector
from privacy_kit.core.types import Span
from privacy_kit.core.vault import Vault

if TYPE_CHECKING:
    from collections.abc import Mapping

    from privacy_kit.gateway.config import Settings

Action = Literal["keep", "redact", "pseudonymize", "block"]

REDACTED = "[REDACTED]"


class PolicyResolver:
    """Resolve an entity label to its action: exact match, longest wildcard
    prefix, then the global default."""

    def __init__(self, default: Action, overrides: Mapping[str, Action]) -> None:
        self.default = default
        self._exact: dict[str, Action] = {}
        self._prefixes: list[tuple[str, Action]] = []
        for pattern, action in overrides.items():
            if pattern.endswith("*"):
                self._prefixes.append((pattern[:-1], action))
            else:
                self._exact[pattern] = action
        self._prefixes.sort(key=lambda entry: -len(entry[0]))

    @classmethod
    def from_settings(cls, settings: Settings) -> PolicyResolver:
        default: Action = "keep" if settings.policy == "monitor" else "pseudonymize"
        return cls(default, settings.policy_overrides)

    def action_for(self, label: str) -> Action:
        exact = self._exact.get(label)
        if exact is not None:
            return exact
        for prefix, action in self._prefixes:
            if label.startswith(prefix):
                return action
        return self.default


def apply_policy(
    text: str,
    detector: Detector,
    vault: Vault,
    resolver: PolicyResolver,
    blocked: dict[str, int],
) -> str:
    """Policy-aware counterpart of :func:`~privacy_kit.core.vault.anonymize_into`.

    Runs detection once and applies each span's resolved action. ``block`` spans
    leave the text untouched but are tallied into ``blocked`` — the caller must
    refuse the request when that dict is non-empty, so the unmodified value can
    never actually leave. Only ``pseudonymize`` spans enter the ``vault`` (and
    therefore rehydrate); ``redact`` is one-way by construction.
    """
    spans = sorted(detector.detect(text), key=lambda s: (s.start, s.end))

    # Same overlap resolution as anonymize_into: earliest non-overlapping wins.
    chosen: list[Span] = []
    last_end = -1
    for span in spans:
        if span.start >= last_end and span.end > span.start:
            chosen.append(span)
            last_end = span.end

    # Left-to-right first so placeholder numbering follows reading order.
    for span in chosen:
        if resolver.action_for(span.label) == "pseudonymize":
            vault.placeholder_for(span.label, text[span.start : span.end])

    out = text
    for span in reversed(chosen):
        action = resolver.action_for(span.label)
        if action == "keep":
            continue
        if action == "block":
            blocked[span.label] = blocked.get(span.label, 0) + 1
            continue
        replacement = (
            REDACTED
            if action == "redact"
            else vault.placeholder_for(span.label, text[span.start : span.end])
        )
        out = out[: span.start] + replacement + out[span.end :]
    return out
