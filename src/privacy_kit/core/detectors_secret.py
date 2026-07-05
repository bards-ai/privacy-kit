"""Deterministic secret detection: API keys, tokens, private keys, credentials.

The NER model has no label for credentials, so without this layer an API key in
a proxied file read leaks upstream whenever it doesn't happen to look like one
of the model's PII types. This detector is the deterministic safety net: pure
regex (plus a Shannon-entropy floor on the broad rules), no model, no network,
microseconds per call.

Rules live in :mod:`privacy_kit.core.secret_rules` (vendored from gitleaks).
"""

from __future__ import annotations

import math
from collections import Counter

from privacy_kit.core.secret_rules import SECRET_RULES, SecretRule
from privacy_kit.core.types import Span

__all__ = ["SecretDetector", "shannon_entropy"]


def shannon_entropy(value: str) -> float:
    """Bits of entropy per character; real credentials typically score > 3.5."""
    if not value:
        return 0.0
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in Counter(value).values())


class SecretDetector:
    """Find credential spans via the vendored deterministic rule table.

    Every match is emitted with ``score=1.0`` — these patterns are exact, not
    probabilistic. Overlapping matches from different rules are resolved by the
    caller (:class:`~privacy_kit.core.detectors.CompositeDetector`).
    """

    def __init__(self, rules: tuple[SecretRule, ...] = SECRET_RULES) -> None:
        self._rules = rules

    def detect(self, text: str) -> list[Span]:
        if not text:
            return []
        spans: list[Span] = []
        for rule in self._rules:
            for match in rule.pattern.finditer(text):
                start, end = match.span(rule.secret_group)
                if end <= start:
                    continue
                if rule.min_entropy and shannon_entropy(match.group(rule.secret_group)) < (
                    rule.min_entropy
                ):
                    continue
                spans.append(Span(start, end, rule.label))
        spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
        return spans
