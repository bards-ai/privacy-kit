"""Reversible pseudonymization.

A :class:`Vault` records the mapping between real PII values and the typed
placeholders that replace them, so the same value always maps to the same
placeholder and the substitution can be undone. ``anonymize`` strips PII out of
text before it leaves your process (e.g. to an upstream LLM); ``deanonymize``
re-inserts the real values into the response.

This is the reversible counterpart to :class:`~privacy_kit.core.redactor.Redactor`:
use the Redactor where the destination should never see the values
(observability storage), and a Vault where the consumer still needs referential
consistency and the caller needs the real values back (a live LLM loop).

This module is model-free: ``anonymize`` accepts any
:class:`~privacy_kit.core.detectors.Detector`, so it can be tested with a stub
and runs without downloading the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from privacy_kit.core.detectors import Detector
from privacy_kit.core.types import Span


@dataclass
class Vault:
    """Bidirectional, type-aware map between real values and placeholders.

    Placeholders look like ``[PERSON_NAME_1]``: the entity type plus a per-type
    counter assigned in first-seen order. The closing bracket makes them
    unambiguous to reverse (``[X_1]`` is never a prefix of ``[X_10]``).
    """

    _to_placeholder: dict[tuple[str, str], str] = field(default_factory=dict)
    _to_value: dict[str, str] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def placeholder_for(self, entity_type: str, value: str) -> str:
        """Return the placeholder for ``value``, creating one on first sight."""
        key = (entity_type, value)
        existing = self._to_placeholder.get(key)
        if existing is not None:
            return existing
        n = self._counts.get(entity_type, 0) + 1
        self._counts[entity_type] = n
        placeholder = f"[{entity_type}_{n}]"
        self._to_placeholder[key] = placeholder
        self._to_value[placeholder] = value
        return placeholder

    @property
    def mapping(self) -> dict[str, str]:
        """A copy of the placeholder -> real value map."""
        return dict(self._to_value)

    @property
    def type_counts(self) -> dict[str, int]:
        """Distinct PII values seen per entity type (safe to persist — no values)."""
        return dict(self._counts)

    def __len__(self) -> int:
        return len(self._to_value)


def anonymize(text: str, detector: Detector) -> tuple[str, Vault]:
    """Replace detected PII in ``text`` with placeholders.

    Returns the sanitized text and a populated :class:`Vault`. Overlapping spans
    are resolved by keeping the earliest, leaving the rest in place. Placeholders
    are numbered left-to-right; substitution happens right-to-left so character
    offsets stay valid during replacement.
    """
    vault = Vault()
    return anonymize_into(text, detector, vault), vault


def anonymize_into(text: str, detector: Detector, vault: Vault) -> str:
    """Anonymize ``text`` into an existing ``vault`` and return the clean text.

    Use this to share one vault across several strings (e.g. every message in a
    single request) so the same value maps to the same placeholder throughout.
    """
    spans = sorted(detector.detect(text), key=lambda s: (s.start, s.end))

    chosen: list[Span] = []
    last_end = -1
    for span in spans:
        if span.start >= last_end and span.end > span.start:
            chosen.append(span)
            last_end = span.end

    # First pass (left-to-right): assign placeholders so numbering follows reading order.
    for span in chosen:
        vault.placeholder_for(span.label, text[span.start : span.end])

    # Second pass (right-to-left): splice placeholders in without shifting offsets.
    out = text
    for span in reversed(chosen):
        placeholder = vault.placeholder_for(span.label, text[span.start : span.end])
        out = out[: span.start] + placeholder + out[span.end :]
    return out


def deanonymize(text: str, vault: Vault) -> str:
    """Re-insert real values, replacing each placeholder in ``text``."""
    # Longest placeholders first as a belt-and-suspenders guard against any
    # placeholder being a substring of another.
    for placeholder, value in sorted(vault.mapping.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(placeholder, value)
    return text
