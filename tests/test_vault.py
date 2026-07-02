"""Reversible pseudonymization tests.

Model-free: a stub detector supplies spans, so these run in the base CI env.
A final round-trip test uses the session ``detector`` fixture, which skips
itself unless ``PII_RUN_MODEL_TESTS=1``.
"""

from __future__ import annotations

from privacy_kit.core.detectors import BardsAiOnnxDetector
from privacy_kit.core.types import Span
from privacy_kit.core.vault import Vault, anonymize, deanonymize


class StubDetector:
    """Returns a fixed span list regardless of input."""

    def __init__(self, spans: list[Span]) -> None:
        self._spans = spans

    def detect(self, text: str) -> list[Span]:
        return list(self._spans)


def spans_for(text: str, *items: tuple[str, str]) -> list[Span]:
    """Build spans by locating each (entity_type, substring) occurrence in order."""
    spans: list[Span] = []
    cursor = 0
    for entity_type, substring in items:
        start = text.index(substring, cursor)
        spans.append(Span(start, start + len(substring), entity_type, 0.99))
        cursor = start + len(substring)
    return spans


def test_round_trip_restores_original() -> None:
    text = "I'm John Smith, email john@x.com."
    spans = spans_for(text, ("PERSON_NAME", "John Smith"), ("EMAIL_ADDRESS", "john@x.com"))
    clean, vault = anonymize(text, StubDetector(spans))
    assert deanonymize(clean, vault) == text


def test_clean_text_contains_no_raw_pii() -> None:
    text = "Call Anna at +48 601 234 567."
    spans = spans_for(text, ("PERSON_NAME", "Anna"), ("PHONE_NUMBER", "+48 601 234 567"))
    clean, _ = anonymize(text, StubDetector(spans))
    assert "Anna" not in clean
    assert "+48 601 234 567" not in clean
    assert "[PERSON_NAME_1]" in clean
    assert "[PHONE_NUMBER_1]" in clean


def test_repeated_value_gets_same_placeholder() -> None:
    text = "John met John; tell John."
    spans = spans_for(
        text, ("PERSON_NAME", "John"), ("PERSON_NAME", "John"), ("PERSON_NAME", "John")
    )
    clean, vault = anonymize(text, StubDetector(spans))
    assert clean == "[PERSON_NAME_1] met [PERSON_NAME_1]; tell [PERSON_NAME_1]."
    assert len(vault) == 1


def test_distinct_values_get_incrementing_indices() -> None:
    text = "Ada and Linus and Grace."
    spans = spans_for(
        text, ("PERSON_NAME", "Ada"), ("PERSON_NAME", "Linus"), ("PERSON_NAME", "Grace")
    )
    clean, vault = anonymize(text, StubDetector(spans))
    assert clean == "[PERSON_NAME_1] and [PERSON_NAME_2] and [PERSON_NAME_3]."
    assert len(vault) == 3


def test_placeholder_for_is_idempotent() -> None:
    vault = Vault()
    first = vault.placeholder_for("EMAIL_ADDRESS", "a@b.com")
    second = vault.placeholder_for("EMAIL_ADDRESS", "a@b.com")
    assert first == second == "[EMAIL_ADDRESS_1]"
    assert len(vault) == 1


def test_overlapping_spans_keep_earliest_without_crashing() -> None:
    text = "Dr John Smith"
    overlapping = [
        Span(0, 13, "PERSON_ROLE_OR_TITLE", 0.9),  # "Dr John Smith"
        Span(3, 13, "PERSON_NAME", 0.95),  # "John Smith" (overlaps)
    ]
    clean, vault = anonymize(text, StubDetector(overlapping))
    assert clean == "[PERSON_ROLE_OR_TITLE_1]"
    assert deanonymize(clean, vault) == text


def test_deanonymize_no_double_digit_prefix_collision() -> None:
    vault = Vault()
    for i in range(11):
        vault.placeholder_for("PERSON_NAME", f"name{i}")
    text = "[PERSON_NAME_1] and [PERSON_NAME_10] and [PERSON_NAME_11]"
    assert deanonymize(text, vault) == "name0 and name9 and name10"


def test_round_trip_with_real_model(detector: BardsAiOnnxDetector) -> None:
    text = "Contact Marie Dubois at marie.dubois@example.fr."
    clean, vault = anonymize(text, detector)
    assert len(vault) >= 1
    assert "Marie Dubois" not in clean or "marie.dubois@example.fr" not in clean
    assert deanonymize(clean, vault) == text
