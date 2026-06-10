"""Span postprocessing tests (no model needed)."""

from __future__ import annotations

from privacy_kit.core.detectors import trim_spans
from privacy_kit.core.types import Span


def span_over(text: str, sub: str, etype: str = "PERSON_NAME") -> Span:
    start = text.index(sub)
    return Span(start, start + len(sub), etype, 0.9)


def test_trims_trailing_comma() -> None:
    text = "Hi Anna, how are you?"
    [trimmed] = trim_spans(text, [span_over(text, "Anna,")])
    assert trimmed.text_of(text) == "Anna"


def test_trims_multiple_trailing_punctuation() -> None:
    text = 'She said "call John Smith!".'
    [trimmed] = trim_spans(text, [span_over(text, 'John Smith!".')])
    assert trimmed.text_of(text) == "John Smith"


def test_keeps_interior_punctuation() -> None:
    text = "Mail anna.k@example.com, thanks"
    [trimmed] = trim_spans(text, [span_over(text, "anna.k@example.com,", "EMAIL_ADDRESS")])
    assert trimmed.text_of(text) == "anna.k@example.com"


def test_trailing_digit_or_letter_untouched() -> None:
    text = "Call +48 123 456 789 now"
    span = span_over(text, "+48 123 456 789", "PHONE_NUMBER")
    assert trim_spans(text, [span]) == [span]


def test_unicode_letters_are_alphanumeric() -> None:
    text = "Pan Michał Swędrowski."
    [trimmed] = trim_spans(text, [span_over(text, "Michał Swędrowski.")])
    assert trimmed.text_of(text) == "Michał Swędrowski"


def test_all_punctuation_span_is_dropped() -> None:
    text = "a ... b"
    assert trim_spans(text, [span_over(text, "...", "OTHER")]) == []
