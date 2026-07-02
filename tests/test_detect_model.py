"""Real-model detection tests.

These download and run the actual model, so they are gated behind
``PII_RUN_MODEL_TESTS=1`` (see ``make test-model``). They are the behavioral
contract for the detector: expected entity types across languages, whole-word
coverage, valid offsets, no overlapping spans.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from privacy_kit.core.detectors import BardsAiOnnxDetector


def test_labels_include_known_entities(detector: BardsAiOnnxDetector) -> None:
    labels = set(detector.labels)
    assert {"PERSON_NAME", "EMAIL_ADDRESS", "PHONE_NUMBER"} <= labels
    assert "O" not in labels  # BIO prefixes and the outside tag are stripped


def test_empty_text_returns_no_spans(detector: BardsAiOnnxDetector) -> None:
    assert detector.detect("") == []
    assert detector.detect("   \n  ") == []


def test_offsets_map_back_to_source(detector: BardsAiOnnxDetector) -> None:
    text = "Contact John Smith at john.smith@example.com."
    spans = detector.detect(text)
    assert spans
    for span in spans:
        assert 0 <= span.start < span.end <= len(text)
        assert span.text_of(text) == text[span.start : span.end]
        assert 0.0 <= span.score <= 1.0


def test_detects_expected_types_per_language(
    detector: BardsAiOnnxDetector, pii_samples: list[dict[str, Any]]
) -> None:
    for sample in pii_samples:
        spans = detector.detect(sample["text"])
        assert spans, f"{sample['lang']}: detected no PII"
        found = {s.label for s in spans}
        expected = set(sample["expect_types"])
        assert found & expected, f"{sample['lang']}: found {found}, expected any of {expected}"


def test_whole_word_coverage_no_partial_leak(detector: BardsAiOnnxDetector) -> None:
    # Regression: word-level aggregation must cover the *whole* email, not just
    # the subwords the model happened to tag (previously "ample.com" leaked).
    text = "Please email anna.k@example.com today."
    email = "anna.k@example.com"
    covered: set[int] = set()
    for span in detector.detect(text):
        covered.update(range(span.start, span.end))
    start = text.index(email)
    assert all(i in covered for i in range(start, start + len(email)))


def test_no_overlapping_spans(
    detector: BardsAiOnnxDetector, pii_samples: list[dict[str, Any]]
) -> None:
    for sample in pii_samples:
        spans = sorted(detector.detect(sample["text"]), key=lambda s: s.start)
        for prev, nxt in pairwise(spans):
            assert prev.end <= nxt.start, f"{sample['lang']}: overlapping spans {prev} {nxt}"


def test_long_text_entity_past_window_limit(detector: BardsAiOnnxDetector) -> None:
    # An entity far past the 512-token window must still be found (overlapping
    # windows, batched). Mirrors the legacy PiiModel chunking demo.
    base = (
        "Jan Kowalski mieszka na ul. Piękna 22, 00-549 Warszawa. "
        "Anna Nowak pracuje w firmie przy ul. Mokotowskiej 10. "
    )
    long_text = (base * 30) + "Wreszcie: Cristiano Ronaldo gra w piłkę."
    spans = detector.detect(long_text)
    target = long_text.index("Cristiano Ronaldo")
    assert any(s.start <= target < s.end for s in spans), "entity at end of long text was lost"
