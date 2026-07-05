"""Unit tests for the ONNX detector internals — no model download needed.

The detector is built with ``object.__new__`` and hand-wired fakes so the
window planning, batched inference, best-per-token merging, and whole-word
aggregation can be verified deterministically.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from privacy_kit.core.detectors import (
    BardsAiOnnxDetector,
    CompositeDetector,
    NullDetector,
    build_detector,
)
from privacy_kit.core.types import Span


def make_detector(**attrs: Any) -> BardsAiOnnxDetector:
    detector = object.__new__(BardsAiOnnxDetector)
    for name, value in attrs.items():
        setattr(detector, name, value)
    return detector


class FakeSession:
    """Records every batch and returns logits whose confidence grows per row.

    Label id for every content token in batch row ``r`` is ``r + 1`` (row
    counted across all batches), with a logit magnitude that increases with the
    row index — so on overlapping tokens the *later* window is more confident
    and must win the merge.
    """

    def __init__(self, n_labels: int) -> None:
        self.n_labels = n_labels
        self.calls: list[dict[str, Any]] = []
        self._rows_seen = 0

    def run(self, outputs: list[str], inputs: dict[str, Any]) -> list[Any]:
        self.calls.append(inputs)
        batch, width = inputs["input_ids"].shape
        logits = np.zeros((batch, width, self.n_labels), dtype=np.float32)
        for row in range(batch):
            global_row = self._rows_seen + row
            logits[row, :, (global_row + 1) % self.n_labels] = 2.0 * (global_row + 1)
        self._rows_seen += batch
        return [logits]


def test_plan_windows_cover_all_tokens_with_overlap() -> None:
    detector = make_detector(max_tokens=6, stride=2)  # window=4, step=2
    assert detector._plan_windows(10) == [0, 2, 4, 6]
    assert detector._plan_windows(4) == [0]
    assert detector._plan_windows(1) == [0]


def test_plan_windows_step_never_stalls() -> None:
    detector = make_detector(max_tokens=6, stride=3)  # window=4, step=1
    starts = detector._plan_windows(6)
    assert starts == [0, 1, 2]
    # every token is covered by at least one window
    covered = {i for s in starts for i in range(s, s + 4)}
    assert set(range(6)) <= covered


def test_best_predictions_batches_pads_and_merges() -> None:
    session = FakeSession(n_labels=8)
    detector = make_detector(
        max_tokens=6,  # window=4
        stride=2,  # step=2
        _np=np,
        _cls_id=101,
        _sep_id=102,
        _pad_id=0,
        _input_names={"input_ids", "attention_mask"},
        _output_name="logits",
        _session=session,
    )

    ids = list(range(1, 10))  # 9 tokens -> windows at 0, 2, 4, 6 (last has 3 tokens)
    label_ids, scores = detector._best_predictions(ids)

    # One batched forward pass: 4 windows padded to a common width of 3+2... the
    # longest chunk has 4 tokens, +2 specials = 6 columns.
    assert len(session.calls) == 1
    input_ids = session.calls[0]["input_ids"]
    attention_mask = session.calls[0]["attention_mask"]
    assert input_ids.shape == (4, 6)
    assert input_ids[0].tolist() == [101, 1, 2, 3, 4, 102]
    assert input_ids[3].tolist() == [101, 7, 8, 9, 102, 0]  # short last window, padded
    assert attention_mask[3].tolist() == [1, 1, 1, 1, 1, 0]

    # Later (more confident) windows win on overlapping tokens:
    # tokens 0-1 only in window 0 (label 1); 2-3 best in window 1 (label 2);
    # 4-5 best in window 2 (label 3); 6-8 best in window 3 (label 4).
    assert label_ids == [1, 1, 2, 2, 3, 3, 4, 4, 4]
    assert all(score > 0 for score in scores)


def test_aggregate_tags_whole_word_from_best_subword() -> None:
    detector = make_detector(
        id2label={0: "O", 1: "B-EMAIL_ADDRESS", 2: "I-EMAIL_ADDRESS"},
    )
    # One word split into three subwords; only two are tagged, with the best
    # score on the last. The whole word must come out as one span.
    offsets = [(0, 2), (3, 9), (9, 12), (12, 21)]
    word_ids = [0, 1, 1, 1]
    label_ids: list[int | None] = [0, 1, 0, 2]
    scores = [0.9, 0.8, 0.4, 0.95]

    spans = detector._aggregate(offsets, word_ids, label_ids, scores)

    assert spans == [Span(3, 21, "EMAIL_ADDRESS", 0.95)]


def test_aggregate_merges_consecutive_words_of_same_type() -> None:
    detector = make_detector(id2label={0: "O", 1: "B-PERSON_NAME", 2: "I-PERSON_NAME"})
    offsets = [(0, 4), (5, 10), (11, 14)]
    word_ids = [0, 1, 2]
    label_ids: list[int | None] = [1, 2, 0]
    scores = [0.8, 0.6, 0.9]

    spans = detector._aggregate(offsets, word_ids, label_ids, scores)

    [span] = spans
    assert (span.start, span.end, span.label) == (0, 10, "PERSON_NAME")
    assert span.score == pytest.approx(0.7)  # average of the merged words


def test_postprocess_drops_low_confidence_and_overlaps() -> None:
    detector = make_detector(threshold=0.5)
    text = "Anna Nowak, ware"
    spans = [
        Span(0, 10, "PERSON_NAME", 0.9),
        Span(5, 12, "PERSON_NAME", 0.8),  # overlaps the first -> dropped
        Span(12, 16, "LOCATION", 0.2),  # below threshold -> dropped
    ]

    assert detector._postprocess(text, spans) == [Span(0, 10, "PERSON_NAME", 0.9)]


def test_regex_backend_builds_deterministic_composite() -> None:
    # No model download: the deterministic layer alone.
    detector = build_detector("regex")
    assert isinstance(detector, CompositeDetector)
    spans = detector.detect("key AKIAIOSFODNN7EXAMPLE leaked")
    assert [s.label for s in spans] == ["SECRET_AWS_ACCESS_KEY"]


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported detector backend: nope"):
        build_detector("nope")


def test_null_detector_finds_nothing() -> None:
    assert NullDetector().detect("Anna Smith lives in Berlin, anna@example.com") == []


def test_null_backend_builds_null_detector() -> None:
    assert isinstance(build_detector("null"), NullDetector)
