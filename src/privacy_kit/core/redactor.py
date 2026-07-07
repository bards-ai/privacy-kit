from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from privacy_kit.core.detectors import Detector, build_detector
from privacy_kit.core.types import Span

PathPattern = tuple[str, ...]
DataPath = tuple[str, ...]


class Redactor:
    def __init__(
        self,
        detector: Detector | None = None,
        replacement: str = "[REDACTED]",
        max_depth: int = 20,
        include_labels: set[str] | None = None,
        exclude_labels: set[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        expand_to_word_boundaries: bool = True,
        allow_terms: Sequence[str] | None = None,
        allow_patterns: Sequence[str] | None = None,
    ) -> None:
        self.detector = detector or build_detector()
        self.replacement = replacement
        self.max_depth = max_depth
        self.include_labels = include_labels
        self.exclude_labels = exclude_labels or set()
        self.include_paths = _compile_path_patterns(include_paths)
        self.exclude_paths = _compile_path_patterns(exclude_paths)
        self.expand_to_word_boundaries = expand_to_word_boundaries
        self.allow_terms = (
            {term.strip().casefold() for term in allow_terms} if allow_terms else set()
        )
        self.allow_patterns = _compile_allow_patterns(allow_patterns)

    def detect(self, text: str) -> list[Span]:
        spans = self.detector.detect(text)
        return [
            span for span in spans if self._should_redact(span) and not self._is_allowed(text, span)
        ]

    def spans_for_text(self, text: str) -> list[Span]:
        return self._prepare_spans(text, self.detect(text))

    def redact_text(self, text: str) -> str:
        spans = self.spans_for_text(text)
        if not spans:
            return text

        chunks: list[str] = []
        cursor = 0
        for span in spans:
            chunks.append(text[cursor : span.start])
            chunks.append(self.replacement)
            cursor = span.end
        chunks.append(text[cursor:])
        return "".join(chunks)

    def redact(self, data: Any, depth: int = 0, path: DataPath = ()) -> Any:
        if depth > self.max_depth:
            return self.replacement
        if self._is_excluded_path(path):
            return data
        if isinstance(data, str):
            if not self._is_included_path(path):
                return data
            return self.redact_text(data)
        if isinstance(data, Mapping):
            return {
                key: self.redact(value, depth + 1, (*path, str(key))) for key, value in data.items()
            }
        if isinstance(data, tuple):
            return tuple(
                self.redact(value, depth + 1, (*path, str(index)))
                for index, value in enumerate(data)
            )
        if isinstance(data, Sequence) and not isinstance(data, (bytes, bytearray)):
            return [
                self.redact(value, depth + 1, (*path, str(index)))
                for index, value in enumerate(data)
            ]
        serializable = _to_serializable(data)
        if serializable is not data:
            return self.redact(serializable, depth + 1, path)
        return data

    def _should_redact(self, span: Span) -> bool:
        if self.include_labels is not None and span.label not in self.include_labels:
            return False
        return span.label not in self.exclude_labels

    def _is_allowed(self, text: str, span: Span) -> bool:
        if not self.allow_terms and not self.allow_patterns:
            return False
        span_text = text[span.start : span.end].strip()
        if span_text.casefold() in self.allow_terms:
            return True
        return any(pattern.fullmatch(span_text) for pattern in self.allow_patterns)

    def _prepare_spans(self, text: str, spans: list[Span]) -> list[Span]:
        if not self.expand_to_word_boundaries:
            return spans

        expanded = [self._expand_span_to_word_boundaries(text, span) for span in spans]
        return _merge_spans(expanded)

    def _expand_span_to_word_boundaries(self, text: str, span: Span) -> Span:
        start = max(0, min(span.start, len(text)))
        end = max(start, min(span.end, len(text)))
        while start > 0 and not text[start - 1].isspace():
            start -= 1
        while end < len(text) and not text[end].isspace():
            end += 1

        while start < end and _is_boundary_punctuation(text[start]):
            start += 1
        while end > start and _is_boundary_punctuation(text[end - 1]):
            end -= 1
        return Span(start, end, span.label, span.score)

    def _is_included_path(self, path: DataPath) -> bool:
        if self.include_paths is None:
            return True
        return any(_path_matches(pattern, path) for pattern in self.include_paths)

    def _is_excluded_path(self, path: DataPath) -> bool:
        if self.exclude_paths is None:
            return False
        return any(_path_matches(pattern, path) for pattern in self.exclude_paths)


def load_allow_file(path: str | os.PathLike[str]) -> tuple[list[str], list[str]]:
    """Load allowlist terms and regex patterns from a text file.

    One entry per line. Blank lines and lines starting with ``#`` are
    ignored. Lines starting with ``re:`` are treated as regex patterns
    (the rest of the line, stripped, is the pattern); everything else is
    a literal term.
    """
    terms: list[str] = []
    patterns: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("re:"):
                patterns.append(line[len("re:") :].strip())
            else:
                terms.append(line)
    return terms, patterns


def _is_boundary_punctuation(character: str) -> bool:
    return re.match(r"[.,;:!?\"'()\[\]{}]", character, flags=re.UNICODE) is not None


def _merge_spans(spans: list[Span]) -> list[Span]:
    merged: list[Span] = []
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if not merged or span.start > merged[-1].end:
            merged.append(span)
            continue
        previous = merged[-1]
        merged[-1] = Span(
            previous.start,
            max(previous.end, span.end),
            previous.label,
            min(previous.score, span.score),
        )
    return merged


def _compile_allow_patterns(patterns: Sequence[str] | None) -> list[re.Pattern[str]]:
    if not patterns:
        return []
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            raise ValueError(f"Invalid allow_patterns regex {pattern!r}: {exc}") from exc
    return compiled


def _compile_path_patterns(paths: list[str] | None) -> list[PathPattern] | None:
    if paths is None:
        return None
    return [tuple(part for part in path.split(".") if part) for path in paths if path.strip()]


def _path_matches(pattern: PathPattern, path: DataPath) -> bool:
    if not pattern:
        return not path
    if len(pattern) > len(path):
        return False
    return all(
        expected == "*" or expected == actual
        for expected, actual in zip(pattern, path, strict=False)
    )


def _to_serializable(data: Any) -> Any:
    if is_dataclass(data) and not isinstance(data, type):
        return asdict(data)

    model_dump = getattr(data, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()

    dict_method = getattr(data, "dict", None)
    if callable(dict_method):
        try:
            return dict_method()
        except TypeError:
            return data

    return data
