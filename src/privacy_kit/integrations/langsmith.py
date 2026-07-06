from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from privacy_kit.core.detectors import build_detector
from privacy_kit.core.redactor import Redactor

# Wiring choice: LangSmith's ``Client(anonymizer=...)`` expects a
# ``Callable[[dict], dict]`` that is applied to the serialized run inputs,
# outputs, and (wrapped) errors before upload. Our ``Redactor.redact`` is
# already exactly this shape: it walks nested dicts/lists/strings and returns
# the same structure. We therefore pass ``redactor.redact`` directly instead of
# wrapping ``redactor.redact_text`` via ``langsmith.anonymizer.create_anonymizer``.
# ``create_anonymizer`` would call our function per-string and would lose the
# ``include_paths``/``exclude_paths`` filtering that only makes sense with the
# full payload in view, so the direct redactor is both simpler and more correct.


def make_anonymizer(
    backend: str | None = None,
    threshold: float | None = None,
    include_labels: set[str] | None = None,
    exclude_labels: set[str] | None = None,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    allow_terms: list[str] | None = None,
    allow_patterns: list[str] | None = None,
) -> Callable[[Any], Any]:
    """Return a callable for LangSmith's ``Client(anonymizer=...)`` parameter."""

    resolved_backend = (
        backend if backend is not None else os.getenv("PII_DETECTOR_BACKEND", "local")
    )
    resolved_threshold = (
        threshold if threshold is not None else float(os.getenv("PII_THRESHOLD", "0.5"))
    )
    resolved_include_labels = include_labels or _labels_from_env("PII_INCLUDE_LABELS")
    resolved_exclude_labels = exclude_labels or _labels_from_env("PII_EXCLUDE_LABELS")
    resolved_include_paths = include_paths or _paths_from_env("PII_INCLUDE_PATHS")
    resolved_exclude_paths = exclude_paths or _paths_from_env("PII_EXCLUDE_PATHS")
    resolved_allow_terms = allow_terms or _paths_from_env("PII_ALLOW_TERMS")
    resolved_allow_patterns = allow_patterns or _paths_from_env("PII_ALLOW_PATTERNS")
    redactor = Redactor(
        detector=build_detector(backend=resolved_backend, threshold=resolved_threshold),
        include_labels=resolved_include_labels,
        exclude_labels=resolved_exclude_labels,
        include_paths=resolved_include_paths,
        exclude_paths=resolved_exclude_paths,
        allow_terms=resolved_allow_terms,
        allow_patterns=resolved_allow_patterns,
    )

    def anonymizer(data: Any) -> Any:
        return redactor.redact(data)

    return anonymizer


def make_client(
    *,
    anonymizer_kwargs: dict[str, Any] | None = None,
    hide_metadata: bool = True,
    **client_kwargs: Any,
) -> Any:
    """Return a LangSmith ``Client`` preconfigured to scrub PII client-side.

    ``hide_metadata`` defaults to ``True`` because run metadata is not routed
    through the anonymizer by LangSmith, so it would otherwise be uploaded raw.
    """

    anonymizer = make_anonymizer(**(anonymizer_kwargs or {}))

    try:
        from langsmith import Client
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LangSmith tracing requires optional dependencies. "
            "Install them with: pip install 'privacy-kit[langsmith]'"
        ) from exc

    return Client(anonymizer=anonymizer, hide_metadata=hide_metadata, **client_kwargs)


def _labels_from_env(name: str) -> set[str] | None:
    value = os.getenv(name)
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _paths_from_env(name: str) -> list[str] | None:
    value = os.getenv(name)
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]
