from __future__ import annotations

import os
from typing import Any, Callable

from privacy_kit.core.detectors import build_detector
from privacy_kit.core.redactor import Redactor


def make_mask(
    backend: str | None = None,
    threshold: float | None = None,
    include_labels: set[str] | None = None,
    exclude_labels: set[str] | None = None,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> Callable[[Any], Any]:
    """Return a Langfuse-compatible mask(data, **kwargs) function."""

    resolved_backend = backend or os.getenv("PII_DETECTOR_BACKEND", "local")
    resolved_threshold = threshold if threshold is not None else float(os.getenv("PII_THRESHOLD", "0.5"))
    resolved_include_labels = include_labels or _labels_from_env("PII_INCLUDE_LABELS")
    resolved_exclude_labels = exclude_labels or _labels_from_env("PII_EXCLUDE_LABELS")
    resolved_include_paths = include_paths or _paths_from_env("PII_INCLUDE_PATHS")
    resolved_exclude_paths = exclude_paths or _paths_from_env("PII_EXCLUDE_PATHS")
    redactor = Redactor(
        detector=build_detector(backend=resolved_backend, threshold=resolved_threshold),
        include_labels=resolved_include_labels,
        exclude_labels=resolved_exclude_labels,
        include_paths=resolved_include_paths,
        exclude_paths=resolved_exclude_paths,
    )

    def mask(data: Any, **_: Any) -> Any:
        return redactor.redact(data)

    return mask


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
