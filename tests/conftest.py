"""Shared test fixtures."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from privacy_kit.core.detectors import BardsAiOnnxDetector

FIXTURES = Path(__file__).parent / "fixtures"

requires_model = pytest.mark.skipif(
    os.getenv("PII_RUN_MODEL_TESTS") != "1",
    reason="set PII_RUN_MODEL_TESTS=1 to download and run the model",
)


@pytest.fixture(scope="session")
def detector() -> BardsAiOnnxDetector:
    """The real on-device model, loaded once per session (gated by PII_RUN_MODEL_TESTS)."""
    if os.getenv("PII_RUN_MODEL_TESTS") != "1":
        pytest.skip("set PII_RUN_MODEL_TESTS=1 to download and run the model")
    return BardsAiOnnxDetector()


@pytest.fixture(scope="session")
def pii_samples() -> list[dict[str, Any]]:
    """Multilingual PII samples with expected entity types."""
    samples: list[dict[str, Any]] = json.loads(
        (FIXTURES / "pii_samples.json").read_text(encoding="utf-8")
    )
    return samples
