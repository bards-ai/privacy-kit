"""Best-effort language detection for audit metadata.

The detected language is advisory metadata attached to each interaction. It must
never affect the proxy path, so detection never raises: any failure (missing
optional dependency, text too short to classify) simply yields ``None``.
"""

from __future__ import annotations

from contextlib import suppress

try:
    from langdetect import DetectorFactory
    from langdetect import detect as _detect

    # langdetect is randomized by default; pin the seed for stable results.
    DetectorFactory.seed = 0
except Exception:  # pragma: no cover - optional dependency not installed
    _detect = None

# Below this many characters langdetect's guess is too noisy to be useful.
_MIN_CHARS = 10
# Classifying the first chunk is plenty; cap the work on large inputs.
_MAX_CHARS = 1000


def detect_language(text: str) -> str | None:
    """Return a best-effort ISO 639-1 code for *text*, or ``None`` if undetermined."""
    if _detect is None:
        return None
    sample = text.strip()[:_MAX_CHARS]
    if len(sample) < _MIN_CHARS:
        return None
    with suppress(Exception):
        return str(_detect(sample))
    return None
