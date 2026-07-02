"""Shared data contracts used across detection, redaction, and pseudonymization."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Span:
    """A detected PII span in a piece of text.

    ``start``/``end`` are character offsets into the original string
    (``text[start:end]`` is the matched substring). ``label`` is the model
    label without BIO prefix (e.g. ``PERSON_NAME``, ``EMAIL_ADDRESS``).
    """

    start: int
    end: int
    label: str
    score: float = 1.0

    def overlaps(self, other: "Span") -> bool:
        """True if this span shares at least one character with ``other``."""
        return self.start < other.end and other.start < self.end

    def text_of(self, source: str) -> str:
        """Return the substring this span covers in ``source``."""
        return source[self.start : self.end]
