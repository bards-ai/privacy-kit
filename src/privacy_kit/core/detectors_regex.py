"""Checksum-validated PII detection: payment cards, IBANs, E.164 phones, US SSNs.

These are format-defined values the NER model only catches when the surrounding
context "looks right" — a bare card number in a log line is routinely missed.
Each candidate here must pass its checksum (Luhn, IBAN mod-97) or structural
validation, so false positives stay near zero and the spans can carry
``score=1.0``.

Labels reuse the model's own entity types (``PAYMENT_CARD``,
``BANK_ACCOUNT_IDENTIFIER``, ``PHONE_NUMBER``, ``PERSON_IDENTIFIER``) so a value
caught by either path gets the same placeholder class in the Vault.
"""

from __future__ import annotations

import re

from privacy_kit.core.types import Span

__all__ = ["ChecksumPiiDetector", "iban_mod97_ok", "luhn_ok"]

# 13-19 digits with optional single space/dash separators between them.
_CARD_CANDIDATE = re.compile(r"(?<![\d.])(?:\d[ \-]?){12,18}\d(?![\d.])")

# Two-letter country, two check digits, then 11-30 alphanumerics (spaces allowed
# in display form). Total length is validated after stripping separators.
_IBAN_CANDIDATE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{2,4}){3,8}\b")

# International format only (leading +): the model handles local formats with
# context; a bare digit run without country code is too false-positive-prone.
_PHONE_CANDIDATE = re.compile(r"(?<![\w+])\+[1-9]\d{0,2}(?:[ \-]?\(?\d{1,4}\)?){2,6}(?!\d)")

# Dashed US SSN with the structurally-invalid ranges excluded.
_SSN_CANDIDATE = re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")


def luhn_ok(digits: str) -> bool:
    """Luhn checksum for payment-card numbers (input must be digits only)."""
    if not digits.isdigit():
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def iban_mod97_ok(iban: str) -> bool:
    """ISO 13616 mod-97 check (input must be separator-free, uppercase)."""
    if not (15 <= len(iban) <= 34):
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged if ch.isalnum())
    if len(numeric) < len(rearranged):
        return False
    return int(numeric) % 97 == 1


class ChecksumPiiDetector:
    """Deterministic detector for checksummable / strictly-formatted PII."""

    def detect(self, text: str) -> list[Span]:
        if not text:
            return []
        spans: list[Span] = []

        for match in _CARD_CANDIDATE.finditer(text):
            digits = re.sub(r"[ \-]", "", match.group())
            if 13 <= len(digits) <= 19 and luhn_ok(digits):
                spans.append(Span(match.start(), match.end(), "PAYMENT_CARD"))

        for match in _IBAN_CANDIDATE.finditer(text):
            compact = match.group().replace(" ", "")
            if iban_mod97_ok(compact):
                spans.append(Span(match.start(), match.end(), "BANK_ACCOUNT_IDENTIFIER"))

        for match in _PHONE_CANDIDATE.finditer(text):
            digits = re.sub(r"\D", "", match.group())
            if 8 <= len(digits) <= 15:
                spans.append(Span(match.start(), match.end(), "PHONE_NUMBER"))

        for match in _SSN_CANDIDATE.finditer(text):
            spans.append(Span(match.start(), match.end(), "PERSON_IDENTIFIER"))

        spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
        return spans
