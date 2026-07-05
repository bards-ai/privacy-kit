"""Deterministic detector tests: secret rules, checksum PII, composite merge.

Model-free — these detectors are pure regex + checksums, so every test runs in
CI unconditionally. Every secret value here is fabricated.
"""

from __future__ import annotations

import pytest

from privacy_kit.core.detectors import CompositeDetector
from privacy_kit.core.detectors_regex import ChecksumPiiDetector, iban_mod97_ok, luhn_ok
from privacy_kit.core.detectors_secret import SecretDetector, shannon_entropy
from privacy_kit.core.types import Span

# --- Secret rules: one positive per rule family, plus lookalike negatives ---

SECRET_POSITIVES = [
    ("AKIAIOSFODNN7EXAMPLE", "SECRET_AWS_ACCESS_KEY"),
    ("ASIAY34FZKBOKMUTVV7A", "SECRET_AWS_ACCESS_KEY"),
    ("AIzaSyA1234567890abcdefghijklmnopqrstuv", "SECRET_GCP_API_KEY"),
    ("ghp_" + "a1B2" * 9, "SECRET_GITHUB_TOKEN"),
    ("ghs_" + "a1B2" * 9, "SECRET_GITHUB_TOKEN"),
    ("github_pat_" + "a" * 22 + "_" + "B" * 59, "SECRET_GITHUB_TOKEN"),
    ("glpat-abcdefghij0123456789", "SECRET_GITLAB_TOKEN"),
    ("sk-" + "a1c3e5g7i9" * 2 + "T3BlbkFJ" + "b2d4f6h8j0" * 2, "SECRET_OPENAI_API_KEY"),
    ("sk-proj-" + "Ab1_" * 12, "SECRET_OPENAI_API_KEY"),
    ("sk-ant-api03-" + "Xy9_" * 22, "SECRET_ANTHROPIC_API_KEY"),
    ("hf_" + "abcdEFGHijkl" * 2 + "mnopQRSTuv", "SECRET_HUGGINGFACE_TOKEN"),
    ("xoxb-123456789012-abcdefABCDEF", "SECRET_SLACK_TOKEN"),
    ("sk_live_" + "a1B2c3D4e5" * 2, "SECRET_STRIPE_KEY"),
    ("SG." + "a1B2c3D4e5f6G7h8i9J0kl" + "." + "m" * 43, "SECRET_SENDGRID_KEY"),
    ("npm_" + "a1B2c3D4e5" * 3 + "f6G7h8", "SECRET_NPM_TOKEN"),
    ("pypi-AgEIcHlwaS5vcmc" + "Ab-_" * 15, "SECRET_PYPI_TOKEN"),
    (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkphbiJ9"
        ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJVadQssw5c",
        "SECRET_JWT",
    ),
]


@pytest.mark.parametrize(("secret", "label"), SECRET_POSITIVES)
def test_secret_rule_catches(secret: str, label: str) -> None:
    text = f"config value: {secret} (do not share)"
    spans = SecretDetector().detect(text)
    assert [s.label for s in spans] == [label]
    assert spans[0].text_of(text) == secret
    assert spans[0].score == 1.0


def test_private_key_block() -> None:
    pem = (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW\n"
        "-----END OPENSSH PRIVATE KEY-----"
    )
    text = f"found this in ~/.ssh/id_ed25519:\n{pem}\n"
    spans = SecretDetector().detect(text)
    assert [s.label for s in spans] == ["SECRET_PRIVATE_KEY"]
    assert spans[0].text_of(text) == pem


def test_connection_string_masks_only_the_password() -> None:
    text = "DATABASE_URL=postgresql://svc_user:S3cr3t.Pw!@db.internal:5432/app"
    spans = SecretDetector().detect(text)
    assert [s.label for s in spans] == ["SECRET_CONNECTION_STRING"]
    assert spans[0].text_of(text) == "S3cr3t.Pw!"


def test_authorization_header_value() -> None:
    text = "curl -H 'Authorization: Bearer kJ8x2mQp9zR4vT6wY1nB3cD5f'"
    spans = SecretDetector().detect(text)
    assert [s.label for s in spans] == ["SECRET_AUTH_HEADER"]
    assert spans[0].text_of(text) == "kJ8x2mQp9zR4vT6wY1nB3cD5f"


def test_env_style_assignment_with_entropy_gate() -> None:
    # High-entropy value -> caught; the span covers only the value.
    text = "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    spans = SecretDetector().detect(text)
    assert [s.label for s in spans] == ["SECRET_GENERIC"]
    assert spans[0].text_of(text) == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"


SECRET_NEGATIVES = [
    "the password = get_token() call returns quickly",  # low entropy value
    "set api_key = placeholder for now",  # low entropy
    "AKIA is the AWS key prefix",  # prefix without body
    "ghp_tooshort",  # wrong length
    "he said eyes.eyes.eyes twice",  # JWT lookalike
    "see https://example.com/path for details",  # URL without credentials
    "przykladowy tekst bez sekretow, tel. brak",  # plain non-English text
]


@pytest.mark.parametrize("text", SECRET_NEGATIVES)
def test_secret_rules_ignore_lookalikes(text: str) -> None:
    assert SecretDetector().detect(text) == []


def test_shannon_entropy_bounds() -> None:
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaaaaaa") == 0.0
    assert shannon_entropy("wJalrXUtnFEMI/K7MDENG") > 3.5


# --- Checksum PII ---


def test_luhn_validation() -> None:
    assert luhn_ok("4111111111111111")
    assert not luhn_ok("4111111111111112")
    assert not luhn_ok("not-digits")


def test_iban_mod97_validation() -> None:
    assert iban_mod97_ok("GB82WEST12345698765432")
    assert not iban_mod97_ok("GB82WEST12345698765431")
    assert not iban_mod97_ok("GB82")


def test_card_detected_with_and_without_separators() -> None:
    detector = ChecksumPiiDetector()
    for rendering in ("4111111111111111", "4111 1111 1111 1111", "4111-1111-1111-1111"):
        text = f"card: {rendering} exp 12/28"
        spans = detector.detect(text)
        assert [s.label for s in spans] == ["PAYMENT_CARD"], rendering
        assert spans[0].text_of(text) == rendering


def test_card_failing_luhn_is_ignored() -> None:
    assert ChecksumPiiDetector().detect("order id 4111111111111112 confirmed") == []


def test_iban_detected_in_display_form() -> None:
    text = "wire to GB82 WEST 1234 5698 7654 32 please"
    spans = ChecksumPiiDetector().detect(text)
    assert [s.label for s in spans] == ["BANK_ACCOUNT_IDENTIFIER"]
    assert spans[0].text_of(text) == "GB82 WEST 1234 5698 7654 32"


def test_e164_phone_detected() -> None:
    text = "call +48 501 222 333 tomorrow"
    spans = ChecksumPiiDetector().detect(text)
    assert [s.label for s in spans] == ["PHONE_NUMBER"]
    assert spans[0].text_of(text) == "+48 501 222 333"


def test_bare_digit_run_is_not_a_phone() -> None:
    # No leading + -> left to the model, which has the context to judge.
    assert ChecksumPiiDetector().detect("invoice 123456789 paid") == []


def test_us_ssn_detected_and_invalid_ranges_skipped() -> None:
    detector = ChecksumPiiDetector()
    spans = detector.detect("SSN 536-90-4399 on file")
    assert [s.label for s in spans] == ["PERSON_IDENTIFIER"]
    assert detector.detect("SSN 000-12-3456 is not valid") == []
    assert detector.detect("SSN 666-12-3456 is not valid") == []


# --- Composite union / precedence ---


class StaticDetector:
    def __init__(self, spans: list[Span]) -> None:
        self._spans = spans

    def detect(self, text: str) -> list[Span]:
        return list(self._spans)


def test_composite_unions_disjoint_spans_in_order() -> None:
    a = StaticDetector([Span(10, 20, "PERSON_NAME", 0.9)])
    b = StaticDetector([Span(0, 5, "SECRET_GENERIC", 1.0)])
    assert CompositeDetector([a, b]).detect("x" * 30) == [
        Span(0, 5, "SECRET_GENERIC", 1.0),
        Span(10, 20, "PERSON_NAME", 0.9),
    ]


def test_composite_secret_wins_overlap_even_when_narrower() -> None:
    model = StaticDetector([Span(0, 30, "PROPER_NAME", 0.95)])
    secrets = StaticDetector([Span(5, 15, "SECRET_AWS_ACCESS_KEY", 1.0)])
    assert CompositeDetector([model, secrets]).detect("x" * 40) == [
        Span(5, 15, "SECRET_AWS_ACCESS_KEY", 1.0)
    ]


def test_composite_prefers_wider_span_among_equals() -> None:
    a = StaticDetector([Span(0, 10, "PERSON_NAME", 0.8)])
    b = StaticDetector([Span(0, 25, "POSTAL_ADDRESS", 0.7)])
    assert CompositeDetector([a, b]).detect("x" * 30) == [Span(0, 25, "POSTAL_ADDRESS", 0.7)]


def test_composite_real_detectors_on_mixed_text() -> None:
    detector = CompositeDetector([SecretDetector(), ChecksumPiiDetector()])
    text = (
        "Deploy notes: export AWS key AKIAIOSFODNN7EXAMPLE, "
        "card 4111 1111 1111 1111, IBAN GB82WEST12345698765432."
    )
    labels = [s.label for s in detector.detect(text)]
    assert labels == ["SECRET_AWS_ACCESS_KEY", "PAYMENT_CARD", "BANK_ACCOUNT_IDENTIFIER"]
