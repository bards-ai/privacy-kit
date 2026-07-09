from pathlib import Path

import pytest

from privacy_kit.core.redactor import Redactor, load_allow_file
from privacy_kit.core.types import Span


class StaticDetector:
    def __init__(self, spans: list[Span]) -> None:
        self.spans = spans

    def detect(self, text: str) -> list[Span]:
        return self.spans


class PatternDetector:
    def __init__(self, patterns: dict[str, str]) -> None:
        self.patterns = patterns

    def detect(self, text: str) -> list[Span]:
        spans = []
        for value, label in self.patterns.items():
            start = text.find(value)
            if start >= 0:
                spans.append(Span(start, start + len(value), label, 1.0))
        return spans


def test_redacts_langfuse_shaped_trace_payload() -> None:
    redactor = Redactor(
        detector=PatternDetector(
            {
                "Jan Kowalski": "PERSON_NAME",
                "jan.kowalski@example.com": "EMAIL",
                "85010112345": "PERSONAL_ID",
                "Jana Kowalskiego": "PERSON_NAME",
                "+48 501 222 333": "PHONE_NUMBER",
            }
        )
    )

    result = redactor.redact(
        {
            "input": "Jan Kowalski ma email jan.kowalski@example.com i PESEL 85010112345.",
            "output": "Odpowiedź dla Jana Kowalskiego powinna być zapisana "
            "bez danych kontaktowych.",
            "metadata": {"customer_contact": "+48 501 222 333"},
        }
    )

    assert result["input"] == "[REDACTED] ma email [REDACTED] i PESEL [REDACTED]."
    assert (
        result["output"] == "Odpowiedź dla [REDACTED] powinna być zapisana bez danych kontaktowych."
    )
    assert result["metadata"]["customer_contact"] == "[REDACTED]"


def test_redacts_nested_observability_payload_by_default() -> None:
    redactor = Redactor(
        detector=PatternDetector(
            {
                "jan.kowalski@example.com": "EMAIL",
                "Anna Nowak": "PERSON_NAME",
                "+48 501 222 333": "PHONE_NUMBER",
            }
        )
    )

    result = redactor.redact(
        {
            "input": {"messages": [{"role": "user", "content": "Email jan.kowalski@example.com"}]},
            "output": {"reasoning": "Call Anna Nowak"},
            "tool_calls": [{"name": "send_sms", "args": {"phone": "+48 501 222 333"}}],
            "usage": {"total_tokens": 123},
        }
    )

    assert result["input"]["messages"][0]["content"] == "Email [REDACTED]"
    assert result["output"]["reasoning"] == "Call [REDACTED]"
    assert result["tool_calls"][0]["args"]["phone"] == "[REDACTED]"
    assert result["usage"]["total_tokens"] == 123


def test_redacts_pydantic_style_message_objects() -> None:
    class Message:
        def __init__(self, content: str) -> None:
            self.content = content

        def model_dump(self, mode: str = "python") -> dict[str, str]:
            return {"type": "human", "content": self.content, "mode": mode}

    redactor = Redactor(detector=PatternDetector({"jan.kowalski@example.com": "EMAIL"}))

    result = redactor.redact({"messages": [Message("Email jan.kowalski@example.com")]})

    assert result["messages"][0]["content"] == "Email [REDACTED]"
    assert result["messages"][0]["mode"] == "json"


def test_include_paths_limits_which_fields_are_scanned() -> None:
    redactor = Redactor(
        detector=PatternDetector(
            {
                "jan.kowalski@example.com": "EMAIL",
                "ops@example.com": "EMAIL",
            }
        ),
        include_paths=["messages.*.content"],
    )

    result = redactor.redact(
        {
            "messages": [{"role": "user", "content": "Email jan.kowalski@example.com"}],
            "metadata": {"support_contact": "ops@example.com"},
        }
    )

    assert result["messages"][0]["content"] == "Email [REDACTED]"
    assert result["metadata"]["support_contact"] == "ops@example.com"


def test_exclude_paths_win_over_include_paths() -> None:
    redactor = Redactor(
        detector=PatternDetector(
            {
                "jan.kowalski@example.com": "EMAIL",
                "ops@example.com": "EMAIL",
            }
        ),
        include_paths=["metadata"],
        exclude_paths=["metadata.support_contact"],
    )

    result = redactor.redact(
        {
            "metadata": {
                "customer_email": "jan.kowalski@example.com",
                "support_contact": "ops@example.com",
            }
        }
    )

    assert result["metadata"]["customer_email"] == "[REDACTED]"
    assert result["metadata"]["support_contact"] == "ops@example.com"


def test_expands_partial_model_spans_to_full_words() -> None:
    redactor = Redactor(
        detector=StaticDetector(
            [
                Span(8, 23, "ORGANIZATION_NAME"),
                Span(38, 40, "PERSON_NAME"),
            ]
        )
    )

    text = "Albo do Stefy Marketingu. Ewentualnie paweł kowalski ci powie co i jak."
    result = redactor.redact_text(text)

    assert result == "Albo do [REDACTED]. Ewentualnie [REDACTED] kowalski ci powie co i jak."


def test_expands_partial_email_span_to_full_attached_token() -> None:
    text = "Email jan.kowalski@example.com. Drugi: admin@test.pl!"
    redactor = Redactor(
        detector=StaticDetector(
            [
                Span(text.index("jan.kowalski"), text.index("example") + len("example"), "EMAIL"),
                Span(text.index("admin"), text.index("test") + len("test"), "EMAIL"),
            ]
        )
    )

    result = redactor.redact_text(text)

    assert result == "Email [REDACTED]. Drugi: [REDACTED]!"


def test_allow_terms_skips_matching_spans_case_insensitively() -> None:
    redactor = Redactor(
        detector=PatternDetector(
            {
                "Acme Corp": "ORGANIZATION_NAME",
                "jan.kowalski@example.com": "EMAIL",
            }
        ),
        allow_terms=["acme corp"],
    )

    result = redactor.redact_text("Acme Corp wrote to jan.kowalski@example.com")

    assert result == "Acme Corp wrote to [REDACTED]"


def test_allow_patterns_use_fullmatch_on_span_text() -> None:
    redactor = Redactor(
        detector=PatternDetector(
            {
                "85010112345": "PERSONAL_ID",
                "jan.kowalski@example.com": "EMAIL",
            }
        ),
        allow_patterns=[r"\d{11}"],
    )

    result = redactor.redact_text("PESEL 85010112345 email jan.kowalski@example.com")

    assert result == "PESEL 85010112345 email [REDACTED]"


def test_allow_invalid_pattern_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Invalid allow_patterns regex"):
        Redactor(detector=PatternDetector({}), allow_patterns=["(unclosed"])


def test_allowlist_checked_before_word_boundary_expansion() -> None:
    text = "Contact Acme Corp, jan@example.com"
    redactor = Redactor(
        detector=StaticDetector(
            [
                Span(text.index("Acme"), text.index("Corp") + len("Corp"), "ORGANIZATION_NAME"),
                Span(text.index("jan"), text.index("example") + len("example"), "EMAIL"),
            ]
        ),
        allow_terms=["Acme Corp"],
    )

    result = redactor.redact_text(text)

    # Allowed span is dropped before expansion and never merged into the adjacent email.
    assert result == "Contact Acme Corp, [REDACTED]"


def test_keeps_boundary_punctuation_outside_expanded_mask() -> None:
    text = 'Kontakt: "jan@example.com", potem (Anna Nowak)!'
    redactor = Redactor(
        detector=StaticDetector(
            [
                Span(text.index("jan"), text.index("example") + len("example"), "EMAIL"),
                Span(text.index("Anna"), text.index("Nowak") + len("Nowak"), "PERSON_NAME"),
            ]
        )
    )

    result = redactor.redact_text(text)

    assert result == 'Kontakt: "[REDACTED]", potem ([REDACTED])!'


def test_load_allow_file_parses_terms_patterns_comments_and_blanks(tmp_path: Path) -> None:
    allow_file = tmp_path / "allow.txt"
    allow_file.write_text(
        "\n".join(
            [
                "# company names",
                "Acme Corp",
                "",
                "  support@acme.com  ",
                "# id patterns",
                "re: \\d{11}",
                "re:ACME-\\d{4}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    terms, patterns = load_allow_file(allow_file)

    assert terms == ["Acme Corp", "support@acme.com"]
    assert patterns == [r"\d{11}", r"ACME-\d{4}"]


def test_load_allow_file_missing_raises_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_allow_file(tmp_path / "missing.txt")
