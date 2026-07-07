from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from privacy_kit.core.types import Span
from privacy_kit.integrations.langsmith import make_anonymizer, make_client


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


def patch_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "privacy_kit.integrations.langsmith.build_detector",
        lambda *_args, **_kwargs: PatternDetector(
            {
                "jan.kowalski@example.com": "EMAIL",
                "+48 501 222 333": "PHONE_NUMBER",
                "Acme Corp": "ORGANIZATION_NAME",
                "85010112345": "PERSONAL_ID",
            }
        ),
    )


def test_make_anonymizer_redacts_nested_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_detector(monkeypatch)
    anonymizer = make_anonymizer()

    result = anonymizer(
        {
            "messages": [{"role": "user", "content": "Email jan.kowalski@example.com"}],
            "metadata": {"customer_contact": "+48 501 222 333"},
        }
    )

    assert result["messages"][0]["content"] == "Email [REDACTED]"
    assert result["metadata"]["customer_contact"] == "[REDACTED]"


def test_make_anonymizer_respects_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_detector(monkeypatch)
    anonymizer = make_anonymizer(allow_terms=["Acme Corp"])

    result = anonymizer({"input": "Acme Corp emailed jan.kowalski@example.com"})

    assert result["input"] == "Acme Corp emailed [REDACTED]"


def test_make_anonymizer_reads_allow_terms_env(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_detector(monkeypatch)
    monkeypatch.setenv("PII_ALLOW_TERMS", "Acme Corp")

    anonymizer = make_anonymizer()

    assert anonymizer("From Acme Corp") == "From Acme Corp"


def test_make_anonymizer_reads_allow_patterns_env(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_detector(monkeypatch)
    monkeypatch.setenv("PII_ALLOW_PATTERNS", r"\d{11}")

    anonymizer = make_anonymizer()

    assert anonymizer("ID 85010112345") == "ID 85010112345"


def test_make_anonymizer_reads_allow_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    patch_detector(monkeypatch)
    allow_file = tmp_path / "allow.txt"
    allow_file.write_text("# companies\nAcme Corp\nre: \\d{11}\n", encoding="utf-8")

    anonymizer = make_anonymizer(allow_file=str(allow_file))

    assert anonymizer("Acme Corp, PESEL 85010112345") == "Acme Corp, PESEL 85010112345"


def test_make_anonymizer_reads_allow_file_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patch_detector(monkeypatch)
    allow_file = tmp_path / "allow.txt"
    allow_file.write_text("Acme Corp\n", encoding="utf-8")
    monkeypatch.setenv("PII_ALLOW_FILE", str(allow_file))

    anonymizer = make_anonymizer()

    assert anonymizer("From Acme Corp") == "From Acme Corp"


def test_make_client_passes_anonymizer_and_hide_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_detector(monkeypatch)
    created: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)

    fake_langsmith = types.ModuleType("langsmith")
    fake_langsmith.Client = FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langsmith", fake_langsmith)

    make_client(api_key="ls-test")

    assert len(created) == 1
    assert created[0]["hide_metadata"] is True
    assert created[0]["api_key"] == "ls-test"
    assert callable(created[0]["anonymizer"])


def test_make_client_has_clear_missing_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_detector(monkeypatch)
    real_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langsmith" or name.startswith("langsmith."):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(RuntimeError, match=r"privacy-kit\[langsmith\]"):
        make_client()
