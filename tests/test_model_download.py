import os

import pytest


@pytest.mark.skipif(os.getenv("PII_RUN_MODEL_TESTS") != "1", reason="set PII_RUN_MODEL_TESTS=1 to download and run the model")
def test_model_downloads_to_configured_cache_and_redacts_email(monkeypatch, tmp_path) -> None:
    from privacy_kit import Redactor

    monkeypatch.setenv("PII_MODEL_CACHE_DIR", str(tmp_path / "hf-cache"))

    result = Redactor().redact_text("Kontakt: jan.kowalski@example.com")

    assert "jan.kowalski@example.com" not in result
    assert "[REDACTED]" in result
    assert any((tmp_path / "hf-cache").iterdir())
