"""Call-purpose classification: main vs safety vs helper side-channels."""

from __future__ import annotations

from privacy_kit.gateway.proxy.classify import classify_kind


def test_safety_classifier_call() -> None:
    body = {
        "system": "You are Claude Code, Anthropic's official CLI for Claude.",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<transcript>\nBash ls\n</transcript>\n Err on the side of "
                        "blocking. Stage 1 does NOT apply user intent or ALLOW exceptions",
                    }
                ],
            }
        ],
    }
    assert classify_kind("anthropic", body) == "safety"


def test_helper_session_title_call() -> None:
    body = {"messages": [{"role": "user", "content": "<session> make setup docker </session>"}]}
    assert classify_kind("anthropic", body) == "helper"


def test_main_conversation_is_default() -> None:
    body = {
        "system": "You are Claude Code, Anthropic's official CLI for Claude.",
        "messages": [{"role": "user", "content": "fix the failing test in app.py"}],
    }
    assert classify_kind("anthropic", body) == "main"


def test_safety_wins_over_helper_when_both_present() -> None:
    # A safety call also carries a <transcript> (a helper marker); safety must win.
    body = {
        "messages": [
            {"role": "user", "content": "<transcript>x</transcript> err on the side of blocking"}
        ]
    }
    assert classify_kind("anthropic", body) == "safety"


def test_tool_output_does_not_trip_helper() -> None:
    # Markers in tool/assistant content must not classify a real turn as helper.
    body = {
        "messages": [
            {"role": "assistant", "content": "<session> internal </session>"},
            {"role": "user", "content": "now summarize my code please"},
        ]
    }
    assert classify_kind("anthropic", body) == "main"


def test_empty_body_is_main() -> None:
    assert classify_kind("anthropic", {}) == "main"
    assert classify_kind("openai_responses", {}) == "main"


def test_openai_responses_main() -> None:
    body = {"instructions": "You are Codex", "input": "refactor the parser"}
    assert classify_kind("openai_responses", body) == "main"
