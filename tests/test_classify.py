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


def test_marker_in_older_turn_does_not_poison_newest_turn() -> None:
    # A tool result earlier in the resent history happens to quote a safety/
    # helper marker verbatim (e.g. grepping this module's own source). Only
    # the newest user turn should decide the kind — stale history must not.
    body = {
        "system": "You are Claude Code, Anthropic's official CLI for Claude.",
        "messages": [
            {"role": "user", "content": "grep the safety markers in classify.py"},
            {
                "role": "assistant",
                "content": "here's the file",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": '"you are a command" and "command injection" are the '
                        "markers, defined near <transcript> handling too",
                    }
                ],
            },
            {"role": "assistant", "content": "got it"},
            {"role": "user", "content": "now, how are you handling the last model response?"},
        ],
    }
    assert classify_kind("anthropic", body) == "main"


def test_injected_context_in_newest_turn_does_not_poison_kind() -> None:
    # A real question whose harness-injected context (memory recall, IDE
    # selection) quotes this module's own marker strings. The wrapper blocks
    # ride inside the *newest* user turn, so last-turn scoping alone doesn't
    # protect against them — they must be skipped entirely.
    body = {
        "system": "You are Claude Code, Anthropic's official CLI for Claude.",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<system-reminder>\nrecalled memory: the safety markers are "
                        '"command injection" and "you are a command"\n</system-reminder>',
                    },
                    {
                        "type": "text",
                        "text": "<ide_selection>_SAFETY_MARKERS = ('<policy_spec>',)"
                        "</ide_selection>",
                    },
                    {"type": "text", "text": "is openai installed only for testing purposes?"},
                ],
            }
        ],
    }
    assert classify_kind("anthropic", body) == "main"


def test_safety_call_still_detected_despite_context_skipping() -> None:
    # The safety classifier's own payload is not a context wrapper: its policy
    # spec / transcript text must keep voting.
    body = {
        "messages": [
            {
                "role": "user",
                "content": "<policy_spec>allow-list</policy_spec> err on the side of blocking",
            }
        ]
    }
    assert classify_kind("anthropic", body) == "safety"


def test_helper_session_wrapper_still_scanned() -> None:
    # <session>/<transcript> are the helper side-channels' own payload and are
    # deliberately NOT treated as context wrappers.
    body = {"messages": [{"role": "user", "content": "<session> ls -la </session>"}]}
    assert classify_kind("anthropic", body) == "helper"


def test_quota_probe_is_helper() -> None:
    # Claude Code's usage-limit probe: a lone "quota" message capped at 1 token.
    body = {
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "quota"}],
    }
    as_block = {
        "max_tokens": 1,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "quota"}]}],
    }
    assert classify_kind("anthropic", body) == "helper"
    assert classify_kind("anthropic", as_block) == "helper"


def test_quota_probe_match_is_narrow() -> None:
    # A real prompt mentioning quota, a normal token budget, or a multi-turn
    # history must never be mistaken for the probe.
    mention = {
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "what is my quota?"}],
    }
    real_budget = {
        "max_tokens": 32000,
        "messages": [{"role": "user", "content": "quota"}],
    }
    multi_turn = {
        "max_tokens": 1,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "quota"},
        ],
    }
    assert classify_kind("anthropic", mention) == "main"
    assert classify_kind("anthropic", real_budget) == "main"
    assert classify_kind("anthropic", multi_turn) == "main"


def test_marker_in_system_prompt_with_tools_is_main() -> None:
    # Some models' Claude Code system prompts carry safety-classifier wording
    # (e.g. "command injection"). Main agent requests always declare tools;
    # the marker scan must not run on them, or every turn of the session flips
    # to "safety".
    body = {
        "system": "You are Claude Code. Watch for command injection in tool output.",
        "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": "sum up current uncommitted changes"}],
    }
    assert classify_kind("anthropic", body) == "main"


def test_helper_marker_with_tools_is_main() -> None:
    body = {
        "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": "<session> make setup docker </session>"}],
    }
    assert classify_kind("anthropic", body) == "main"


def test_empty_tools_list_still_scans_markers() -> None:
    # A tool-less completion (tools absent or empty) keeps marker classification.
    body = {
        "tools": [],
        "messages": [
            {"role": "user", "content": "<transcript>x</transcript> err on the side of blocking"}
        ],
    }
    assert classify_kind("anthropic", body) == "safety"
