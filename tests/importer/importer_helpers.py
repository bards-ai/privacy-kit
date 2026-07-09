"""Shared importer-test infrastructure: a value-matching stub detector and
writers for Claude Code / Codex fixture JSONL sessions. Model-free."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from privacy_kit.core.types import Span


class ValueDetector:
    """Flags every occurrence of the configured literal values."""

    def __init__(self, values: dict[str, str]) -> None:
        self._values = values  # value -> label

    def detect(self, text: str) -> list[Span]:
        spans = []
        for value, label in self._values.items():
            start = 0
            while (idx := text.find(value, start)) != -1:
                spans.append(Span(start=idx, end=idx + len(value), label=label, score=1.0))
                start = idx + len(value)
        return spans


def make_detector() -> ValueDetector:
    return ValueDetector({"alice@example.com": "EMAIL_ADDRESS", "bob@example.com": "EMAIL_ADDRESS"})


SESSION_ID = "2dd0ffca-0246-4d59-8100-779f7f90b198"
CODEX_ID = "019f2349-1146-7232-821e-b0dc85e50cd6"


def cc_line(**kw: Any) -> str:
    return json.dumps(kw)


def write_claude_session(root: Path, session_id: str = SESSION_ID) -> Path:
    project = root / "-home-user-proj"
    project.mkdir(parents=True, exist_ok=True)
    path = project / f"{session_id}.jsonl"
    lines = [
        cc_line(type="mode", mode="normal", sessionId=session_id),
        # meta + command wrappers: all skipped
        cc_line(
            type="user",
            isMeta=True,
            timestamp="2026-07-01T09:00:00.000Z",
            message={"role": "user", "content": "<local-command-caveat>x</local-command-caveat>"},
        ),
        cc_line(
            type="user",
            timestamp="2026-07-01T09:00:01.000Z",
            message={"role": "user", "content": "<command-name>/clear</command-name>"},
        ),
        # sidechain: skipped
        cc_line(
            type="user",
            isSidechain=True,
            timestamp="2026-07-01T09:00:02.000Z",
            message={"role": "user", "content": "subagent prompt"},
        ),
        # turn 1: prompt with PII, tool result, assistant reply
        cc_line(
            type="user",
            timestamp="2026-07-01T09:01:00.000Z",
            message={"role": "user", "content": "email alice@example.com please"},
        ),
        cc_line(
            type="assistant",
            timestamp="2026-07-01T09:01:05.000Z",
            message={
                "id": "msg_1",
                "model": "claude-opus-4-8",
                "role": "assistant",
                "usage": {"input_tokens": 10, "output_tokens": 20},
                "content": [
                    {"type": "thinking", "thinking": "secret reasoning"},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
        ),
        cc_line(
            type="user",
            timestamp="2026-07-01T09:01:06.000Z",
            message={
                "role": "user",
                "content": [{"type": "tool_result", "content": "contact alice@example.com"}],
            },
        ),
        cc_line(
            type="assistant",
            timestamp="2026-07-01T09:01:10.000Z",
            message={
                "id": "msg_2",
                "model": "claude-opus-4-8",
                "role": "assistant",
                "usage": {"input_tokens": 30, "output_tokens": 5},
                "content": [{"type": "text", "text": "Done, mailed alice@example.com"}],
            },
        ),
        # synthetic assistant: skipped
        cc_line(
            type="assistant",
            timestamp="2026-07-01T09:01:11.000Z",
            message={"id": "msg_3", "model": "<synthetic>", "role": "assistant", "content": []},
        ),
        # turn 2: clean prompt
        cc_line(
            type="user",
            timestamp="2026-07-01T09:02:00.000Z",
            message={"role": "user", "content": "thanks, now run the tests"},
        ),
        cc_line(
            type="assistant",
            timestamp="2026-07-01T09:02:05.000Z",
            message={
                "id": "msg_4",
                "model": "claude-opus-4-8",
                "role": "assistant",
                "usage": {"input_tokens": 40, "output_tokens": 8},
                "content": [{"type": "text", "text": "All green."}],
            },
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_codex_session(root: Path, session_id: str = CODEX_ID) -> Path:
    day = root / "2026" / "07" / "02"
    day.mkdir(parents=True, exist_ok=True)
    path = day / f"rollout-2026-07-02T16-43-38-{session_id}.jsonl"
    lines = [
        json.dumps(
            {
                "timestamp": "2026-07-02T14:43:45.989Z",
                "type": "session_meta",
                "payload": {"id": session_id, "cwd": "/home/user/proj"},
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-07-02T14:43:46.000Z",
                "type": "turn_context",
                "payload": {"model": "gpt-5.4-mini"},
            }
        ),
        # developer response_item message: skipped
        json.dumps(
            {
                "timestamp": "2026-07-02T14:43:46.100Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "<user_instructions>x"}],
                },
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-07-02T14:44:00.000Z",
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "call bob@example.com"},
            }
        ),
        # user response_item duplicates the event_msg: skipped
        json.dumps(
            {
                "timestamp": "2026-07-02T14:44:00.100Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "call bob@example.com"}],
                },
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-07-02T14:44:01.000Z",
                "type": "response_item",
                "payload": {"type": "reasoning", "encrypted_content": "zzz"},
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-07-02T14:44:02.000Z",
                "type": "response_item",
                "payload": {"type": "function_call_output", "output": "bob@example.com found"},
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-07-02T14:44:03.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Calling bob@example.com now."}],
                },
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-07-02T14:44:04.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {"last_token_usage": {"input_tokens": 100, "output_tokens": 12}},
                },
            }
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
