"""Classify a request by its *purpose*, so the dashboard can keep an agent's
real conversation distinct from the background chatter it fires alongside it.

AI CLIs interleave the user-facing conversation with small side-channel calls:
a command-safety classifier before each tool run, a title/topic/summary helper
that re-feeds a flattened transcript, and so on. They share the same model,
source, and auth, so only the *content* tells them apart. We bucket each request
into one of three kinds from stable markers in its system/user text:

* ``"safety"``  — the tool/command permission classifier.
* ``"helper"``  — title, topic-detection, and summarization side-channels.
* ``"main"``    — everything else: the actual agent conversation (the default).

The markers are matched case-insensitively as substrings and are deliberately
easy to extend — add a phrase to the relevant tuple when a tool ships a new
side-channel. Classification runs on the *original* body before anonymization;
none of the markers are PII, and reading the system prompt verbatim is the whole
point. Unknown shapes fall through to ``"main"`` so a real turn is never hidden.
"""

from __future__ import annotations

from typing import Any

MAIN = "main"
SAFETY = "safety"
HELPER = "helper"

# Cap on how much text we scan per request — markers live near the top of the
# system/user content, so this bounds work on large histories cheaply.
_SCAN_CHARS = 16_000

# Command/tool safety-classifier signatures (Claude Code fires one before each
# tool call). Distinctive enough that a normal turn won't trip them.
_SAFETY_MARKERS: tuple[str, ...] = (
    "stage 1 does not apply user intent",
    "err on the side of blocking",
    "you are a command",  # "You are a command-safety classifier…"
    "command injection",
    "<policy_spec>",
)

# Side-channel helper signatures: title/topic/summary requests re-send a
# flattened transcript wrapped in <session>/<transcript> and ask for a short
# label or summary. Checked only after safety (a safety call also carries a
# <transcript>, and must win).
_HELPER_MARKERS: tuple[str, ...] = (
    "<session>",
    "<transcript>",
    "isnewtopic",
    "new conversation topic",
    "word title",  # "write a 5-10 word title…"
    "concise title",
    "summarize the conversation",
    "summary of the conversation",
)


def _add(parts: list[str], budget: int, value: Any) -> int:
    """Append ``value`` (if a non-empty str) to ``parts`` until ``budget`` runs out."""
    if budget > 0 and isinstance(value, str) and value:
        parts.append(value)
        budget -= len(value)
    return budget


def _collect_text(wire: str, body: dict[str, Any]) -> str:
    """Gather the system + user-authored text relevant to classification.

    Tool output and assistant turns are skipped: the markers we match on live in
    the system prompt and the (helper-injected) user content, and scanning data
    blobs only adds false-positive surface. Result is lowercased and capped.
    """
    parts: list[str] = []
    budget = _SCAN_CHARS

    def blocks(content: Any) -> None:
        nonlocal budget
        if isinstance(content, str):
            budget = _add(parts, budget, content)
        elif isinstance(content, list):
            for b in content:
                if isinstance(b, dict):
                    budget = _add(parts, budget, b.get("text"))

    if wire == "anthropic":
        blocks(body.get("system"))
        for m in body.get("messages", []):
            if isinstance(m, dict) and m.get("role") == "user":
                blocks(m.get("content"))
    elif wire == "openai_chat":
        for m in body.get("messages", []):
            if isinstance(m, dict) and m.get("role") in ("system", "user"):
                blocks(m.get("content"))
    elif wire == "openai_responses":
        budget = _add(parts, budget, body.get("instructions"))
        inp = body.get("input")
        if isinstance(inp, str):
            budget = _add(parts, budget, inp)
        elif isinstance(inp, list):
            for item in inp:
                if isinstance(item, dict) and item.get("role") in ("system", "user"):
                    blocks(item.get("content"))

    return " ".join(parts)[:_SCAN_CHARS].lower()


def classify_kind(wire: str, body: dict[str, Any]) -> str:
    """Bucket a request as ``"main"``, ``"safety"``, or ``"helper"``."""
    text = _collect_text(wire, body)
    if not text:
        return MAIN
    if any(marker in text for marker in _SAFETY_MARKERS):
        return SAFETY
    if any(marker in text for marker in _HELPER_MARKERS):
        return HELPER
    return MAIN
