"""Classify a request by its *purpose*, so the dashboard can keep an agent's
real conversation distinct from the background chatter it fires alongside it.

AI CLIs interleave the user-facing conversation with small side-channel calls:
a command-safety classifier before each tool run, a title/topic/summary helper
that re-feeds a flattened transcript, and so on. They share the same model,
source, and auth, so only the *content* tells them apart. We bucket each request
into one of three kinds from stable markers in its system/user text:

* ``"safety"``  — the tool/command permission classifier.
* ``"helper"``  — title, topic-detection, and summarization side-channels,
  plus Claude Code's tiny usage-limit probes (see ``_is_quota_probe``).
* ``"main"``    — everything else: the actual agent conversation (the default).

The markers are matched case-insensitively as substrings and are deliberately
easy to extend — add a phrase to the relevant tuple when a tool ships a new
side-channel. Classification runs on the *original* body before anonymization;
none of the markers are PII, and reading the system prompt verbatim is the whole
point. Unknown shapes fall through to ``"main"`` so a real turn is never hidden.
"""

from __future__ import annotations

import re
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


# Harness *context* wrappers that ride along inside a real user turn: memory
# recall and CLAUDE.md content (<system-reminder>), the user's editor selection
# (<ide_selection>), slash-command output, Codex environment blocks. Their
# content quotes arbitrary file text — including, when someone works on this
# very repo, the literal marker strings above — so text in these wrappers must
# never vote on a request's kind. Deliberately narrower than transform.py's
# _HARNESS_TAGS: <session>/<transcript> wrappers are the helper side-channels'
# own payload and must stay scannable, or helper detection breaks.
_CONTEXT_TAGS: frozenset[str] = frozenset(
    {
        "system-reminder",
        "ide_selection",
        "command-name",
        "command-message",
        "command-args",
        "local-command-stdout",
        "local-command-stderr",
        "user_instructions",
        "environment_context",
    }
)
_LEADING_TAG_RE = re.compile(r"^\s*<([A-Za-z][\w-]*)\b")


def _is_context_wrapper(text: str) -> bool:
    """True if ``text`` opens with a harness context tag (whole block skipped)."""
    lead = _LEADING_TAG_RE.match(text)
    return bool(lead and lead.group(1).lower() in _CONTEXT_TAGS)


def _add(parts: list[str], budget: int, value: Any) -> int:
    """Append ``value`` (if a non-empty str) to ``parts`` until ``budget`` runs out."""
    if budget > 0 and isinstance(value, str) and value and not _is_context_wrapper(value):
        parts.append(value)
        budget -= len(value)
    return budget


def _collect_text(wire: str, body: dict[str, Any]) -> str:
    """Gather the system prompt + newest user turn's text relevant to classification.

    Tool output and assistant turns are skipped: the markers we match on live in
    the system prompt and the (helper-injected) user content, and scanning data
    blobs only adds false-positive surface. Only the *last* user-role message is
    scanned, not the whole resent history: APIs are stateless and resend every
    prior turn (including tool results, which Anthropic packages as role="user"
    content) on each request, so a marker phrase appearing anywhere in an older
    turn — e.g. a grep/read of this very file — would otherwise misclassify
    every later request in the session. Only the newest turn tells us what
    *this* request is for. Result is lowercased and capped.
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

    def last_with_role(items: list[Any], roles: tuple[str, ...]) -> dict[str, Any] | None:
        for item in reversed(items):
            if isinstance(item, dict) and item.get("role") in roles:
                return item
        return None

    if wire == "anthropic":
        blocks(body.get("system"))
        last_user = last_with_role(body.get("messages", []), ("user",))
        if last_user is not None:
            blocks(last_user.get("content"))
    elif wire == "openai_chat":
        messages = body.get("messages", [])
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                blocks(m.get("content"))
        last_user = last_with_role(messages, ("user",))
        if last_user is not None:
            blocks(last_user.get("content"))
    elif wire == "openai_responses":
        budget = _add(parts, budget, body.get("instructions"))
        inp = body.get("input")
        if isinstance(inp, str):
            budget = _add(parts, budget, inp)
        elif isinstance(inp, list):
            last_user = last_with_role(inp, ("user",))
            if last_user is not None:
                blocks(last_user.get("content"))

    return " ".join(parts)[:_SCAN_CHARS].lower()


def _is_quota_probe(wire: str, body: dict[str, Any]) -> bool:
    """True for Claude Code's usage-limit probes.

    Claude Code periodically checks whether the account has quota left by
    sending a minimal request: a single user message whose whole text is
    "quota" with ``max_tokens: 1`` (answered with ``#``). It is background
    chatter, not a real turn. Matched narrowly — exact whole-message text
    plus the tiny token cap — so a genuine prompt mentioning quota can
    never trip it.
    """
    if wire != "anthropic":
        return False
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 2:
        return False
    messages = body.get("messages", [])
    if len(messages) != 1 or not isinstance(messages[0], dict):
        return False
    content = messages[0].get("content")
    if isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict):
        content = content[0].get("text")
    return isinstance(content, str) and content.strip().lower() == "quota"


def classify_kind(wire: str, body: dict[str, Any]) -> str:
    """Bucket a request as ``"main"``, ``"safety"``, or ``"helper"``."""
    if _is_quota_probe(wire, body):
        return HELPER
    text = _collect_text(wire, body)
    if not text:
        return MAIN
    if any(marker in text for marker in _SAFETY_MARKERS):
        return SAFETY
    if any(marker in text for marker in _HELPER_MARKERS):
        return HELPER
    return MAIN
