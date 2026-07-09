"""Wire-format-aware text transforms for the proxy.

Each request/response transform walks the *known* structure of a given API
format and rewrites only the human-text fields in place, leaving everything else
untouched.

Request transforms receive a single ``anon`` callback with the signature::

    anon(text: str, author: Author, novel: bool) -> str

``Author`` classifies the segment's origin so the proxy can decide which
segments to save in the audit store:

* ``Author.USER``    — text the human typed (savable).
* ``Author.TOOL``    — output produced by the user's local tools/files (savable).
* ``Author.MACHINE`` — system prompts, instruction blocks, assistant turns, and
  LLM-authored tool-call arguments (never saved).

``novel`` indicates whether the segment is new on this request turn.  AI tools
running in stateless multi-turn mode re-submit the full conversation history on
every request; only content that appears *after* the last assistant message in
the history is new (``novel=True``).  The proxy uses ``novel`` to avoid
re-saving the same user text on every subsequent turn.  On the first turn (no
assistant message yet) every segment is novel.

The callback always anonymizes regardless of author/novel so upstream/model
processing is unaffected by the save/skip classification.

``deanon`` used by response transforms is a plain ``str -> str`` callable.

Supported formats:

* ``anthropic``         — Anthropic Messages API (Claude Code)
* ``openai_chat``       — OpenAI Chat Completions (Cursor, Codex chat)
* ``openai_responses``  — OpenAI Responses API (Codex default)
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from enum import Enum, auto
from typing import Any

TextFn = Callable[[str], str]  # internal helper type: simple str -> str
AnonFn = Callable[[str, "Author", bool], str]  # public transform callback type


class Author(Enum):
    """Origin classification for a text segment passed to the ``anon`` callback."""

    USER = auto()  # human-typed message text
    TOOL = auto()  # tool/file output (data the user's local tools produced)
    MACHINE = auto()  # system prompts, instructions, assistant turns, tool-call args


def _bind(anon: AnonFn, author: Author, novel: bool) -> TextFn:
    """Return a simple ``str -> str`` that fixes ``author`` and ``novel``."""

    def fn(text: str) -> str:
        return anon(text, author, novel)

    return fn


def _user_text(anon: AnonFn, novel: bool) -> TextFn:
    """Like ``_bind(anon, Author.USER, novel)`` but downgrades harness-injected
    system blocks (see ``_is_injected_system_text``) to MACHINE so they are
    anonymized for upstream yet never saved."""

    def fn(text: str) -> str:
        author = Author.MACHINE if _is_injected_system_text(text) else Author.USER
        return anon(text, author, novel)

    return fn


# --- shared helpers ---------------------------------------------------------


def _map_text_blocks(content: Any, fn: TextFn, *, text_key: str = "text") -> Any:
    """Apply ``fn`` to a content value that is either a string or a block list."""
    if isinstance(content, str):
        return fn(content)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get(text_key), str):
                block[text_key] = fn(block[text_key])
    return content


def _walk_strings(value: Any, fn: TextFn) -> Any:
    """Apply ``fn`` to every string value in a JSON-ish structure (keys untouched)."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, list):
        return [_walk_strings(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _walk_strings(item, fn) for key, item in value.items()}
    return value


# --- Anthropic Messages -----------------------------------------------------

# Extended-thinking blocks are model-signed: ``signature`` is computed upstream
# over the exact thinking bytes, and Anthropic rejects a re-submitted thinking
# block that was altered. They must therefore round-trip through the proxy
# verbatim in *both* directions (request history and response). That is also
# privacy-safe: the model only ever sees anonymized input, so thinking text can
# contain placeholders but never raw PII — which is equally why responses must
# not de-anonymize it (restoring real values would change the signed bytes on
# the next turn's re-submission). ``redacted_thinking`` carries opaque
# encrypted ``data`` and gets the same treatment. The type strings are pinned
# against the anthropic SDK's official types in tests/test_wire_fidelity.py.
_ANTHROPIC_SIGNED_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})

# Claude Code's subscription (Max/Pro OAuth) requests carry this exact string as
# the first system block. Anthropic validates it as an anti-spoofing check and
# rejects OAuth requests whose identifier has been altered. It must therefore
# reach the upstream verbatim — but the detector tags "Claude Code"/"Anthropic"
# in it as PERSON_NAME, so we explicitly preserve the identifier preamble and
# anonymize only the text that follows it.
CLAUDE_CODE_SYSTEM_IDENTIFIER = "You are Claude Code, Anthropic's official CLI for Claude."

# Some CLIs embed harness-authored context inside a *user* turn rather than the
# system field. Claude Code wraps injected agent/skill/context info in
# <system-reminder> tags; the user did not type these, so they must be
# anonymized for upstream but never saved. To support a new tool, add its
# wrapper tag here.
# Some CLIs embed harness-authored content inside a *user* turn rather than the
# system field: injected context blocks, slash-command output, side-channel
# "suggestion"/title helper prompts, and flattened transcripts re-fed as input.
# The user did not type these, so they must be anonymized for upstream but never
# saved. The rules below are deliberately generic so a new tool rarely needs a
# new entry; when one does, extend the matching tuple.
#
# Rule 1 — a block wholly wrapped in a matching XML-ish tag, e.g.
#   <system-reminder>…</system-reminder>, <session>…</session>,
#   <local-command-stdout>…</local-command-stdout>, <ide_selection>…</ide_selection>.
_WRAPPED_BLOCK_RE = re.compile(r"^<([A-Za-z][\w-]*)\b[^>]*>.*</\1\s*>$", re.DOTALL)

# Rule 2 — a flattened transcript: side-channel requests (title/topic/summary
# helpers) re-send history as one block prefixed with role labels. A genuinely
# typed message never begins with one of these.
_TRANSCRIPT_PREFIXES: tuple[str, ...] = ("User:", "Assistant:", "Human:", "System:")

# Rule 3 — a bracketed system directive that drives a helper request, e.g.
#   "[SUGGESTION MODE: …]". Add new directive prefixes here.
_BRACKET_DIRECTIVES: tuple[str, ...] = ("[SUGGESTION MODE:",)

# Rule 4 — a block that *opens* with a known harness wrapper tag. Rule 1 only
# fires when the whole string is one wrapped block; it misses the shapes the
# harness actually emits when several wrappers are concatenated into one text
# block, or a wrapper is followed by trailing text:
#   * Claude Code slash commands —
#       <command-name>/foo</command-name><command-message>…</command-message>
#       <command-args>…</command-args><local-command-stdout>…</local-command-stdout>
#   * sibling context blocks — <ide_selection>…</ide_selection><system-reminder>…
# Unlike Rule 1 this only flags an allow-list of known harness tags, so genuine
# user-pasted markup ("<div>x</div> and more") isn't swept up. Codex wrappers
# (<environment_context>, <user_instructions>) arrive as their own whole blocks
# and are already covered by Rule 1, but are listed here for completeness.
_HARNESS_TAGS: frozenset[str] = frozenset(
    {
        "system-reminder",
        "command-name",
        "command-message",
        "command-args",
        "local-command-stdout",
        "local-command-stderr",
        "ide_selection",
        "session",
        "transcript",
        "user_instructions",
        "environment_context",
    }
)
_LEADING_TAG_RE = re.compile(r"^<([A-Za-z][\w-]*)\b[^>]*>")


def _is_injected_system_text(text: str) -> bool:
    """True if ``text`` is harness-injected system/helper content, not user-typed."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith(_TRANSCRIPT_PREFIXES):
        return True
    if stripped.startswith(_BRACKET_DIRECTIVES):
        return True
    if _WRAPPED_BLOCK_RE.match(stripped):
        return True
    lead = _LEADING_TAG_RE.match(stripped)
    return bool(lead and lead.group(1).lower() in _HARNESS_TAGS)


def _anon_preserving_identifier(text: str, fn: TextFn) -> str:
    """Anonymize ``text`` while keeping a leading Claude Code identifier verbatim."""
    if text.startswith(CLAUDE_CODE_SYSTEM_IDENTIFIER):
        rest = text[len(CLAUDE_CODE_SYSTEM_IDENTIFIER) :]
        return CLAUDE_CODE_SYSTEM_IDENTIFIER + (fn(rest) if rest else "")
    return fn(text)


def _anthropic_system(system: Any, machine: TextFn) -> Any:
    """Rewrite the Anthropic ``system`` field via the machine callback (never saved).

    Subscription/OAuth auth needs the identifier preamble to reach Anthropic
    unchanged; everything after it (the user's CLAUDE.md, environment, etc.) is
    still anonymized. ``system`` is a plain string or a list of text blocks.
    """
    if isinstance(system, str):
        return _anon_preserving_identifier(system, machine)
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                block["text"] = _anon_preserving_identifier(block["text"], machine)
        return system
    return system


def _anthropic_content(
    content: Any,
    text_fn: TextFn,
    data_fn: TextFn,
    machine: TextFn,
) -> Any:
    """Rewrite text in Anthropic content: plain text blocks AND tool blocks.

    ``text_fn`` is applied to plain text (bound to USER author + is_new).
    ``data_fn`` is applied to ``tool_result`` content (file/command data).
    ``machine`` is applied to ``tool_use`` inputs (LLM-authored function
    arguments — never saved).

    Signed thinking blocks are skipped entirely (see
    ``_ANTHROPIC_SIGNED_BLOCK_TYPES``); unknown block types keep the generic
    "rewrite any text field" fallback — over-anonymizing is the safe default
    for everything that is not signature-protected.
    """
    if isinstance(content, str):
        return text_fn(content)
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in _ANTHROPIC_SIGNED_BLOCK_TYPES:
                continue
            if isinstance(block.get("text"), str):
                block["text"] = text_fn(block["text"])
            if block.get("type") == "tool_result" and "content" in block:
                block["content"] = _anthropic_content(block["content"], data_fn, data_fn, machine)
            elif block.get("type") == "tool_use" and "input" in block:
                block["input"] = _walk_strings(block["input"], machine)
    return content


def anthropic_request(body: dict[str, Any], anon: AnonFn) -> None:
    machine = _bind(anon, Author.MACHINE, False)
    if "system" in body:
        body["system"] = _anthropic_system(body["system"], machine)
        if body.get("system") is None:
            body.pop("system", None)
    messages = body.get("messages", [])
    # Only content from the latest user turn (after the last assistant message)
    # is new. Earlier history has already been audited on previous requests.
    last_assistant = max(
        (i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "assistant"),
        default=-1,
    )
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or "content" not in message:
            continue
        role = message.get("role")
        is_new = i > last_assistant
        text_fn = _user_text(anon, is_new) if role == "user" else machine
        data_fn = _bind(anon, Author.TOOL, is_new)
        message["content"] = _anthropic_content(message["content"], text_fn, data_fn, machine)


def anthropic_response(body: dict[str, Any], deanon: TextFn) -> None:
    content = body.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict) or block.get("type") in _ANTHROPIC_SIGNED_BLOCK_TYPES:
            continue
        if isinstance(block.get("text"), str):
            block["text"] = deanon(block["text"])


# --- OpenAI Chat Completions ------------------------------------------------


def openai_chat_request(body: dict[str, Any], anon: AnonFn) -> None:
    machine = _bind(anon, Author.MACHINE, False)
    messages = body.get("messages", [])
    # Only content after the last assistant message is new for this turn.
    last_assistant = max(
        (i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "assistant"),
        default=-1,
    )
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        is_new = i > last_assistant
        # "user" role: plain text the human typed.
        # "tool" role: output from a tool (file contents, command results).
        # Both are savable only when they are new (not historical re-submissions).
        if role == "user":
            text_fn = _user_text(anon, is_new)
        elif role == "tool":
            text_fn = _bind(anon, Author.TOOL, is_new)
        else:
            text_fn = machine
        if "content" in message:
            message["content"] = _map_text_blocks(message["content"], text_fn)
        # Tool-call arguments are LLM-authored; always machine regardless of novelty.
        for call in message.get("tool_calls") or []:
            fn_block = call.get("function") if isinstance(call, dict) else None
            if isinstance(fn_block, dict) and isinstance(fn_block.get("arguments"), str):
                fn_block["arguments"] = machine(fn_block["arguments"])


def openai_chat_response(body: dict[str, Any], deanon: TextFn) -> None:
    for choice in body.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            message["content"] = deanon(message["content"])


# --- OpenAI Responses -------------------------------------------------------


def openai_responses_request(body: dict[str, Any], anon: AnonFn) -> None:
    machine = _bind(anon, Author.MACHINE, False)
    # instructions = system-level prompt; not user-authored, always machine.
    if isinstance(body.get("instructions"), str):
        body["instructions"] = machine(body["instructions"])
    inp = body.get("input")
    if isinstance(inp, str):
        # Top-level string input is the user's message (always new — no history).
        body["input"] = _user_text(anon, True)(inp)
    elif isinstance(inp, list):
        # In stateless mode Codex re-submits the full conversation history.
        # Only items after the last assistant-generated entry are new.
        # Both role=="assistant" messages and type=="function_call" items
        # (LLM-generated tool invocations re-submitted as input history) mark
        # the assistant boundary.
        last_assistant = max(
            (
                i
                for i, m in enumerate(inp)
                if isinstance(m, dict)
                and (m.get("role") == "assistant" or m.get("type") == "function_call")
            ),
            default=-1,
        )
        for i, item in enumerate(inp):
            if not isinstance(item, dict):
                continue
            # type=="reasoning" items (re-submitted model reasoning) must pass
            # through verbatim: ``encrypted_content`` is opaque and verified
            # upstream, and the summary text is model-authored from anonymized
            # input, so it is placeholder-only by construction. They carry
            # none of the keys rewritten below ("content"/"output"/
            # "arguments"), so they fall through untouched — pinned in
            # tests/test_wire_fidelity.py.
            is_new = i > last_assistant
            role = item.get("role")
            text_fn = _user_text(anon, is_new) if role == "user" else machine
            if "content" in item:
                item["content"] = _map_text_blocks(item["content"], text_fn)
            # function_call_output: tool return data — savable when new.
            # function_call arguments: LLM-authored — always machine.
            if item.get("type") == "function_call_output" and isinstance(item.get("output"), str):
                item["output"] = _bind(anon, Author.TOOL, is_new)(item["output"])
            elif item.get("type") == "function_call" and isinstance(item.get("arguments"), str):
                item["arguments"] = machine(item["arguments"])


def openai_responses_response(body: dict[str, Any], deanon: TextFn) -> None:
    for item in body.get("output", []):
        if isinstance(item, dict):
            _map_text_blocks(item.get("content"), deanon)
    if isinstance(body.get("output_text"), str):
        body["output_text"] = deanon(body["output_text"])


# --- conversation grouping --------------------------------------------------


def _first_user_text_anthropic(body: dict[str, Any]) -> str | None:
    """Extract the first user-*typed* text from an Anthropic Messages API request.

    Harness-injected blocks (``<system-reminder>`` context, slash-command
    wrappers — see ``_is_injected_system_text``) are skipped: a session that
    opens with e.g. Claude Code's ``/clear`` output would otherwise key every
    such session on the same near-constant injected text instead of the prompt
    the user actually typed.
    """
    messages = body.get("messages", [])
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                if not _is_injected_system_text(content):
                    return content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text")
                        if isinstance(text, str) and not _is_injected_system_text(text):
                            return text
    return None


def _first_user_text_openai_chat(body: dict[str, Any]) -> str | None:
    """Extract first user-typed text from an OpenAI Chat Completions request.

    Skips harness-injected blocks; see ``_first_user_text_anthropic``.
    """
    messages = body.get("messages", [])
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                if not _is_injected_system_text(content):
                    return content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text")
                        if isinstance(text, str) and not _is_injected_system_text(text):
                            return text
    return None


def _first_user_text_openai_responses(body: dict[str, Any]) -> str | None:
    """Extract first user-typed text from an OpenAI Responses API request.

    Skips harness-injected blocks (Codex wraps environment/instructions in
    ``<environment_context>``/``<user_instructions>``); see
    ``_first_user_text_anthropic``.
    """
    inp = body.get("input")
    if isinstance(inp, str):
        return None if _is_injected_system_text(inp) else inp
    if isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content")
                if isinstance(content, str):
                    if not _is_injected_system_text(content):
                        return content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            text = block.get("text")
                            if isinstance(text, str) and not _is_injected_system_text(text):
                                return text
    return None


def _explicit_session_id_openai_responses(body: dict[str, Any]) -> str | None:
    """Look for a client-supplied per-conversation id on a Responses API call.

    Confirmed by capturing real Codex CLI traffic: Codex sends a top-level
    ``prompt_cache_key`` and a ``client_metadata.{session_id,thread_id}`` that
    all hold the *same* UUID, stay identical across every turn of one
    conversation, and change together when the user starts a new conversation
    (e.g. Codex's ``/clear``). ``turn_id`` inside ``client_metadata`` changes
    every turn and must not be used. Prefer the standard ``prompt_cache_key``
    field (also settable by other Responses-API clients); fall back to
    Codex's ``client_metadata`` in case a client only sets one of the two.
    """
    prompt_cache_key = body.get("prompt_cache_key")
    if isinstance(prompt_cache_key, str) and prompt_cache_key:
        return prompt_cache_key
    client_metadata = body.get("client_metadata")
    if isinstance(client_metadata, dict):
        for field in ("thread_id", "session_id"):
            value = client_metadata.get(field)
            if isinstance(value, str) and value:
                return value
    return None


_LEGACY_SESSION_ID_RE = re.compile(
    r"_session_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)


def _explicit_session_id_anthropic(body: dict[str, Any]) -> str | None:
    """Look for Claude Code's per-session id on a Messages API call.

    Confirmed by capturing real Claude Code traffic: every request carries
    ``metadata.user_id`` holding a JSON-encoded object whose ``session_id``
    equals the local transcript filename in ``~/.claude/projects/<project>/``
    and changes when a new session starts (``/clear``, new window). Matching
    the transcript stem also makes live ids line up with what the history
    importer records (``importer/runner.py`` uses the same stem), so imports
    dedupe against already-proxied sessions. Older Claude Code versions sent
    a plain string ending in ``_session_<uuid>``; take the uuid from that
    shape too.

    Never fall back to the raw ``user_id`` itself: the Messages API documents
    it as a stable per-*user* identifier, so a generic client's value would
    merge all of that user's conversations into one.
    """
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        return None
    user_id = metadata.get("user_id")
    if not isinstance(user_id, str) or not user_id:
        return None
    try:
        parsed = json.loads(user_id)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        session_id = parsed.get("session_id")
        if isinstance(session_id, str) and session_id:
            return session_id
    match = _LEGACY_SESSION_ID_RE.search(user_id)
    return match.group(1) if match else None


def conversation_key(wire: str, body: dict[str, Any]) -> str | None:
    """Derive a stable conversation identifier for a request.

    Prefers an explicit, client-supplied per-conversation id when the wire
    format offers one (see ``_explicit_session_id_anthropic`` and
    ``_explicit_session_id_openai_responses``). Falls back to hashing the
    first user-typed message text: since AI APIs are stateless and resend
    full history on each turn, that text is a stable, deterministic prefix
    shared by every turn of the same conversation. We hash it to keep it
    opaque.

    For Cursor hook events, check for an explicit conversation_id field in
    the body; if absent, return None (each hook stays ungrouped).

    Known limitation: for clients without an explicit id, if history is
    compacted/summarized mid-session, the first message text can change,
    splitting one real conversation into multiple conversation_id values.
    Acceptable for v1.

    Args:
        wire: Wire format string ("anthropic", "openai_chat", "openai_responses", etc.)
        body: Request body dict

    Returns:
        A stable conversation id, or None if no signal was found.
    """
    if wire == "cursor":
        # Cursor hook events are single-shot; no history resent. If the hook
        # payload includes an explicit conversation_id, use it verbatim.
        conv_id = body.get("conversation_id")
        return str(conv_id) if conv_id else None

    if wire == "anthropic":
        explicit = _explicit_session_id_anthropic(body)
        if explicit:
            return explicit

    if wire == "openai_responses":
        explicit = _explicit_session_id_openai_responses(body)
        if explicit:
            return explicit

    # For stateless API formats without an explicit id, extract first user text.
    first_user = None
    if wire == "anthropic":
        first_user = _first_user_text_anthropic(body)
    elif wire == "openai_chat":
        first_user = _first_user_text_openai_chat(body)
    elif wire == "openai_responses":
        first_user = _first_user_text_openai_responses(body)

    if not first_user:
        return None

    # Hash to 16 chars to keep it opaque (not storing raw user text).
    digest = hashlib.sha256(first_user.encode()).hexdigest()
    return digest[:16]


# --- registry + usage extraction --------------------------------------------

REQUEST_TRANSFORMS: dict[str, Callable[[dict[str, Any], AnonFn], None]] = {
    "anthropic": anthropic_request,
    "openai_chat": openai_chat_request,
    "openai_responses": openai_responses_request,
}

RESPONSE_TRANSFORMS: dict[str, Callable[[dict[str, Any], TextFn], None]] = {
    "anthropic": anthropic_response,
    "openai_chat": openai_chat_response,
    "openai_responses": openai_responses_response,
}


def extract_tokens(wire: str, body: dict[str, Any]) -> tuple[int | None, int | None]:
    """Pull (input_tokens, output_tokens) from a response's usage block."""
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    assert isinstance(usage, dict)
    if wire == "openai_chat":
        return usage.get("prompt_tokens"), usage.get("completion_tokens")
    return usage.get("input_tokens"), usage.get("output_tokens")
