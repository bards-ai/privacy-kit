"""Wire-fidelity tests for the proxy transforms, validated by the vendor SDKs.

The proxy rewrites provider-native bodies in place and must forward everything
it does not deliberately rewrite byte-faithful. Instead of hand-rolled dict
fixtures, these tests build request/response bodies from the *official* typed
models shipped by the ``anthropic`` and ``openai`` SDKs (used purely as schema
oracles — never as HTTP clients), so:

* fixtures are mypy-checked against the real API schemas;
* per-format round-trip tests prove an identity transform leaves the whole
  body deep-equal (nothing is dropped, reshaped, or coerced);
* signed/opaque content — Anthropic extended-thinking blocks and OpenAI
  Responses reasoning items — is pinned as verbatim pass-through in requests,
  responses, and SSE streams (rewriting it would break upstream signature
  checks; it is placeholder-only by construction, so passing it through leaks
  nothing);
* a canary over the SDK's content-block unions fails when a vendor ships a new
  block type, forcing an explicit decision on its handling.
"""

from __future__ import annotations

import copy
import json
import typing
from typing import Any, cast

import pytest
import typing_extensions

pytest.importorskip("anthropic")
pytest.importorskip("openai")

from anthropic.types import (
    ContentBlockParam,
    Message,
    MessageParam,
    RawContentBlockDeltaEvent,
    RedactedThinkingBlockParam,
    SignatureDelta,
    TextBlock,
    TextBlockParam,
    TextDelta,
    ThinkingBlock,
    ThinkingBlockParam,
    ThinkingDelta,
    ToolResultBlockParam,
    ToolUseBlockParam,
    Usage,
)
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import Function
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseFunctionToolCallParam,
    ResponseInputParam,
    ResponseReasoningItemParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from openai.types.responses.response_reasoning_item_param import Summary

from privacy_kit.core.vault import Vault
from privacy_kit.gateway.proxy.streaming import PlaceholderStreamDecoder, rewrite_sse
from privacy_kit.gateway.proxy.transform import (
    _ANTHROPIC_SIGNED_BLOCK_TYPES,
    Author,
    anthropic_request,
    anthropic_response,
    openai_chat_request,
    openai_responses_request,
)

# --- callbacks ---------------------------------------------------------------


def _identity_anon(text: str, author: Author, novel: bool) -> str:
    return text


def _marking_anon(text: str, author: Author, novel: bool) -> str:
    return f"«{text}»"


def _marking_deanon(text: str) -> str:
    return f"«{text}»"


# --- fixtures built from the vendor SDK types --------------------------------

_THINKING = ThinkingBlockParam(
    type="thinking",
    thinking="[PERSON_NAME_1] wants the report; plan the read.",
    signature="c2lnbmVkLW92ZXItZXhhY3QtYnl0ZXM=",
)
_REDACTED = RedactedThinkingBlockParam(type="redacted_thinking", data="b3BhcXVl")


def _anthropic_request_body() -> dict[str, Any]:
    messages: list[MessageParam] = [
        MessageParam(
            role="user",
            content=[TextBlockParam(type="text", text="Hi, I'm John Smith")],
        ),
        MessageParam(
            role="assistant",
            content=[
                copy.deepcopy(_THINKING),
                copy.deepcopy(_REDACTED),
                TextBlockParam(type="text", text="Hello John Smith"),
                ToolUseBlockParam(
                    type="tool_use", id="toolu_1", name="read", input={"path": "/home/john"}
                ),
            ],
        ),
        MessageParam(
            role="user",
            content=[
                ToolResultBlockParam(
                    type="tool_result",
                    tool_use_id="toolu_1",
                    content=[TextBlockParam(type="text", text="john@example.com")],
                ),
                TextBlockParam(type="text", text="thanks, summarize it"),
            ],
        ),
    ]
    params = MessageCreateParamsNonStreaming(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system="Be concise.",
        messages=messages,
    )
    return cast(dict[str, Any], dict(params))


def _anthropic_response_body() -> dict[str, Any]:
    message = Message(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-sonnet-4-5",
        content=[
            ThinkingBlock(
                type="thinking",
                thinking="[PERSON_NAME_1] asked for a summary.",
                signature="c2ln",
            ),
            TextBlock(type="text", text="Done, [PERSON_NAME_1]."),
        ],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    return message.model_dump(mode="json", exclude_unset=True)


def _openai_chat_request_body() -> dict[str, Any]:
    messages = [
        ChatCompletionSystemMessageParam(role="system", content="Be concise."),
        ChatCompletionUserMessageParam(role="user", content="Hi, I'm John Smith"),
        ChatCompletionAssistantMessageParam(
            role="assistant",
            content="Hello John Smith",
            tool_calls=[
                ChatCompletionMessageFunctionToolCallParam(
                    type="function",
                    id="call_1",
                    function=Function(name="read", arguments='{"path": "/home/john"}'),
                )
            ],
        ),
        ChatCompletionToolMessageParam(
            role="tool", tool_call_id="call_1", content="john@example.com"
        ),
    ]
    return {"model": "gpt-5", "messages": [dict(m) for m in messages]}


_REASONING = ResponseReasoningItemParam(
    type="reasoning",
    id="rs_1",
    summary=[Summary(type="summary_text", text="[PERSON_NAME_1] wants the file read.")],
    encrypted_content="Z2NtOnNlYWxlZA==",
)


def _openai_responses_request_body() -> dict[str, Any]:
    items: ResponseInputParam = [
        EasyInputMessageParam(role="user", content="Hi, I'm John Smith"),
        copy.deepcopy(_REASONING),
        ResponseFunctionToolCallParam(
            type="function_call",
            call_id="call_1",
            name="read",
            arguments='{"path": "/home/john"}',
        ),
        FunctionCallOutput(
            type="function_call_output", call_id="call_1", output="john@example.com"
        ),
        EasyInputMessageParam(role="user", content="thanks, summarize it"),
    ]
    return {
        "model": "gpt-5",
        "instructions": "Be concise.",
        "input": [dict(item) for item in items],
    }


# --- round-trip fidelity: identity transform must be a no-op -----------------


def test_anthropic_request_identity_is_byte_faithful() -> None:
    body = _anthropic_request_body()
    body["unknown_future_field"] = {"nested": ["kept", 1, None]}
    before = copy.deepcopy(body)
    anthropic_request(body, _identity_anon)
    assert body == before


def test_openai_chat_request_identity_is_byte_faithful() -> None:
    body = _openai_chat_request_body()
    body["prompt_cache_key"] = "conv-uuid"
    before = copy.deepcopy(body)
    openai_chat_request(body, _identity_anon)
    assert body == before


def test_openai_responses_request_identity_is_byte_faithful() -> None:
    body = _openai_responses_request_body()
    body["prompt_cache_key"] = "conv-uuid"
    before = copy.deepcopy(body)
    openai_responses_request(body, _identity_anon)
    assert body == before


# --- signed thinking blocks pass through verbatim, siblings are rewritten ----


def test_anthropic_request_thinking_blocks_pass_through_verbatim() -> None:
    body = _anthropic_request_body()
    anthropic_request(body, _marking_anon)

    assistant_blocks = body["messages"][1]["content"]
    assert assistant_blocks[0] == dict(_THINKING)
    assert assistant_blocks[1] == dict(_REDACTED)
    # Everything around them is still anonymized.
    assert assistant_blocks[2]["text"] == "«Hello John Smith»"
    assert assistant_blocks[3]["input"] == {"path": "«/home/john»"}
    assert body["messages"][0]["content"][0]["text"] == "«Hi, I'm John Smith»"
    assert body["messages"][2]["content"][0]["content"][0]["text"] == "«john@example.com»"


def test_anthropic_response_thinking_blocks_are_not_deanonymized() -> None:
    body = _anthropic_response_body()
    thinking_before = copy.deepcopy(body["content"][0])
    anthropic_response(body, _marking_deanon)
    assert body["content"][0] == thinking_before
    assert body["content"][1]["text"] == "«Done, [PERSON_NAME_1].»"


def test_openai_responses_reasoning_items_pass_through_verbatim() -> None:
    body = _openai_responses_request_body()
    openai_responses_request(body, _marking_anon)

    items = body["input"]
    assert items[1] == dict(_REASONING)
    # Function-call arguments and tool output around it are still rewritten.
    assert items[2]["arguments"] == '«{"path": "/home/john"}»'
    assert items[3]["output"] == "«john@example.com»"
    assert items[4]["content"] == "«thanks, summarize it»"


# --- SSE: thinking/signature deltas stream through verbatim ------------------


def _sse_line(event: RawContentBlockDeltaEvent) -> list[str]:
    return [
        f"event: {event.type}",
        "data: " + json.dumps(event.model_dump(mode="json")),
        "",
    ]


async def _collect_sse(lines: list[str]) -> str:
    async def gen() -> typing.AsyncIterator[str]:
        for line in lines:
            yield line

    vault = Vault()
    assert vault.placeholder_for("PERSON_NAME", "John Smith") == "[PERSON_NAME_1]"
    out: list[str] = []
    async for chunk in rewrite_sse(gen(), "anthropic", PlaceholderStreamDecoder(vault)):
        out.append(chunk)
    return "".join(out)


async def test_anthropic_stream_thinking_deltas_pass_through_verbatim() -> None:
    events = [
        RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=ThinkingDelta(type="thinking_delta", thinking="[PERSON_NAME_1] wants"),
        ),
        RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=SignatureDelta(type="signature_delta", signature="c2ln"),
        ),
        RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=1,
            delta=TextDelta(type="text_delta", text="Done, [PERSON_NAME_1]."),
        ),
    ]
    lines = [line for event in events for line in _sse_line(event)]
    output = await _collect_sse(lines)

    # Thinking and signature deltas keep their placeholders/bytes untouched...
    assert "[PERSON_NAME_1] wants" in output
    assert '"signature": "c2ln"' in output
    # ...while the text delta is de-anonymized for the client.
    assert "Done, John Smith." in output
    assert '"text": "Done, [PERSON_NAME_1]."' not in output


# --- canary: a new vendor block type must get an explicit decision -----------


def _union_type_literals(union: Any) -> set[str]:
    """Collect the ``type`` Literal values of every member of a TypedDict union."""
    literals: set[str] = set()
    for member in typing.get_args(union):
        hints = typing_extensions.get_type_hints(member, include_extras=False)
        literals.update(str(v) for v in typing.get_args(hints.get("type")))
    return literals


# Every Anthropic content-block type we have looked at and decided handling
# for. "signed pass-through" types must stay in sync with the runtime set;
# everything else is covered by the generic rule in ``_anthropic_content``
# (rewrite text fields, recurse into tool_result, walk tool_use input; unknown
# fields pass through). When the anthropic SDK ships a new block type, this
# test fails: look at the new block's schema and either add it here (generic
# rule is fine) or extend the transform first.
_DECIDED_ANTHROPIC_BLOCK_TYPES = frozenset(
    {
        "text",
        "image",
        "document",
        "search_result",
        "thinking",
        "redacted_thinking",
        "tool_use",
        "tool_result",
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
        "code_execution_tool_result",
        "bash_code_execution_tool_result",
        "text_editor_code_execution_tool_result",
        "tool_search_tool_result",
        "container_upload",
        "mid_conv_system",
    }
)


def test_runtime_signed_set_matches_sdk_thinking_types() -> None:
    assert (
        _union_type_literals(ThinkingBlockParam | RedactedThinkingBlockParam)
        == _ANTHROPIC_SIGNED_BLOCK_TYPES
    )


def test_anthropic_sdk_block_types_are_all_decided() -> None:
    found = _union_type_literals(ContentBlockParam)
    new = found - _DECIDED_ANTHROPIC_BLOCK_TYPES
    assert not new, (
        f"anthropic SDK ships content block types with no decided proxy handling: {sorted(new)}"
    )
