from __future__ import annotations

from collections.abc import Callable
from typing import Any

from privacy_kit.integrations.langfuse import make_mask


def make_langfuse_callback(
    *,
    langfuse_kwargs: dict[str, Any] | None = None,
    mask_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Return a Langfuse CallbackHandler for LangChain/LangGraph tracing."""

    mask = make_mask(**(mask_kwargs or {}))

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LangChain/LangGraph tracing requires optional dependencies. "
            "Install them with: pip install 'privacy-kit[langchain]'"
        ) from exc

    class RedactingCallbackHandler(CallbackHandler):
        def on_chain_start(
            self,
            serialized: Any,
            inputs: Any,
            *,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            return super().on_chain_start(
                serialized,
                mask(inputs),
                metadata=mask(metadata) if metadata is not None else None,
                **_mask_inputs_kwarg(kwargs, mask),
            )

        def on_llm_start(
            self,
            serialized: Any,
            prompts: list[str],
            *,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            return super().on_llm_start(
                serialized,
                mask(prompts),
                metadata=mask(metadata) if metadata is not None else None,
                **_mask_inputs_kwarg(kwargs, mask),
            )

        def on_chat_model_start(
            self,
            serialized: Any,
            messages: Any,
            *,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            return super().on_chat_model_start(
                serialized,
                messages,
                metadata=mask(metadata) if metadata is not None else None,
                **_mask_inputs_kwarg(kwargs, mask),
            )

        def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> Any:
            return super().on_chain_end(mask(outputs), **_mask_inputs_kwarg(kwargs, mask))

    Langfuse(mask=mask, **(langfuse_kwargs or {}))
    return RedactingCallbackHandler()


def _mask_inputs_kwarg(kwargs: dict[str, Any], mask: Callable[[Any], Any]) -> dict[str, Any]:
    if "inputs" not in kwargs:
        return kwargs
    return {**kwargs, "inputs": mask(kwargs["inputs"])}
