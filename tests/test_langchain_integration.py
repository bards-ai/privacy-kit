from __future__ import annotations

import builtins
import sys
import types

import pytest


def test_make_langfuse_callback_configures_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    created_clients = []
    created_handlers = []

    class FakeLangfuse:
        def __init__(self, **kwargs):
            created_clients.append(kwargs)

    class FakeCallbackHandler:
        def __init__(self):
            created_handlers.append(self)

    fake_langfuse = types.ModuleType("langfuse")
    fake_langfuse.Langfuse = FakeLangfuse
    fake_langfuse_langchain = types.ModuleType("langfuse.langchain")
    fake_langfuse_langchain.CallbackHandler = FakeCallbackHandler

    monkeypatch.setitem(sys.modules, "langfuse", fake_langfuse)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", fake_langfuse_langchain)

    import privacy_kit.integrations.langchain as integration

    mask = object()
    monkeypatch.setattr(integration, "make_mask", lambda **kwargs: (kwargs, mask))

    handler = integration.make_langfuse_callback(
        langfuse_kwargs={"public_key": "pk-test"},
        mask_kwargs={"exclude_paths": ["metadata.trace_id"]},
    )

    assert handler is created_handlers[0]
    assert created_clients == [
        {
            "public_key": "pk-test",
            "mask": ({"exclude_paths": ["metadata.trace_id"]}, mask),
        }
    ]


def test_make_langfuse_callback_has_clear_missing_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import privacy_kit.integrations.langchain as integration

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "langfuse" or name.startswith("langfuse."):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(RuntimeError, match=r"privacy-kit\[langchain\]"):
        integration.make_langfuse_callback()
