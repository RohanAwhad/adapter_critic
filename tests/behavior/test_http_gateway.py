from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from loguru import logger

from adapter_critic.contracts import ChatMessage
from adapter_critic.http_gateway import OpenAICompatibleHttpGateway, UpstreamResponseFormatError


@pytest.mark.anyio
async def test_openai_compatible_http_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = FastAPI()

    @upstream.post("/v1/chat/completions")
    async def chat(payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["model"] == "api-model"
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "hello"
        return {
            "id": "chatcmpl-upstream",
            "object": "chat.completion",
            "created": 0,
            "model": "api-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    transport = httpx.ASGITransport(app=upstream)
    original_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

    gateway = OpenAICompatibleHttpGateway(api_key="dummy", timeout_seconds=5.0)
    result = await gateway.complete(
        model="api-model",
        base_url="http://testserver/v1",
        messages=[ChatMessage(role="user", content="hello")],
    )

    assert result.content == "ok"
    assert result.usage.prompt_tokens == 2
    assert result.usage.completion_tokens == 3
    assert result.usage.total_tokens == 5


@pytest.mark.anyio
async def test_openai_compatible_http_gateway_uses_stage_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = FastAPI()

    @upstream.post("/v1/chat/completions")
    async def chat(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["model"] == "api-model"
        assert request.headers.get("authorization") == "Bearer groq-secret"
        return {
            "id": "chatcmpl-upstream",
            "object": "chat.completion",
            "created": 0,
            "model": "api-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    transport = httpx.ASGITransport(app=upstream)
    original_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")

    gateway = OpenAICompatibleHttpGateway(api_key=None, default_api_key_env="OPENAI_API_KEY", timeout_seconds=5.0)
    result = await gateway.complete(
        model="api-model",
        base_url="http://testserver/v1",
        messages=[ChatMessage(role="user", content="hello")],
        api_key_env="GROQ_API_KEY",
    )

    assert result.content == "ok"


@pytest.mark.anyio
async def test_openai_compatible_http_gateway_uses_default_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = FastAPI()

    @upstream.post("/v1/chat/completions")
    async def chat(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["model"] == "api-model"
        assert request.headers.get("authorization") == "Bearer openai-secret"
        return {
            "id": "chatcmpl-upstream",
            "object": "chat.completion",
            "created": 0,
            "model": "api-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    transport = httpx.ASGITransport(app=upstream)
    original_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    gateway = OpenAICompatibleHttpGateway(api_key=None, default_api_key_env="OPENAI_API_KEY", timeout_seconds=5.0)
    result = await gateway.complete(
        model="api-model",
        base_url="http://testserver/v1",
        messages=[ChatMessage(role="user", content="hello")],
    )

    assert result.content == "ok"


@pytest.mark.anyio
async def test_openai_compatible_http_gateway_raises_for_missing_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = FastAPI()

    @upstream.post("/v1/chat/completions")
    async def chat(payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["model"] == "api-model"
        return {"error": {"message": "temporary upstream failure"}}

    transport = httpx.ASGITransport(app=upstream)
    original_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

    gateway = OpenAICompatibleHttpGateway(api_key="dummy", timeout_seconds=5.0)
    with pytest.raises(UpstreamResponseFormatError) as exc_info:
        await gateway.complete(
            model="api-model",
            base_url="http://testserver/v1",
            messages=[ChatMessage(role="user", content="hello")],
        )

    exc = exc_info.value
    assert exc.reason == "response missing non-empty choices"
    assert exc.model == "api-model"
    assert exc.base_url == "http://testserver/v1"
    assert exc.message_count == 1
    assert exc.status_code == 200
    assert '"error"' in exc.payload_preview


@pytest.mark.anyio
async def test_openai_compatible_http_gateway_forwards_request_options_and_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = FastAPI()

    @upstream.post("/v1/chat/completions")
    async def chat(payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["model"] == "api-model"
        assert payload["tool_choice"] == "auto"
        assert payload["tools"][0]["function"]["name"] == "cancel_reservation"
        return {
            "id": "chatcmpl-upstream",
            "object": "chat.completion",
            "created": 0,
            "model": "api-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_cancel",
                                "type": "function",
                                "function": {
                                    "name": "cancel_reservation",
                                    "arguments": '{"reservation_id":"EHGLP3"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    transport = httpx.ASGITransport(app=upstream)
    original_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

    gateway = OpenAICompatibleHttpGateway(api_key="dummy", timeout_seconds=5.0)
    result = await gateway.complete(
        model="api-model",
        base_url="http://testserver/v1",
        messages=[ChatMessage(role="user", content="hello")],
        request_options={
            "tool_choice": "auto",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "cancel_reservation",
                        "parameters": {
                            "type": "object",
                            "properties": {"reservation_id": {"type": "string"}},
                        },
                    },
                }
            ],
        },
    )

    assert result.content == ""
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls is not None
    assert result.tool_calls[0]["function"]["name"] == "cancel_reservation"


@pytest.mark.anyio
async def test_openai_compatible_http_gateway_rejects_empty_content_without_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = FastAPI()

    @upstream.post("/v1/chat/completions")
    async def chat(payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["model"] == "api-model"
        return {
            "id": "chatcmpl-upstream",
            "object": "chat.completion",
            "created": 0,
            "model": "api-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    transport = httpx.ASGITransport(app=upstream)
    original_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

    gateway = OpenAICompatibleHttpGateway(api_key="dummy", timeout_seconds=5.0)
    with pytest.raises(UpstreamResponseFormatError) as exc_info:
        await gateway.complete(
            model="api-model",
            base_url="http://testserver/v1",
            messages=[ChatMessage(role="user", content="hello")],
        )

    assert exc_info.value.reason == "assistant message has empty content and no tool calls"


@pytest.mark.anyio
async def test_openai_compatible_http_gateway_rejects_empty_content_with_empty_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = FastAPI()

    @upstream.post("/v1/chat/completions")
    async def chat(payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["model"] == "api-model"
        return {
            "id": "chatcmpl-upstream",
            "object": "chat.completion",
            "created": 0,
            "model": "api-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None, "tool_calls": []},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    transport = httpx.ASGITransport(app=upstream)
    original_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

    gateway = OpenAICompatibleHttpGateway(api_key="dummy", timeout_seconds=5.0)
    with pytest.raises(UpstreamResponseFormatError) as exc_info:
        await gateway.complete(
            model="api-model",
            base_url="http://testserver/v1",
            messages=[ChatMessage(role="user", content="hello")],
        )

    assert exc_info.value.reason == "assistant message has empty content and no tool calls"


@pytest.mark.anyio
async def test_openai_compatible_http_gateway_logs_malformed_outbound_tool_call_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records: list[str] = []

    def capture(message: Any) -> None:
        records.append(message.record["message"])

    sink_id = logger.add(capture, level="WARNING")
    upstream = FastAPI()

    @upstream.post("/v1/chat/completions")
    async def chat(payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["model"] == "api-model"
        return {
            "id": "chatcmpl-upstream",
            "object": "chat.completion",
            "created": 0,
            "model": "api-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    transport = httpx.ASGITransport(app=upstream)
    original_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

    gateway = OpenAICompatibleHttpGateway(api_key="dummy", timeout_seconds=5.0)
    malformed_history_message = ChatMessage.model_validate(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "foo", "arguments": {"x": 1}},
                }
            ],
        }
    )
    try:
        result = await gateway.complete(
            model="api-model",
            base_url="http://testserver/v1",
            messages=[ChatMessage(role="user", content="hello"), malformed_history_message],
        )
    finally:
        logger.remove(sink_id)

    assert result.content == "ok"
    assert any(
        "detected malformed assistant tool calls before upstream request" in record
        and "tool_call.function.arguments must be a string" in record
        for record in records
    )
