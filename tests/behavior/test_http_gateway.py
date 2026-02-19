from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from adapter_critic.contracts import ChatMessage
from adapter_critic.http_gateway import OpenAICompatibleHttpGateway


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
