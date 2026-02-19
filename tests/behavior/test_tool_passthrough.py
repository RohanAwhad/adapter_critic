from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from adapter_critic.app import create_app
from adapter_critic.config import AppConfig
from adapter_critic.http_gateway import OpenAICompatibleHttpGateway
from adapter_critic.runtime import build_runtime_state


@pytest.mark.anyio
async def test_served_direct_forwards_tools_and_tool_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = FastAPI()

    @upstream.post("/v1/chat/completions")
    async def chat(payload: dict[str, Any]) -> dict[str, Any]:
        has_tools = "tools" in payload and payload.get("tool_choice") == "auto"
        if has_tools:
            message: dict[str, Any] = {
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
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": ""}
            finish_reason = "stop"
        return {
            "id": "chatcmpl-upstream",
            "object": "chat.completion",
            "created": 0,
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    transport = httpx.ASGITransport(app=upstream)
    original_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

    config = AppConfig.model_validate(
        {
            "served_models": {
                "served-direct": {
                    "mode": "direct",
                    "api": {"model": "api-model", "base_url": "http://testserver/v1"},
                }
            }
        }
    )
    gateway = OpenAICompatibleHttpGateway(api_key="dummy", timeout_seconds=5.0)
    state = build_runtime_state(config=config, gateway=gateway)
    app = create_app(config=config, gateway=gateway, state=state)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "served-direct",
            "messages": [{"role": "user", "content": "cancel reservation EHGLP3"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "cancel_reservation",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "reservation_id": {"type": "string"},
                            },
                            "required": ["reservation_id"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["content"] == ""
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "cancel_reservation"
