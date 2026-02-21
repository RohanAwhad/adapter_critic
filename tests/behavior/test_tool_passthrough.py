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
from adapter_critic.upstream import UpstreamResult
from tests.helpers import build_client, usage


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


def test_served_adapter_can_edit_tool_calls_end_to_end(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(
                content="",
                usage=usage(3, 2, 5),
                tool_calls=[
                    {
                        "id": "call_cancel",
                        "type": "function",
                        "function": {
                            "name": "cancel_reservation",
                            "arguments": '{"reservation_id":"WRONG"}',
                        },
                    }
                ],
                finish_reason="tool_calls",
            ),
            UpstreamResult(
                content=(
                    '{"decision":"patch","patches":['
                    '{"op":"replace","path":"/tool_calls/0/function/arguments","value":"{\\"reservation_id\\":\\"EHGLP3\\"}"}'
                    "]}"
                ),
                usage=usage(2, 1, 3),
            ),
        ],
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "served-adapter",
            "messages": [{"role": "user", "content": "cancel reservation EHGLP3"}],
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
            "tool_choice": "auto",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["content"] == ""
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "cancel_reservation"
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"reservation_id":"EHGLP3"}'

    first_request_options = gateway.calls[0]["request_options"]
    assert first_request_options is not None
    assert first_request_options["tool_choice"] == "auto"
    assert first_request_options["tools"][0]["function"]["name"] == "cancel_reservation"
    adapter_request_options = gateway.calls[1]["request_options"]
    assert adapter_request_options is not None
    assert adapter_request_options["response_format"]["type"] == "json_schema"
    assert "tool_choice" not in adapter_request_options

    adapter_prompt_content = gateway.calls[1]["messages"][1].content
    assert adapter_prompt_content is not None
    assert "<ADAPTER_DRAFT_TOOL_CALLS>" in adapter_prompt_content
    assert "cancel_reservation" in adapter_prompt_content

    adapter_system_content = gateway.calls[1]["messages"][0].content
    assert adapter_system_content is not None
    assert "Authoritative tool contract for this request" in adapter_system_content
    assert '"name": "cancel_reservation"' in adapter_system_content
    assert '"tool_choice": "auto"' in adapter_system_content


def test_served_critic_forwards_tool_options_and_returns_tool_calls(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(
                content="",
                usage=usage(3, 2, 5),
                tool_calls=[
                    {
                        "id": "call_cancel",
                        "type": "function",
                        "function": {
                            "name": "cancel_reservation",
                            "arguments": '{"reservation_id":"WRONG"}',
                        },
                    }
                ],
                finish_reason="tool_calls",
            ),
            UpstreamResult(content="Fix reservation_id argument", usage=usage(2, 1, 3)),
            UpstreamResult(
                content="",
                usage=usage(4, 2, 6),
                tool_calls=[
                    {
                        "id": "call_cancel",
                        "type": "function",
                        "function": {
                            "name": "cancel_reservation",
                            "arguments": '{"reservation_id":"EHGLP3"}',
                        },
                    }
                ],
                finish_reason="tool_calls",
            ),
        ],
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "served-critic",
            "messages": [{"role": "user", "content": "cancel reservation EHGLP3"}],
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
            "tool_choice": "auto",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["content"] == ""
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "cancel_reservation"
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"reservation_id":"EHGLP3"}'

    first_request_options = gateway.calls[0]["request_options"]
    final_request_options = gateway.calls[2]["request_options"]
    assert first_request_options is not None
    assert final_request_options is not None
    assert first_request_options["tool_choice"] == "auto"
    assert final_request_options["tool_choice"] == "auto"
    assert gateway.calls[1]["request_options"] is None

    critic_prompt_content = gateway.calls[1]["messages"][1].content
    assert critic_prompt_content is not None
    assert "<ADAPTER_DRAFT_TOOL_CALLS>" in critic_prompt_content
    assert "cancel_reservation" in critic_prompt_content

    critic_system_content = gateway.calls[1]["messages"][0].content
    assert critic_system_content is not None
    assert "Authoritative tool contract for this request" in critic_system_content
    assert '"name": "cancel_reservation"' in critic_system_content
    assert '"tool_choice": "auto"' in critic_system_content
