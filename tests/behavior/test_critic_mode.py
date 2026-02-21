from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from adapter_critic.app import create_app
from adapter_critic.config import AppConfig
from adapter_critic.contracts import ChatMessage
from adapter_critic.http_gateway import UpstreamResponseFormatError
from adapter_critic.runtime import build_runtime_state
from adapter_critic.upstream import UpstreamResult
from tests.helpers import build_client, usage


class FlakyFinalGateway:
    def __init__(self, outcomes: list[UpstreamResult | Exception]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        model: str,
        base_url: str,
        messages: list[ChatMessage],
        api_key_env: str | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> UpstreamResult:
        self.calls.append(
            {
                "model": model,
                "base_url": base_url,
                "messages": messages,
                "api_key_env": api_key_env,
                "request_options": request_options,
            }
        )
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _build_flaky_client(
    config: AppConfig,
    outcomes: list[UpstreamResult | Exception],
) -> tuple[TestClient, FlakyFinalGateway]:
    gateway = FlakyFinalGateway(outcomes)
    state = build_runtime_state(
        config=config,
        gateway=gateway,
        id_provider=lambda: "chatcmpl-test",
        time_provider=lambda: 1700000000,
    )
    return TestClient(create_app(config=config, gateway=gateway, state=state)), gateway


def _empty_assistant_error() -> UpstreamResponseFormatError:
    return UpstreamResponseFormatError(
        reason="assistant message has empty content and no tool calls",
        model="api-model",
        base_url="https://api.example",
        message_count=4,
        status_code=200,
        response_body={
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None, "tool_calls": []},
                    "finish_reason": "stop",
                }
            ]
        },
    )


def _invalid_tool_call_arguments_error() -> UpstreamResponseFormatError:
    return UpstreamResponseFormatError(
        reason="choices[0].message.tool_calls[*].function.arguments is not valid JSON at index 0",
        model="api-model",
        base_url="https://api.example",
        message_count=4,
        status_code=200,
        response_body={
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_transfer",
                                "type": "function",
                                "function": {
                                    "name": "transfer_to_human_agents",
                                    "arguments": "{...}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    )


def test_critic_mode_path(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(content="Draft response", usage=usage(2, 2, 4)),
            UpstreamResult(content="Needs more detail", usage=usage(1, 1, 2)),
            UpstreamResult(content="Final response", usage=usage(2, 3, 5)),
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "served-critic",
            "messages": [
                {"role": "system", "content": "be precise"},
                {"role": "user", "content": "question"},
            ],
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["choices"][0]["message"]["content"] == "Final response"
    assert [call["model"] for call in gateway.calls] == ["api-model", "critic-model", "api-model"]
    critic_prompt_content = gateway.calls[1]["messages"][1].content
    assert critic_prompt_content is not None
    assert "be precise" in critic_prompt_content
    assert "Draft response" in critic_prompt_content
    final_pass_system = gateway.calls[2]["messages"][-1].content
    assert final_pass_system is not None
    assert "Needs more detail" in final_pass_system


def test_critic_mode_without_system_message_uses_empty_system_fallback(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(content="Draft response", usage=usage(2, 2, 4)),
            UpstreamResult(content="Needs structure", usage=usage(1, 1, 2)),
            UpstreamResult(content="Final response", usage=usage(2, 3, 5)),
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "served-critic",
            "messages": [{"role": "user", "content": "question"}],
        },
    )
    assert response.status_code == 200
    critic_prompt_content = gateway.calls[1]["messages"][1].content
    assert critic_prompt_content is not None
    assert "System instructions:\n\n\nConversation history" in critic_prompt_content


def test_critic_intermediate_includes_draft_tool_calls(base_config: AppConfig) -> None:
    draft_tool_calls = [
        {
            "id": "call_cancel",
            "type": "function",
            "function": {
                "name": "cancel_reservation",
                "arguments": '{"reservation_id":"EHGLP3"}',
            },
        }
    ]
    client, _gateway = build_client(
        base_config,
        [
            UpstreamResult(
                content="",
                usage=usage(3, 2, 5),
                tool_calls=draft_tool_calls,
                finish_reason="tool_calls",
            ),
            UpstreamResult(content="LGTM", usage=usage(2, 1, 3)),
            UpstreamResult(
                content="",
                usage=usage(4, 2, 6),
                tool_calls=draft_tool_calls,
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
    intermediate = payload["adapter_critic"]["intermediate"]
    assert "api_draft_tool_calls" in intermediate
    parsed = json.loads(intermediate["api_draft_tool_calls"])
    assert parsed[0]["function"]["name"] == "cancel_reservation"


def test_critic_mode_retries_final_pass_once(base_config: AppConfig) -> None:
    client, gateway = _build_flaky_client(
        base_config,
        outcomes=[
            UpstreamResult(content="Draft response", usage=usage(2, 2, 4)),
            UpstreamResult(content="Needs one more attempt", usage=usage(1, 1, 2)),
            _empty_assistant_error(),
            UpstreamResult(content="Final response", usage=usage(2, 3, 5)),
        ],
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "served-critic", "messages": [{"role": "user", "content": "question"}]},
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["choices"][0]["message"]["content"] == "Final response"
    assert [call["model"] for call in gateway.calls] == ["api-model", "critic-model", "api-model", "api-model"]


def test_critic_mode_falls_back_to_api_draft_after_final_pass_retries(base_config: AppConfig) -> None:
    draft_tool_calls = [
        {
            "id": "call_transfer",
            "type": "function",
            "function": {
                "name": "transfer_to_human_agents",
                "arguments": '{"summary":"needs exception"}',
            },
        }
    ]
    client, gateway = _build_flaky_client(
        base_config,
        outcomes=[
            UpstreamResult(
                content="",
                usage=usage(2, 2, 4),
                tool_calls=draft_tool_calls,
                finish_reason="tool_calls",
            ),
            UpstreamResult(content="Use transfer tool call", usage=usage(1, 1, 2)),
            _empty_assistant_error(),
            _empty_assistant_error(),
        ],
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "served-critic", "messages": [{"role": "user", "content": "question"}]},
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["content"] == ""
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "transfer_to_human_agents"
    assert payload["adapter_critic"]["intermediate"]["final_fallback_reason"].startswith(
        "api_final failed after 2 attempts"
    )
    assert [call["model"] for call in gateway.calls] == ["api-model", "critic-model", "api-model", "api-model"]


def test_critic_mode_falls_back_when_final_pass_tool_call_arguments_are_malformed(base_config: AppConfig) -> None:
    draft_tool_calls = [
        {
            "id": "call_transfer",
            "type": "function",
            "function": {
                "name": "transfer_to_human_agents",
                "arguments": '{"summary":"needs exception"}',
            },
        }
    ]
    client, gateway = _build_flaky_client(
        base_config,
        outcomes=[
            UpstreamResult(
                content="",
                usage=usage(2, 2, 4),
                tool_calls=draft_tool_calls,
                finish_reason="tool_calls",
            ),
            UpstreamResult(content="Use transfer tool call", usage=usage(1, 1, 2)),
            _invalid_tool_call_arguments_error(),
            _invalid_tool_call_arguments_error(),
        ],
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "served-critic", "messages": [{"role": "user", "content": "question"}]},
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["content"] == ""
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "transfer_to_human_agents"
    assert "function.arguments is not valid JSON" in payload["adapter_critic"]["intermediate"]["final_fallback_reason"]
    assert [call["model"] for call in gateway.calls] == ["api-model", "critic-model", "api-model", "api-model"]
