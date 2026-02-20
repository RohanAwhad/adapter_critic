from __future__ import annotations

import json

from adapter_critic.config import AppConfig
from adapter_critic.upstream import UpstreamResult
from tests.helpers import build_client, usage


def test_adapter_mode_path(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(content="Hello wrld", usage=usage(2, 2, 4)),
            UpstreamResult(
                content=('{"decision":"patch","patches":[{"op":"replace","path":"/content","value":"Hello world"}]}'),
                usage=usage(1, 3, 4),
            ),
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "served-adapter", "messages": [{"role": "user", "content": "hello"}]},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["choices"][0]["message"]["content"] == "Hello world"
    assert [call["model"] for call in gateway.calls] == ["api-model", "adapter-model"]
    adapter_prompt_content = gateway.calls[1]["messages"][1].content
    assert adapter_prompt_content is not None
    assert "Latest API draft" in adapter_prompt_content
    assert "Hello wrld" in adapter_prompt_content


def test_adapter_mode_missing_search_passthrough(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(content="Hello world", usage=usage(2, 2, 4)),
            UpstreamResult(
                content='{"decision":"patch","patches":[{"op":"replace","path":"/unknown","value":"replacement"}]}',
                usage=usage(1, 3, 4),
            ),
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "served-adapter", "messages": [{"role": "user", "content": "hello"}]},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["choices"][0]["message"]["content"] == "Hello world"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["adapter_critic"]["intermediate"]["api_draft"] == "Hello world"
    assert payload["adapter_critic"]["intermediate"]["final"] == "Hello world"
    assert [call["model"] for call in gateway.calls] == ["api-model", "adapter-model"]


def test_adapter_mode_retries_after_invalid_adapter_output(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(
                content="",
                usage=usage(2, 2, 4),
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
            UpstreamResult(content="not valid edits", usage=usage(1, 1, 2)),
            UpstreamResult(content='{"decision":"lgtm"}', usage=usage(1, 1, 2)),
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "served-adapter",
            "messages": [{"role": "user", "content": "cancel reservation EHGLP3"}],
            "x_adapter_critic": {"max_adapter_retries": 1},
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
    payload = response.json()
    assert response.status_code == 200
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "cancel_reservation"
    assert payload["adapter_critic"]["tokens"]["stages"]["adapter"]["total_tokens"] == 4
    assert [call["model"] for call in gateway.calls] == ["api-model", "adapter-model", "adapter-model"]


def test_adapter_mode_accepts_drop_tool_call_when_tool_choice_auto(base_config: AppConfig) -> None:
    adapter_rewrite = (
        '{"decision":"patch","patches":['
        '{"op":"replace","path":"/content","value":"Please confirm the reservation id."},'
        '{"op":"replace","path":"/tool_calls","value":[]}'
        "]}"
    )

    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(
                content="",
                usage=usage(2, 2, 4),
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
            UpstreamResult(content=adapter_rewrite, usage=usage(1, 1, 2)),
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "served-adapter",
            "messages": [{"role": "user", "content": "cancel reservation EHGLP3"}],
            "x_adapter_critic": {"max_adapter_retries": 1},
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
    payload = response.json()
    assert response.status_code == 200
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["choices"][0]["message"]["content"] == "Please confirm the reservation id."
    assert "tool_calls" not in payload["choices"][0]["message"]
    assert [call["model"] for call in gateway.calls] == ["api-model", "adapter-model"]


def test_adapter_mode_falls_back_when_tool_choice_required_and_adapter_drops_call(base_config: AppConfig) -> None:
    adapter_rewrite = (
        '{"decision":"patch","patches":['
        '{"op":"replace","path":"/content","value":"Please confirm the reservation id."},'
        '{"op":"replace","path":"/tool_calls","value":[]}'
        "]}"
    )

    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(
                content="",
                usage=usage(2, 2, 4),
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
            UpstreamResult(content=adapter_rewrite, usage=usage(1, 1, 2)),
            UpstreamResult(content=adapter_rewrite, usage=usage(1, 1, 2)),
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "served-adapter",
            "messages": [{"role": "user", "content": "cancel reservation EHGLP3"}],
            "x_adapter_critic": {"max_adapter_retries": 1},
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
            "tool_choice": "required",
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["content"] == ""
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "cancel_reservation"
    assert [call["model"] for call in gateway.calls] == ["api-model", "adapter-model", "adapter-model"]


def test_adapter_mode_exposes_api_draft_tool_calls_in_intermediate(base_config: AppConfig) -> None:
    client, _gateway = build_client(
        base_config,
        [
            UpstreamResult(
                content="",
                usage=usage(2, 2, 4),
                tool_calls=[
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "hyperdimensional_laser_mango_telemetry_v42",
                            "arguments": '{"city":"Paris","units":"celsius"}',
                        },
                    }
                ],
                finish_reason="tool_calls",
            ),
            UpstreamResult(content='{"decision":"lgtm"}', usage=usage(1, 1, 2)),
        ],
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "served-adapter",
            "messages": [{"role": "user", "content": "what is the weather in paris"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "hyperdimensional_laser_mango_telemetry_v42",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "units": {"type": "string"},
                            },
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )

    payload = response.json()
    assert response.status_code == 200
    tool_calls = json.loads(payload["adapter_critic"]["intermediate"]["api_draft_tool_calls"])
    assert tool_calls[0]["function"]["name"] == "hyperdimensional_laser_mango_telemetry_v42"


def test_adapter_mode_default_no_retry_on_invalid_adapter_output(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(
                content="",
                usage=usage(2, 2, 4),
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
            UpstreamResult(content="not valid edits", usage=usage(1, 1, 2)),
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

    payload = response.json()
    assert response.status_code == 200
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "cancel_reservation"
    assert payload["adapter_critic"]["tokens"]["stages"]["adapter"]["total_tokens"] == 2
    assert [call["model"] for call in gateway.calls] == ["api-model", "adapter-model"]


def test_adapter_mode_falls_back_when_adapter_double_encodes_tool_arguments(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(
                content="",
                usage=usage(2, 2, 4),
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
            UpstreamResult(
                content=(
                    '{"decision":"patch","patches":['
                    '{"op":"replace","path":"/tool_calls/0/function/arguments","value":"\\"{}\\""}'
                    "]}"
                ),
                usage=usage(1, 1, 2),
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

    payload = response.json()
    assert response.status_code == 200
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"reservation_id":"EHGLP3"}'
    assert "adapter_rejection_reason" in payload["adapter_critic"]["intermediate"]
    assert "arguments" in payload["adapter_critic"]["intermediate"]["adapter_rejection_reason"]
    assert [call["model"] for call in gateway.calls] == ["api-model", "adapter-model"]


def test_adapter_mode_falls_back_when_adapter_sets_invalid_tool_arguments_json(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(
                content="",
                usage=usage(2, 2, 4),
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
            UpstreamResult(
                content=(
                    '{"decision":"patch","patches":['
                    '{"op":"replace","path":"/tool_calls/0/function/arguments","value":"{bad"}'
                    "]}"
                ),
                usage=usage(1, 1, 2),
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

    payload = response.json()
    assert response.status_code == 200
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"reservation_id":"EHGLP3"}'
    assert "adapter_rejection_reason" in payload["adapter_critic"]["intermediate"]
    assert "arguments" in payload["adapter_critic"]["intermediate"]["adapter_rejection_reason"]
    assert [call["model"] for call in gateway.calls] == ["api-model", "adapter-model"]
