from __future__ import annotations

from adapter_critic.config import AppConfig
from adapter_critic.upstream import UpstreamResult
from tests.helpers import build_client, usage


def _config_with_served_advisor(base_config: AppConfig) -> AppConfig:
    payload = base_config.model_dump(exclude_none=True)
    payload["served_models"]["served-advisor"] = {
        "mode": "advisor",
        "api": {"model": "api-model", "base_url": "https://api.example"},
        "advisor": {"model": "advisor-model", "base_url": "https://advisor.example"},
    }
    return AppConfig.model_validate(payload)


def test_startup_served_model_direct_without_override(base_config: AppConfig) -> None:
    client, gateway = build_client(base_config, [UpstreamResult(content="ok", usage=usage(1, 1, 2))])
    response = client.post(
        "/v1/chat/completions",
        json={"model": "served-direct", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert [call["model"] for call in gateway.calls] == ["api-model"]


def test_startup_served_model_adapter_without_override(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(content="a", usage=usage(1, 1, 2)),
            UpstreamResult(content='{"decision":"lgtm"}', usage=usage(1, 1, 2)),
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "served-adapter", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert [call["model"] for call in gateway.calls] == ["api-model", "adapter-model"]


def test_startup_served_model_critic_without_override(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(content="a", usage=usage(1, 1, 2)),
            UpstreamResult(content="b", usage=usage(1, 1, 2)),
            UpstreamResult(content="c", usage=usage(1, 1, 2)),
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "served-critic", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert [call["model"] for call in gateway.calls] == ["api-model", "critic-model", "api-model"]


def test_startup_served_model_advisor_without_override(base_config: AppConfig) -> None:
    client, gateway = build_client(
        _config_with_served_advisor(base_config),
        [
            UpstreamResult(content="check policy and constraints", usage=usage(1, 1, 2)),
            UpstreamResult(content="final answer", usage=usage(1, 1, 2)),
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "served-advisor", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert [call["model"] for call in gateway.calls] == ["advisor-model", "api-model"]


def test_request_override_precedence(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(content="draft", usage=usage(1, 1, 2)),
            UpstreamResult(content='{"decision":"lgtm"}', usage=usage(1, 1, 2)),
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "served-direct",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {
                "x_adapter_critic": {
                    "mode": "adapter",
                    "adapter_model": "adapter-override",
                    "adapter_base_url": "https://override.example",
                }
            },
        },
    )
    assert response.status_code == 200
    assert [call["model"] for call in gateway.calls] == ["api-model", "adapter-override"]


def test_mode_override_without_secondary_target_uses_api_target(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(content="draft", usage=usage(1, 1, 2)),
            UpstreamResult(content='{"decision":"lgtm"}', usage=usage(1, 1, 2)),
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "served-direct",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"x_adapter_critic": {"mode": "adapter"}},
        },
    )
    assert response.status_code == 200
    assert response.json()["adapter_critic"]["mode"] == "adapter"
    assert [call["model"] for call in gateway.calls] == ["api-model", "api-model"]


def test_critic_mode_override_without_secondary_target_uses_api_target(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(content="draft", usage=usage(1, 1, 2)),
            UpstreamResult(content="feedback", usage=usage(1, 1, 2)),
            UpstreamResult(content="final", usage=usage(1, 1, 2)),
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "served-direct",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"x_adapter_critic": {"mode": "critic"}},
        },
    )
    assert response.status_code == 200
    assert response.json()["adapter_critic"]["mode"] == "critic"
    assert [call["model"] for call in gateway.calls] == ["api-model", "api-model", "api-model"]


def test_advisor_mode_override_without_secondary_target_uses_api_target(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(content="advisor guidance", usage=usage(1, 1, 2)),
            UpstreamResult(content="final", usage=usage(1, 1, 2)),
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "served-direct",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"x_adapter_critic": {"mode": "advisor"}},
        },
    )
    assert response.status_code == 200
    assert response.json()["adapter_critic"]["mode"] == "advisor"
    assert [call["model"] for call in gateway.calls] == ["api-model", "api-model"]


def test_advisor_stage_uses_full_messages_and_no_request_options(base_config: AppConfig) -> None:
    client, gateway = build_client(
        _config_with_served_advisor(base_config),
        [
            UpstreamResult(content="check cancellation policy", usage=usage(1, 1, 2)),
            UpstreamResult(content="done", usage=usage(1, 1, 2)),
        ],
    )
    original_messages = [
        {"role": "system", "content": "You are a reservation assistant"},
        {"role": "user", "content": "cancel reservation EHGLP3"},
    ]
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "served-advisor",
            "messages": original_messages,
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
    assert gateway.calls[0]["request_options"] is None

    advisor_messages = gateway.calls[0]["messages"]
    assert advisor_messages[1].role == "system"
    assert advisor_messages[1].content == "You are a reservation assistant"
    assert advisor_messages[2].role == "user"
    assert advisor_messages[2].content == "cancel reservation EHGLP3"

    api_messages = gateway.calls[1]["messages"]
    assert api_messages[0].role == "system"
    assert api_messages[0].content == "You are a reservation assistant"
    assert api_messages[1].role == "user"

    assert gateway.calls[1]["request_options"] is not None
    assert gateway.calls[1]["request_options"]["tool_choice"] == "auto"

    final_user_content = api_messages[1].content
    assert final_user_content is not None
    assert "[ADVISOR_GUIDANCE]" in final_user_content
    assert "check cancellation policy" in final_user_content
