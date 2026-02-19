from __future__ import annotations

from adapter_critic.config import AppConfig
from adapter_critic.upstream import UpstreamResult
from tests.helpers import build_client, usage


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
