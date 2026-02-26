from __future__ import annotations

from adapter_critic.config import AppConfig
from adapter_critic.upstream import UpstreamResult
from tests.helpers import build_client, usage


def _prompt_config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "served_models": {
                "served-direct": {
                    "mode": "direct",
                    "api": {"model": "api-model", "base_url": "https://api.example"},
                    "adapter_system_prompt": "adapter prompt from config",
                    "critic_system_prompt": "critic prompt from config",
                }
            }
        }
    )


def _advisor_prompt_config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "served_models": {
                "served-advisor": {
                    "mode": "advisor",
                    "api": {"model": "api-model", "base_url": "https://api.example"},
                    "advisor": {"model": "advisor-model", "base_url": "https://advisor.example"},
                    "advisor_system_prompt": "advisor prompt from config",
                }
            }
        }
    )


def test_adapter_prompt_is_configurable_per_served_model() -> None:
    client, gateway = build_client(
        _prompt_config(),
        [
            UpstreamResult(content="draft", usage=usage(1, 1, 2)),
            UpstreamResult(content='{"decision":"lgtm"}', usage=usage(1, 1, 2)),
        ],
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "served-direct",
            "messages": [{"role": "user", "content": "hello"}],
            "extra_body": {"x_adapter_critic": {"mode": "adapter"}},
        },
    )

    assert response.status_code == 200
    assert gateway.calls[1]["messages"][0].content == "adapter prompt from config"


def test_critic_prompt_is_configurable_per_served_model() -> None:
    client, gateway = build_client(
        _prompt_config(),
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
            "messages": [{"role": "user", "content": "hello"}],
            "extra_body": {"x_adapter_critic": {"mode": "critic"}},
        },
    )

    assert response.status_code == 200
    assert gateway.calls[1]["messages"][0].content == "critic prompt from config"


def test_advisor_prompt_is_configurable_per_served_model() -> None:
    client, gateway = build_client(
        _advisor_prompt_config(),
        [
            UpstreamResult(content="check account status first", usage=usage(1, 1, 2)),
            UpstreamResult(content="final", usage=usage(1, 1, 2)),
        ],
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "served-advisor",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert gateway.calls[0]["messages"][0].content == "advisor prompt from config"
