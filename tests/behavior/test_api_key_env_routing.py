from __future__ import annotations

from adapter_critic.config import AppConfig
from adapter_critic.upstream import UpstreamResult
from tests.helpers import build_client, usage


def _api_key_env_config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "served_models": {
                "served-direct": {
                    "mode": "direct",
                    "api": {
                        "model": "api-model",
                        "base_url": "https://api.example",
                        "api_key_var": "OPENAI_API_KEY",
                    },
                },
                "served-adapter": {
                    "mode": "adapter",
                    "api": {
                        "model": "api-model",
                        "base_url": "https://api.example",
                        "api_key_var": "OPENAI_API_KEY",
                    },
                    "adapter": {
                        "model": "adapter-model",
                        "base_url": "https://adapter.example",
                        "api_key_var": "GROQ_API_KEY",
                    },
                },
                "served-critic": {
                    "mode": "critic",
                    "api": {
                        "model": "api-model",
                        "base_url": "https://api.example",
                        "api_key_var": "OPENAI_API_KEY",
                    },
                    "critic": {
                        "model": "critic-model",
                        "base_url": "https://critic.example",
                        "api_key_var": "ANTHROPIC_API_KEY",
                    },
                },
            }
        }
    )


def test_adapter_mode_routes_stage_api_key_env() -> None:
    client, gateway = build_client(
        _api_key_env_config(),
        [
            UpstreamResult(content="draft", usage=usage(1, 1, 2)),
            UpstreamResult(content='{"decision":"lgtm"}', usage=usage(1, 1, 2)),
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "served-adapter", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200
    assert gateway.calls[0]["api_key_env"] == "OPENAI_API_KEY"
    assert gateway.calls[1]["api_key_env"] == "GROQ_API_KEY"


def test_critic_mode_routes_stage_api_key_env() -> None:
    client, gateway = build_client(
        _api_key_env_config(),
        [
            UpstreamResult(content="draft", usage=usage(1, 1, 2)),
            UpstreamResult(content="feedback", usage=usage(1, 1, 2)),
            UpstreamResult(content="final", usage=usage(1, 1, 2)),
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "served-critic", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200
    assert [call["api_key_env"] for call in gateway.calls] == [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ]
