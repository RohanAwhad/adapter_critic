from __future__ import annotations

from adapter_critic.config import AppConfig
from adapter_critic.upstream import UpstreamResult
from tests.helpers import build_client, usage


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
            UpstreamResult(content="lgtm", usage=usage(1, 1, 2)),
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


def test_request_override_precedence(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(content="draft", usage=usage(1, 1, 2)),
            UpstreamResult(content="lgtm", usage=usage(1, 1, 2)),
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
