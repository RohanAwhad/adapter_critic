from __future__ import annotations

from adapter_critic.config import AppConfig
from adapter_critic.upstream import UpstreamResult
from tests.helpers import build_client, usage


def test_intermediate_and_tokens_are_visible(base_config: AppConfig) -> None:
    client, _gateway = build_client(
        base_config,
        [
            UpstreamResult(content="Draft response", usage=usage(3, 2, 5)),
            UpstreamResult(content="Needs detail", usage=usage(2, 1, 3)),
            UpstreamResult(content="Final response", usage=usage(4, 2, 6)),
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "served-critic", "messages": [{"role": "user", "content": "hello"}]},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["adapter_critic"]["mode"] == "critic"
    assert "api_draft" in payload["adapter_critic"]["intermediate"]
    assert "critic" in payload["adapter_critic"]["intermediate"]
    assert payload["adapter_critic"]["tokens"]["total"]["prompt_tokens"] == 9
    assert payload["adapter_critic"]["tokens"]["total"]["completion_tokens"] == 5
    assert payload["adapter_critic"]["tokens"]["total"]["total_tokens"] == 14


def test_telemetry_schema_stage_keys_for_each_mode(base_config: AppConfig) -> None:
    direct_client, _direct_gateway = build_client(
        base_config,
        [UpstreamResult(content="direct", usage=usage(1, 2, 3))],
    )
    direct_payload = direct_client.post(
        "/v1/chat/completions",
        json={"model": "served-direct", "messages": [{"role": "user", "content": "hi"}]},
    ).json()
    assert set(direct_payload["adapter_critic"]["intermediate"].keys()) == {"api"}
    assert set(direct_payload["adapter_critic"]["tokens"]["stages"].keys()) == {"api"}

    adapter_client, _adapter_gateway = build_client(
        base_config,
        [
            UpstreamResult(content="draft", usage=usage(1, 1, 2)),
            UpstreamResult(content="lgtm", usage=usage(2, 1, 3)),
        ],
    )
    adapter_payload = adapter_client.post(
        "/v1/chat/completions",
        json={"model": "served-adapter", "messages": [{"role": "user", "content": "hi"}]},
    ).json()
    assert set(adapter_payload["adapter_critic"]["intermediate"].keys()) == {"api_draft", "adapter", "final"}
    assert set(adapter_payload["adapter_critic"]["tokens"]["stages"].keys()) == {"api", "adapter"}

    critic_client, _critic_gateway = build_client(
        base_config,
        [
            UpstreamResult(content="draft", usage=usage(3, 2, 5)),
            UpstreamResult(content="feedback", usage=usage(2, 1, 3)),
            UpstreamResult(content="final", usage=usage(4, 2, 6)),
        ],
    )
    critic_payload = critic_client.post(
        "/v1/chat/completions",
        json={"model": "served-critic", "messages": [{"role": "user", "content": "hi"}]},
    ).json()
    assert set(critic_payload["adapter_critic"]["intermediate"].keys()) == {"api_draft", "critic", "final"}
    assert set(critic_payload["adapter_critic"]["tokens"]["stages"].keys()) == {"api_draft", "critic", "api_final"}
