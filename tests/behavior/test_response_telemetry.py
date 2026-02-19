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
