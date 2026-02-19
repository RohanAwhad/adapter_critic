from __future__ import annotations

from adapter_critic.config import AppConfig
from adapter_critic.upstream import UpstreamResult
from tests.helpers import build_client, usage


def test_direct_mode_path(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [UpstreamResult(content="direct-answer", usage=usage(2, 3, 5))],
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "served-direct", "messages": [{"role": "user", "content": "hello"}]},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["choices"][0]["message"]["content"] == "direct-answer"
    assert len(gateway.calls) == 1
    assert gateway.calls[0]["model"] == "api-model"
    assert payload["adapter_critic"]["tokens"]["total"]["total_tokens"] == 5
