from __future__ import annotations

from adapter_critic.config import AppConfig
from adapter_critic.upstream import UpstreamResult
from tests.helpers import build_client, usage


def test_adapter_mode_path(base_config: AppConfig) -> None:
    client, gateway = build_client(
        base_config,
        [
            UpstreamResult(content="Hello wrld", usage=usage(2, 2, 4)),
            UpstreamResult(
                content="<<<<<<< SEARCH\nwrld\n=======\nworld\n>>>>>>> REPLACE",
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
                content="<<<<<<< SEARCH\nnot-in-draft\n=======\nreplacement\n>>>>>>> REPLACE",
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
