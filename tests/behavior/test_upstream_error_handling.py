from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from adapter_critic.app import create_app
from adapter_critic.config import AppConfig
from adapter_critic.contracts import ChatMessage
from adapter_critic.http_gateway import UpstreamResponseFormatError
from adapter_critic.upstream import UpstreamResult


class BrokenGateway:
    async def complete(
        self,
        *,
        model: str,
        base_url: str,
        messages: list[ChatMessage],
        api_key_env: str | None = None,
    ) -> UpstreamResult:
        del api_key_env
        raise UpstreamResponseFormatError(
            reason="response missing non-empty choices",
            model=model,
            base_url=base_url,
            message_count=len(messages),
            status_code=200,
            response_body={"error": {"message": "temporary upstream failure"}},
        )


def test_upstream_format_error_returns_502_and_logs_context(
    base_config: AppConfig, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR, logger="adapter_critic.app")
    client = TestClient(create_app(config=base_config, gateway=BrokenGateway()))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "served-direct", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "upstream returned non-OpenAI response shape"}
    messages = [record.getMessage() for record in caplog.records if record.name == "adapter_critic.app"]
    assert any(
        "model=api-model" in message
        and "base_url=https://api.example" in message
        and "message_count=1" in message
        and "status_code=200" in message
        and "reason=response missing non-empty choices" in message
        for message in messages
    )
