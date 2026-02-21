from __future__ import annotations

from typing import Any

import httpx
from fastapi.testclient import TestClient
from loguru import logger

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
        request_options: dict[str, Any] | None = None,
    ) -> UpstreamResult:
        del api_key_env, request_options
        raise UpstreamResponseFormatError(
            reason="response missing non-empty choices",
            model=model,
            base_url=base_url,
            message_count=len(messages),
            status_code=200,
            response_body={"error": {"message": "temporary upstream failure"}},
        )


class HttpFailureGateway:
    async def complete(
        self,
        *,
        model: str,
        base_url: str,
        messages: list[ChatMessage],
        api_key_env: str | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> UpstreamResult:
        del api_key_env, request_options, model, base_url, messages
        request = httpx.Request("POST", "https://api.example/v1/chat/completions")
        response = httpx.Response(500, request=request, json={"error": {"message": "upstream down"}})
        raise httpx.HTTPStatusError("500 Server Error", request=request, response=response)


def test_upstream_format_error_returns_502_and_logs_context(
    base_config: AppConfig,
) -> None:
    records: list[str] = []

    def capture(message: Any) -> None:
        records.append(message.record["message"])

    sink_id = logger.add(capture, level="ERROR")
    client = TestClient(create_app(config=base_config, gateway=BrokenGateway()))

    try:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "served-direct", "messages": [{"role": "user", "content": "hello"}]},
        )
    finally:
        logger.remove(sink_id)

    assert response.status_code == 502
    assert response.json() == {"detail": "upstream returned non-OpenAI response shape"}
    assert any(
        "model=api-model" in message
        and "base_url=https://api.example" in message
        and "message_count=1" in message
        and "status_code=200" in message
        and "reason=response missing non-empty choices" in message
        for message in records
    )


def test_upstream_http_error_returns_502(base_config: AppConfig) -> None:
    records: list[dict[str, Any]] = []

    def capture(message: Any) -> None:
        records.append(message.record)

    sink_id = logger.add(capture, level="ERROR")
    client = TestClient(create_app(config=base_config, gateway=HttpFailureGateway()))
    try:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "served-direct", "messages": [{"role": "user", "content": "hello"}]},
        )
    finally:
        logger.remove(sink_id)

    assert response.status_code == 502
    assert response.json() == {"detail": "upstream request failed"}
    assert any(
        "status_code=500" in record["message"]
        and 'response_body={"error":{"message":"upstream down"}}' in record["message"]
        and record["exception"] is not None
        for record in records
    )
