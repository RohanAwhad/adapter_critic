from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

from fastapi.testclient import TestClient

from adapter_critic.app import create_app
from adapter_critic.config import AppConfig
from adapter_critic.contracts import ChatMessage
from adapter_critic.runtime import build_runtime_state
from adapter_critic.upstream import TokenUsage, UpstreamResult


class GatewayCall(TypedDict):
    model: str
    base_url: str
    messages: list[ChatMessage]
    api_key_env: str | None
    request_options: dict[str, Any] | None


class FakeGateway:
    def __init__(self, responses: Sequence[UpstreamResult]) -> None:
        self._responses = list(responses)
        self.calls: list[GatewayCall] = []

    async def complete(
        self,
        *,
        model: str,
        base_url: str,
        messages: list[ChatMessage],
        api_key_env: str | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> UpstreamResult:
        self.calls.append(
            {
                "model": model,
                "base_url": base_url,
                "messages": messages,
                "api_key_env": api_key_env,
                "request_options": request_options,
            }
        )
        return self._responses.pop(0)


def build_client(
    config: AppConfig,
    responses: Sequence[UpstreamResult],
    *,
    response_id: str = "chatcmpl-test",
    created: int = 1700000000,
) -> tuple[TestClient, FakeGateway]:
    gateway = FakeGateway(responses)
    state = build_runtime_state(
        config=config,
        gateway=gateway,
        id_provider=lambda: response_id,
        time_provider=lambda: created,
    )
    app = create_app(config=config, gateway=gateway, state=state)
    return TestClient(app), gateway


def usage(prompt: int, completion: int, total: int) -> TokenUsage:
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
