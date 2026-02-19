from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from fastapi.testclient import TestClient

from adapter_critic.app import create_app
from adapter_critic.config import AppConfig
from adapter_critic.contracts import ChatMessage
from adapter_critic.upstream import TokenUsage, UpstreamResult


class GatewayCall(TypedDict):
    model: str
    base_url: str
    messages: list[ChatMessage]


class FakeGateway:
    def __init__(self, responses: Sequence[UpstreamResult]) -> None:
        self._responses = list(responses)
        self.calls: list[GatewayCall] = []

    async def complete(self, *, model: str, base_url: str, messages: list[ChatMessage]) -> UpstreamResult:
        self.calls.append({"model": model, "base_url": base_url, "messages": messages})
        return self._responses.pop(0)


def build_client(config: AppConfig, responses: Sequence[UpstreamResult]) -> tuple[TestClient, FakeGateway]:
    gateway = FakeGateway(responses)
    app = create_app(config=config, gateway=gateway)
    return TestClient(app), gateway


def usage(prompt: int, completion: int, total: int) -> TokenUsage:
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
