from __future__ import annotations

from typing import Any

import pytest

from adapter_critic.contracts import ChatMessage
from adapter_critic.routing_gateway import RoutingGateway
from adapter_critic.upstream import TokenUsage, UpstreamResult


class _RecordingGateway:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []

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
        return UpstreamResult(
            content=self.response_text,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


@pytest.mark.anyio
async def test_routing_gateway_uses_vertex_gateway_for_vertex_claude_target() -> None:
    openai_gateway = _RecordingGateway(response_text="openai")
    vertex_gateway = _RecordingGateway(response_text="vertex")
    gateway = RoutingGateway(openai_gateway=openai_gateway, vertex_gateway=vertex_gateway)

    result = await gateway.complete(
        model="claude-sonnet-4-5@20250929",
        base_url="https://us-east5-aiplatform.googleapis.com/v1/projects/p/locations/us-east5",
        messages=[ChatMessage(role="user", content="ping")],
    )

    assert result.content == "vertex"
    assert len(vertex_gateway.calls) == 1
    assert len(openai_gateway.calls) == 0


@pytest.mark.anyio
async def test_routing_gateway_uses_openai_gateway_for_non_vertex_target() -> None:
    openai_gateway = _RecordingGateway(response_text="openai")
    vertex_gateway = _RecordingGateway(response_text="vertex")
    gateway = RoutingGateway(openai_gateway=openai_gateway, vertex_gateway=vertex_gateway)

    result = await gateway.complete(
        model="gpt-4.1-mini",
        base_url="https://api.openai.com/v1",
        messages=[ChatMessage(role="user", content="ping")],
    )

    assert result.content == "openai"
    assert len(openai_gateway.calls) == 1
    assert len(vertex_gateway.calls) == 0
