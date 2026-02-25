from __future__ import annotations

from typing import Any

from .contracts import ChatMessage
from .upstream import UpstreamGateway, UpstreamResult
from .vertex_gateway import is_vertex_anthropic_target


class RoutingGateway:
    def __init__(
        self,
        *,
        openai_gateway: UpstreamGateway,
        vertex_gateway: UpstreamGateway,
    ) -> None:
        self._openai_gateway = openai_gateway
        self._vertex_gateway = vertex_gateway

    async def complete(
        self,
        *,
        model: str,
        base_url: str,
        messages: list[ChatMessage],
        api_key_env: str | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> UpstreamResult:
        gateway = self._openai_gateway
        if is_vertex_anthropic_target(model=model, base_url=base_url):
            gateway = self._vertex_gateway

        return await gateway.complete(
            model=model,
            base_url=base_url,
            messages=messages,
            api_key_env=api_key_env,
            request_options=request_options,
        )
