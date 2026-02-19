from __future__ import annotations

from typing import Any

import httpx

from .contracts import ChatMessage
from .upstream import TokenUsage, UpstreamResult


class OpenAICompatibleHttpGateway:
    def __init__(self, *, api_key: str | None, timeout_seconds: float = 120.0) -> None:
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    async def complete(self, *, model: str, base_url: str, messages: list[ChatMessage]) -> UpstreamResult:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key is not None and self._api_key != "":
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload: dict[str, Any] = {
            "model": model,
            "messages": [message.model_dump() for message in messages],
        }

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        usage = data.get("usage", {})
        content = data["choices"][0]["message"].get("content") or ""
        return UpstreamResult(
            content=content,
            usage=TokenUsage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
            ),
        )
