from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from .contracts import ChatMessage


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class UpstreamResult(BaseModel):
    content: str
    usage: TokenUsage


class UpstreamGateway(Protocol):
    async def complete(self, *, model: str, base_url: str, messages: list[ChatMessage]) -> UpstreamResult: ...
