from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Mode = Literal["direct", "adapter", "critic"]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class AdapterCriticOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Mode | None = None
    api_model: str | None = None
    api_base_url: str | None = None
    adapter_model: str | None = None
    adapter_base_url: str | None = None
    critic_model: str | None = None
    critic_base_url: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    extra_body: dict[str, Any] = Field(default_factory=dict)


class ParsedRequest(BaseModel):
    request: ChatCompletionRequest
    overrides: AdapterCriticOverrides


def parse_request_payload(payload: dict[str, Any]) -> ParsedRequest:
    request = ChatCompletionRequest.model_validate(payload)
    override_payload = payload.get("x_adapter_critic")
    if override_payload is None:
        override_payload = request.extra_body.get("x_adapter_critic", {})
    overrides = AdapterCriticOverrides.model_validate(override_payload or {})
    return ParsedRequest(request=request, overrides=overrides)
