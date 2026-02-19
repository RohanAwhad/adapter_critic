from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Mode = Literal["direct", "adapter", "critic"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = ""


class AdapterCriticOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Mode | None = None
    api_model: str | None = None
    api_base_url: str | None = None
    adapter_model: str | None = None
    adapter_base_url: str | None = None
    critic_model: str | None = None
    critic_base_url: str | None = None
    max_adapter_retries: int | None = Field(default=None, ge=0)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    extra_body: dict[str, Any] = Field(default_factory=dict)


class ParsedRequest(BaseModel):
    request: ChatCompletionRequest
    overrides: AdapterCriticOverrides
    request_options: dict[str, Any]


def parse_request_payload(payload: dict[str, Any]) -> ParsedRequest:
    request = ChatCompletionRequest.model_validate(payload)
    override_payload = payload.get("x_adapter_critic")
    if override_payload is None:
        override_payload = request.extra_body.get("x_adapter_critic", {})
    overrides = AdapterCriticOverrides.model_validate(override_payload or {})
    request_options = {key: value for key, value in (request.model_extra or {}).items() if key != "x_adapter_critic"}
    return ParsedRequest(request=request, overrides=overrides, request_options=request_options)
