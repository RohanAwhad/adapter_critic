from __future__ import annotations

import json
import os
from typing import Any

import httpx

from .contracts import ChatMessage
from .upstream import TokenUsage, UpstreamResult


def _payload_preview(payload: Any, *, max_chars: int = 400) -> str:
    try:
        serialized = json.dumps(payload, default=str)
    except TypeError:
        serialized = str(payload)
    if len(serialized) <= max_chars:
        return serialized
    return f"{serialized[:max_chars]}..."


class UpstreamResponseFormatError(RuntimeError):
    def __init__(
        self,
        *,
        reason: str,
        model: str,
        base_url: str,
        message_count: int,
        status_code: int,
        response_body: Any,
    ) -> None:
        self.reason = reason
        self.model = model
        self.base_url = base_url
        self.message_count = message_count
        self.status_code = status_code
        self.payload_preview = _payload_preview(response_body)
        super().__init__(
            "upstream response format error "
            f"reason={reason} status_code={status_code} model={model} "
            f"base_url={base_url} message_count={message_count} "
            f"payload={self.payload_preview}"
        )


class OpenAICompatibleHttpGateway:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_api_key_env: str | None = "OPENAI_API_KEY",
        timeout_seconds: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._default_api_key_env = default_api_key_env
        self._timeout_seconds = timeout_seconds

    def _resolve_api_key(self, api_key_env: str | None) -> str | None:
        if self._api_key is not None and self._api_key != "":
            return self._api_key

        key_env = api_key_env if api_key_env is not None else self._default_api_key_env
        if key_env is None or key_env == "":
            return None
        return os.environ.get(key_env)

    async def complete(
        self,
        *,
        model: str,
        base_url: str,
        messages: list[ChatMessage],
        api_key_env: str | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> UpstreamResult:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        resolved_api_key = self._resolve_api_key(api_key_env)
        if resolved_api_key is not None and resolved_api_key != "":
            headers["Authorization"] = f"Bearer {resolved_api_key}"

        payload: dict[str, Any] = {
            "model": model,
            "messages": [message.model_dump(exclude_none=True) for message in messages],
        }
        if request_options is not None:
            for key, value in request_options.items():
                if key not in {"model", "messages"}:
                    payload[key] = value

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError as exc:
                raise UpstreamResponseFormatError(
                    reason="response body is not valid JSON",
                    model=model,
                    base_url=base_url,
                    message_count=len(messages),
                    status_code=response.status_code,
                    response_body=response.text,
                ) from exc

        if not isinstance(data, dict):
            raise UpstreamResponseFormatError(
                reason="response body is not a JSON object",
                model=model,
                base_url=base_url,
                message_count=len(messages),
                status_code=response.status_code,
                response_body=data,
            )

        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            raise UpstreamResponseFormatError(
                reason="response missing non-empty choices",
                model=model,
                base_url=base_url,
                message_count=len(messages),
                status_code=response.status_code,
                response_body=data,
            )

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise UpstreamResponseFormatError(
                reason="choices[0] is not an object",
                model=model,
                base_url=base_url,
                message_count=len(messages),
                status_code=response.status_code,
                response_body=data,
            )

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise UpstreamResponseFormatError(
                reason="choices[0].message is not an object",
                model=model,
                base_url=base_url,
                message_count=len(messages),
                status_code=response.status_code,
                response_body=data,
            )

        tool_calls_value = message.get("tool_calls")
        if tool_calls_value is None:
            tool_calls: list[dict[str, Any]] | None = None
        elif isinstance(tool_calls_value, list) and all(isinstance(item, dict) for item in tool_calls_value):
            tool_calls = tool_calls_value
        else:
            raise UpstreamResponseFormatError(
                reason="choices[0].message.tool_calls is not a list of objects",
                model=model,
                base_url=base_url,
                message_count=len(messages),
                status_code=response.status_code,
                response_body=data,
            )

        raw_usage = data.get("usage", {})
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        content_value = message.get("content")
        if isinstance(content_value, str):
            content = content_value
        elif isinstance(content_value, list):
            content = "".join(
                part["text"] for part in content_value if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        else:
            content = ""

        if content == "" and tool_calls is None:
            raise UpstreamResponseFormatError(
                reason="assistant message has empty content and no tool calls",
                model=model,
                base_url=base_url,
                message_count=len(messages),
                status_code=response.status_code,
                response_body=data,
            )

        finish_reason_value = first_choice.get("finish_reason")
        finish_reason = finish_reason_value if isinstance(finish_reason_value, str) else "stop"

        return UpstreamResult(
            content=content,
            usage=TokenUsage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
            ),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )
