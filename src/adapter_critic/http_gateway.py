from __future__ import annotations

import json
import os
from typing import Any

import httpx
from loguru import logger

from .contracts import ChatMessage
from .upstream import TokenUsage, UpstreamResult


def _payload_preview(payload: Any, *, max_chars: int | None = 400) -> str:
    try:
        serialized = json.dumps(payload, default=str)
    except TypeError:
        serialized = str(payload)
    if max_chars is None or max_chars <= 0:
        return serialized
    if len(serialized) <= max_chars:
        return serialized
    return f"{serialized[:max_chars]}..."


def _json_char_len(value: Any) -> int:
    return len(json.dumps(value, default=str, separators=(",", ":")))


def _approx_token_count(char_len: int) -> int:
    return max(1, (char_len + 3) // 4)


def _malformed_tool_call_issues(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for message_idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls")
        if tool_calls is None:
            continue
        if not isinstance(tool_calls, list):
            issues.append(
                {
                    "message_index": message_idx,
                    "issue": "assistant tool_calls is not a list",
                    "tool_calls_type": type(tool_calls).__name__,
                }
            )
            continue

        for tool_call_idx, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                issues.append(
                    {
                        "message_index": message_idx,
                        "tool_call_index": tool_call_idx,
                        "issue": "tool_call entry is not an object",
                        "tool_call_type": type(tool_call).__name__,
                    }
                )
                continue

            function = tool_call.get("function")
            if not isinstance(function, dict):
                issues.append(
                    {
                        "message_index": message_idx,
                        "tool_call_index": tool_call_idx,
                        "issue": "tool_call.function is not an object",
                        "function_type": type(function).__name__,
                    }
                )
                continue

            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                issues.append(
                    {
                        "message_index": message_idx,
                        "tool_call_index": tool_call_idx,
                        "issue": "tool_call.function.arguments must be a string",
                        "arguments_type": type(arguments).__name__,
                        "tool_name": function.get("name"),
                        "arguments_preview": _payload_preview(arguments, max_chars=0),
                    }
                )

    return issues


def _is_empty_assistant_edge_case(*, content_value: Any, tool_calls_value: Any) -> bool:
    content_is_empty_shape = content_value is None or (isinstance(content_value, list) and len(content_value) == 0)
    tool_calls_is_empty_shape = tool_calls_value is None or (
        isinstance(tool_calls_value, list) and len(tool_calls_value) == 0
    )
    return content_is_empty_shape and tool_calls_is_empty_shape


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

        messages_char_len = _json_char_len(payload["messages"])
        request_options_char_len = _json_char_len(
            {key: value for key, value in payload.items() if key not in {"model", "messages"}}
        )
        approx_prompt_tokens = _approx_token_count(messages_char_len + request_options_char_len)

        malformed_issues = _malformed_tool_call_issues(payload["messages"])
        if len(malformed_issues) > 0:
            logger.warning(
                "detected malformed assistant tool calls before upstream request "
                "model={} base_url={} issues_count={} issues={}",
                model,
                base_url,
                len(malformed_issues),
                _payload_preview(malformed_issues, max_chars=0),
            )

        logger.debug(
            "upstream request model={} base_url={} message_count={} messages_char_len={} "
            "request_options_char_len={} approx_prompt_tokens={}",
            model,
            base_url,
            len(messages),
            messages_char_len,
            request_options_char_len,
            approx_prompt_tokens,
        )

        max_empty_assistant_attempts = 2
        for attempt in range(1, max_empty_assistant_attempts + 1):
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

            logger.debug(
                "upstream raw response model={} base_url={} status={} attempt={}/{} message={}",
                model,
                base_url,
                response.status_code,
                attempt,
                max_empty_assistant_attempts,
                _payload_preview(data, max_chars=2000),
            )

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
                tool_calls = tool_calls_value if len(tool_calls_value) > 0 else None
            else:
                raise UpstreamResponseFormatError(
                    reason="choices[0].message.tool_calls is not a list of objects",
                    model=model,
                    base_url=base_url,
                    message_count=len(messages),
                    status_code=response.status_code,
                    response_body=data,
                )

            if tool_calls is not None:
                for index, tool_call in enumerate(tool_calls):
                    function = tool_call.get("function")
                    if not isinstance(function, dict):
                        raise UpstreamResponseFormatError(
                            reason="choices[0].message.tool_calls[*].function is not an object",
                            model=model,
                            base_url=base_url,
                            message_count=len(messages),
                            status_code=response.status_code,
                            response_body=data,
                        )
                    arguments = function.get("arguments")
                    if not isinstance(arguments, str):
                        raise UpstreamResponseFormatError(
                            reason="choices[0].message.tool_calls[*].function.arguments is not a string",
                            model=model,
                            base_url=base_url,
                            message_count=len(messages),
                            status_code=response.status_code,
                            response_body=data,
                        )
                    try:
                        json.loads(arguments)
                    except ValueError as exc:
                        raise UpstreamResponseFormatError(
                            reason=(
                                "choices[0].message.tool_calls[*].function.arguments is not valid JSON "
                                f"at index {index}"
                            ),
                            model=model,
                            base_url=base_url,
                            message_count=len(messages),
                            status_code=response.status_code,
                            response_body=data,
                        ) from exc

            raw_usage = data.get("usage", {})
            usage = raw_usage if isinstance(raw_usage, dict) else {}
            content_value = message.get("content")
            if isinstance(content_value, str):
                content = content_value
            elif isinstance(content_value, list):
                content = "".join(
                    part["text"]
                    for part in content_value
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                )
            else:
                content = ""

            if content == "" and tool_calls is None:
                if _is_empty_assistant_edge_case(content_value=content_value, tool_calls_value=tool_calls_value):
                    if attempt < max_empty_assistant_attempts:
                        logger.warning(
                            "empty assistant payload without tool calls; retrying upstream request "
                            "model={} base_url={} attempt={}/{} content_type={} tool_calls_type={}",
                            model,
                            base_url,
                            attempt,
                            max_empty_assistant_attempts,
                            type(content_value).__name__,
                            type(tool_calls_value).__name__,
                        )
                        continue
                    logger.warning(
                        "empty assistant payload without tool calls persisted after retry; accepting empty content "
                        "model={} base_url={} attempts={} content_type={} tool_calls_type={}",
                        model,
                        base_url,
                        max_empty_assistant_attempts,
                        type(content_value).__name__,
                        type(tool_calls_value).__name__,
                    )
                else:
                    raise UpstreamResponseFormatError(
                        reason="assistant message has empty content and no tool calls",
                        model=model,
                        base_url=base_url,
                        message_count=len(messages),
                        status_code=response.status_code,
                        response_body=data,
                    )

            logger.debug(
                "upstream parsed model={} content_len={} content_type={} "
                "tool_calls_count={} tool_calls_raw_type={} finish_reason={} "
                "prompt_tokens={} completion_tokens={} total_tokens={}",
                model,
                len(content),
                type(content_value).__name__,
                len(tool_calls) if tool_calls is not None else "None",
                type(tool_calls_value).__name__,
                first_choice.get("finish_reason"),
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)),
                int(usage.get("total_tokens", 0)),
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

        raise RuntimeError("unreachable: max_empty_assistant_attempts exhausted")
