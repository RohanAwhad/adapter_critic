from __future__ import annotations

import json
import subprocess
from typing import Any

import httpx
from loguru import logger

from .contracts import ChatMessage
from .http_gateway import UpstreamResponseFormatError
from .upstream import TokenUsage, UpstreamResult


def _payload_preview(payload: Any, *, max_chars: int | None = 400) -> str:
    serialized = json.dumps(payload, default=str)
    if max_chars is None or max_chars <= 0:
        return serialized
    if len(serialized) <= max_chars:
        return serialized
    return f"{serialized[:max_chars]}..."


def is_vertex_anthropic_target(*, model: str, base_url: str) -> bool:
    normalized_base_url = base_url.lower()
    if "aiplatform.googleapis.com" not in normalized_base_url:
        return False
    if "/publishers/anthropic/models/" in normalized_base_url:
        return True
    if "/projects/" not in normalized_base_url or "/locations/" not in normalized_base_url:
        return False
    if "/endpoints/openapi" in normalized_base_url:
        return False
    normalized_model = model.lower()
    return normalized_model.startswith("anthropic/") or "claude" in normalized_model


def _normalize_vertex_model_name(model: str) -> str:
    if model.startswith("anthropic/"):
        return model.split("/", maxsplit=1)[1]
    return model


def _resolve_vertex_endpoint(*, model: str, base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    lowered = trimmed.lower()

    stream_suffix = ":streamrawpredict"
    if lowered.endswith(stream_suffix):
        return f"{trimmed[: len(trimmed) - len(stream_suffix)]}:rawPredict"
    if lowered.endswith(":rawpredict"):
        return trimmed

    if "/publishers/anthropic/models/" in lowered:
        return f"{trimmed}:rawPredict"

    normalized_model = _normalize_vertex_model_name(model)
    return f"{trimmed}/publishers/anthropic/models/{normalized_model}:rawPredict"


def _resolve_gcloud_access_token() -> str:
    process = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        check=True,
        capture_output=True,
        text=True,
    )
    token = process.stdout.strip()
    if token == "":
        raise RuntimeError("gcloud auth print-access-token returned empty access token")
    return token


def _extract_system_prompt(messages: list[ChatMessage]) -> str:
    system_parts: list[str] = []
    for message in messages:
        if message.role == "system" and message.content is not None and message.content != "":
            system_parts.append(message.content)
    return "\n\n".join(system_parts)


def _string_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""


def _message_to_vertex_content(message: ChatMessage) -> dict[str, Any] | None:
    dumped = message.model_dump(exclude_none=True)
    role_value = dumped.get("role")

    if role_value == "system":
        return None

    if role_value == "user":
        return {"role": "user", "content": _string_content(dumped.get("content"))}

    if role_value == "assistant":
        content_text = _string_content(dumped.get("content"))
        content_blocks: list[dict[str, Any]] = []
        if content_text != "":
            content_blocks.append({"type": "text", "text": content_text})

        tool_calls_value = dumped.get("tool_calls")
        if isinstance(tool_calls_value, list):
            for tool_call in tool_calls_value:
                if not isinstance(tool_call, dict):
                    raise ValueError("assistant tool_calls entry must be an object")
                tool_call_id = tool_call.get("id")
                function_value = tool_call.get("function")
                if not isinstance(tool_call_id, str):
                    raise ValueError("assistant tool_call id must be a string")
                if not isinstance(function_value, dict):
                    raise ValueError("assistant tool_call function must be an object")

                function_name = function_value.get("name")
                function_arguments = function_value.get("arguments")
                if not isinstance(function_name, str):
                    raise ValueError("assistant tool_call function.name must be a string")
                if not isinstance(function_arguments, str):
                    raise ValueError("assistant tool_call function.arguments must be a string")

                parsed_arguments = json.loads(function_arguments)
                if not isinstance(parsed_arguments, dict):
                    raise ValueError("assistant tool_call function.arguments must decode to an object")

                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call_id,
                        "name": function_name,
                        "input": parsed_arguments,
                    }
                )

        if len(content_blocks) == 0:
            return {"role": "assistant", "content": ""}
        return {"role": "assistant", "content": content_blocks}

    if role_value == "tool":
        tool_call_id = dumped.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            raise ValueError("tool role message requires tool_call_id")
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": _string_content(dumped.get("content")),
                }
            ],
        }

    raise ValueError(f"unsupported message role for vertex anthropic: {role_value}")


def _map_stop_sequences(stop_value: Any) -> list[str] | None:
    if isinstance(stop_value, str):
        return [stop_value]
    if isinstance(stop_value, list) and all(isinstance(item, str) for item in stop_value):
        return stop_value
    return None


def _map_tools(tools_value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(tools_value, list):
        return None

    mapped_tools: list[dict[str, Any]] = []
    for tool in tools_value:
        if not isinstance(tool, dict):
            continue
        function_value = tool.get("function")
        if not isinstance(function_value, dict):
            continue

        name_value = function_value.get("name")
        if not isinstance(name_value, str):
            continue

        mapped: dict[str, Any] = {"name": name_value}
        description_value = function_value.get("description")
        if isinstance(description_value, str):
            mapped["description"] = description_value

        parameters_value = function_value.get("parameters")
        if isinstance(parameters_value, dict):
            mapped["input_schema"] = parameters_value
        else:
            mapped["input_schema"] = {"type": "object", "properties": {}}

        mapped_tools.append(mapped)

    if len(mapped_tools) == 0:
        return None
    return mapped_tools


def _map_tool_choice(tool_choice_value: Any) -> dict[str, Any] | None:
    if tool_choice_value == "auto":
        return {"type": "auto"}
    if tool_choice_value == "required":
        return {"type": "any"}
    if isinstance(tool_choice_value, dict) and tool_choice_value.get("type") == "function":
        function_value = tool_choice_value.get("function")
        if isinstance(function_value, dict):
            name_value = function_value.get("name")
            if isinstance(name_value, str):
                return {"type": "tool", "name": name_value}
    return None


def _map_request_options(request_options: dict[str, Any] | None) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    if request_options is None:
        return mapped

    max_tokens_value = request_options.get("max_tokens")
    if isinstance(max_tokens_value, int) and max_tokens_value > 0:
        mapped["max_tokens"] = max_tokens_value

    temperature_value = request_options.get("temperature")
    if isinstance(temperature_value, int | float):
        mapped["temperature"] = temperature_value

    top_p_value = request_options.get("top_p")
    if isinstance(top_p_value, int | float):
        mapped["top_p"] = top_p_value

    top_k_value = request_options.get("top_k")
    if isinstance(top_k_value, int):
        mapped["top_k"] = top_k_value

    stop_sequences = _map_stop_sequences(request_options.get("stop"))
    if stop_sequences is not None:
        mapped["stop_sequences"] = stop_sequences

    mapped_tools = _map_tools(request_options.get("tools"))
    if mapped_tools is not None:
        mapped["tools"] = mapped_tools

    mapped_tool_choice = _map_tool_choice(request_options.get("tool_choice"))
    if mapped_tool_choice is not None:
        mapped["tool_choice"] = mapped_tool_choice

    return mapped


def _map_finish_reason(stop_reason: Any) -> str:
    if stop_reason == "tool_use":
        return "tool_calls"
    if stop_reason == "max_tokens":
        return "length"
    return "stop"


def _map_usage(usage: Any) -> TokenUsage:
    usage_dict = usage if isinstance(usage, dict) else {}

    prompt_tokens_raw = usage_dict.get("input_tokens", usage_dict.get("prompt_tokens", 0))
    completion_tokens_raw = usage_dict.get("output_tokens", usage_dict.get("completion_tokens", 0))
    prompt_tokens = prompt_tokens_raw if isinstance(prompt_tokens_raw, int) else 0
    completion_tokens = completion_tokens_raw if isinstance(completion_tokens_raw, int) else 0

    total_tokens_raw = usage_dict.get("total_tokens")
    total_tokens = total_tokens_raw if isinstance(total_tokens_raw, int) else prompt_tokens + completion_tokens

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


class VertexAICompatibleHttpGateway:
    def __init__(
        self,
        *,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._timeout_seconds = timeout_seconds

    async def complete(
        self,
        *,
        model: str,
        base_url: str,
        messages: list[ChatMessage],
        api_key_env: str | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> UpstreamResult:
        del api_key_env

        endpoint_url = _resolve_vertex_endpoint(model=model, base_url=base_url)
        access_token = _resolve_gcloud_access_token()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        anthropic_messages: list[dict[str, Any]] = []
        for message in messages:
            mapped_message = _message_to_vertex_content(message)
            if mapped_message is not None:
                anthropic_messages.append(mapped_message)

        mapped_request_options = _map_request_options(request_options)
        payload: dict[str, Any] = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": anthropic_messages,
            "max_tokens": int(mapped_request_options.get("max_tokens", 8192)),
        }

        system_prompt = _extract_system_prompt(messages)
        if system_prompt != "":
            payload["system"] = system_prompt

        for key, value in mapped_request_options.items():
            if key != "max_tokens":
                payload[key] = value

        logger.debug(
            "vertex anthropic request model={} endpoint={} message_count={} payload={}",
            model,
            endpoint_url,
            len(anthropic_messages),
            _payload_preview(payload),
        )

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(endpoint_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        logger.debug(
            "vertex anthropic raw response model={} endpoint={} status={} payload={}",
            model,
            endpoint_url,
            response.status_code,
            _payload_preview(data, max_chars=2000),
        )

        if not isinstance(data, dict):
            raise UpstreamResponseFormatError(
                reason="vertex anthropic response body is not a JSON object",
                model=model,
                base_url=endpoint_url,
                message_count=len(messages),
                status_code=response.status_code,
                response_body=data,
            )

        content_value = data.get("content")
        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        if isinstance(content_value, str):
            content_parts.append(content_value)
        elif isinstance(content_value, list):
            for block in content_value:
                if not isinstance(block, dict):
                    raise UpstreamResponseFormatError(
                        reason="vertex anthropic content block is not an object",
                        model=model,
                        base_url=endpoint_url,
                        message_count=len(messages),
                        status_code=response.status_code,
                        response_body=data,
                    )

                block_type = block.get("type")
                if block_type == "text":
                    text_value = block.get("text")
                    if isinstance(text_value, str):
                        content_parts.append(text_value)
                elif block_type == "tool_use":
                    tool_call_id = block.get("id")
                    tool_name = block.get("name")
                    tool_input = block.get("input")
                    if not isinstance(tool_call_id, str):
                        raise UpstreamResponseFormatError(
                            reason="vertex anthropic tool_use block id is not a string",
                            model=model,
                            base_url=endpoint_url,
                            message_count=len(messages),
                            status_code=response.status_code,
                            response_body=data,
                        )
                    if not isinstance(tool_name, str):
                        raise UpstreamResponseFormatError(
                            reason="vertex anthropic tool_use block name is not a string",
                            model=model,
                            base_url=endpoint_url,
                            message_count=len(messages),
                            status_code=response.status_code,
                            response_body=data,
                        )
                    if not isinstance(tool_input, dict):
                        raise UpstreamResponseFormatError(
                            reason="vertex anthropic tool_use block input is not an object",
                            model=model,
                            base_url=endpoint_url,
                            message_count=len(messages),
                            status_code=response.status_code,
                            response_body=data,
                        )

                    tool_calls.append(
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_input, separators=(",", ":"), sort_keys=True),
                            },
                        }
                    )

        content = "".join(content_parts)
        normalized_tool_calls = tool_calls if len(tool_calls) > 0 else None

        if content == "" and normalized_tool_calls is None:
            raise UpstreamResponseFormatError(
                reason="assistant message has empty content and no tool calls",
                model=model,
                base_url=endpoint_url,
                message_count=len(messages),
                status_code=response.status_code,
                response_body=data,
            )

        finish_reason = _map_finish_reason(data.get("stop_reason"))
        usage = _map_usage(data.get("usage"))

        logger.debug(
            "vertex anthropic parsed model={} endpoint={} content_len={} tool_calls_count={} finish_reason={} "
            "prompt_tokens={} completion_tokens={} total_tokens={}",
            model,
            endpoint_url,
            len(content),
            len(tool_calls),
            finish_reason,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )

        return UpstreamResult(
            content=content,
            usage=usage,
            tool_calls=normalized_tool_calls,
            finish_reason=finish_reason,
        )
