from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse, urlunparse

from anthropic import AsyncAnthropicVertex
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


def _as_log_payload(payload: Any) -> Any:
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return payload


def _value_get(payload: Any, key: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


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


def _required_path_segment(path_parts: list[str], marker: str) -> str:
    try:
        marker_index = path_parts.index(marker)
    except ValueError as exc:
        raise ValueError(f"vertex base_url missing '{marker}' segment") from exc

    value_index = marker_index + 1
    if value_index >= len(path_parts) or path_parts[value_index] == "":
        raise ValueError(f"vertex base_url missing value after '{marker}' segment")
    return path_parts[value_index]


def _resolve_vertex_client_config(*, model: str, base_url: str) -> tuple[str, str, str, str]:
    parsed = urlparse(base_url.rstrip("/"))
    if parsed.scheme == "" or parsed.netloc == "":
        raise ValueError("vertex base_url must include scheme and host")

    path_parts = [part for part in parsed.path.split("/") if part != ""]
    project_id = _required_path_segment(path_parts, "projects")
    region = _required_path_segment(path_parts, "locations")

    if "v1" in path_parts:
        v1_index = path_parts.index("v1")
        base_path = "/" + "/".join(path_parts[: v1_index + 1])
    else:
        base_path = "/v1"

    client_base_url = urlunparse((parsed.scheme, parsed.netloc, base_path, "", "", ""))

    normalized_model = _normalize_vertex_model_name(model)
    if normalized_model == "":
        raise ValueError("vertex model must not be empty")

    return client_base_url, project_id, region, normalized_model


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


def _safe_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    return 0


def _map_usage(usage: Any) -> TokenUsage:
    prompt_tokens = _safe_int(_value_get(usage, "input_tokens", _value_get(usage, "prompt_tokens", 0)))
    completion_tokens = _safe_int(_value_get(usage, "output_tokens", _value_get(usage, "completion_tokens", 0)))
    total_tokens = _safe_int(_value_get(usage, "total_tokens", prompt_tokens + completion_tokens))

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

        client_base_url, project_id, region, normalized_model = _resolve_vertex_client_config(
            model=model,
            base_url=base_url,
        )

        anthropic_messages: list[dict[str, Any]] = []
        for message in messages:
            mapped_message = _message_to_vertex_content(message)
            if mapped_message is not None:
                anthropic_messages.append(mapped_message)

        mapped_request_options = _map_request_options(request_options)
        create_kwargs: dict[str, Any] = {
            "model": normalized_model,
            "messages": anthropic_messages,
            "max_tokens": int(mapped_request_options.get("max_tokens", 8192)),
        }

        system_prompt = _extract_system_prompt(messages)
        if system_prompt != "":
            create_kwargs["system"] = system_prompt

        for key, value in mapped_request_options.items():
            if key != "max_tokens":
                create_kwargs[key] = value

        logger.debug(
            "vertex anthropic request model={} client_base_url={} region={} project_id={} message_count={} payload={}",
            normalized_model,
            client_base_url,
            region,
            project_id,
            len(anthropic_messages),
            _payload_preview(create_kwargs),
        )

        client_kwargs: dict[str, Any] = {
            "region": region,
            "project_id": project_id,
            "base_url": client_base_url,
            "timeout": self._timeout_seconds,
        }

        async def _run_request(kwargs: dict[str, Any]) -> Any:
            async with AsyncAnthropicVertex(**kwargs) as client:
                return await client.messages.create(**create_kwargs)

        response = await _run_request(client_kwargs)

        logger.debug(
            "vertex anthropic raw response model={} client_base_url={} payload={}",
            normalized_model,
            client_base_url,
            _payload_preview(_as_log_payload(response), max_chars=2000),
        )

        content_value = _value_get(response, "content")
        if not isinstance(content_value, list):
            raise UpstreamResponseFormatError(
                reason="vertex anthropic content is not a list",
                model=normalized_model,
                base_url=base_url,
                message_count=len(messages),
                status_code=200,
                response_body=_as_log_payload(response),
            )

        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content_value:
            block_type = _value_get(block, "type")
            if block_type == "text":
                text_value = _value_get(block, "text")
                if isinstance(text_value, str):
                    content_parts.append(text_value)
            elif block_type == "tool_use":
                tool_call_id = _value_get(block, "id")
                tool_name = _value_get(block, "name")
                tool_input = _value_get(block, "input")
                if not isinstance(tool_call_id, str):
                    raise UpstreamResponseFormatError(
                        reason="vertex anthropic tool_use block id is not a string",
                        model=normalized_model,
                        base_url=base_url,
                        message_count=len(messages),
                        status_code=200,
                        response_body=_as_log_payload(response),
                    )
                if not isinstance(tool_name, str):
                    raise UpstreamResponseFormatError(
                        reason="vertex anthropic tool_use block name is not a string",
                        model=normalized_model,
                        base_url=base_url,
                        message_count=len(messages),
                        status_code=200,
                        response_body=_as_log_payload(response),
                    )
                if not isinstance(tool_input, dict):
                    raise UpstreamResponseFormatError(
                        reason="vertex anthropic tool_use block input is not an object",
                        model=normalized_model,
                        base_url=base_url,
                        message_count=len(messages),
                        status_code=200,
                        response_body=_as_log_payload(response),
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
                model=normalized_model,
                base_url=base_url,
                message_count=len(messages),
                status_code=200,
                response_body=_as_log_payload(response),
            )

        finish_reason = _map_finish_reason(_value_get(response, "stop_reason"))
        usage = _map_usage(_value_get(response, "usage"))

        logger.debug(
            "vertex anthropic parsed model={} content_len={} tool_calls_count={} finish_reason={} "
            "prompt_tokens={} completion_tokens={} total_tokens={}",
            normalized_model,
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
