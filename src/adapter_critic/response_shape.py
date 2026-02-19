from __future__ import annotations

from typing import Any


def normalize_tool_calls(tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if tool_calls is None or len(tool_calls) == 0:
        return None
    return tool_calls


def has_valid_tool_calls(tool_calls: list[dict[str, Any]] | None) -> bool:
    normalized = normalize_tool_calls(tool_calls)
    if normalized is None:
        return True

    for tool_call in normalized:
        if not isinstance(tool_call.get("id"), str):
            return False
        if tool_call.get("type") != "function":
            return False
        function = tool_call.get("function")
        if not isinstance(function, dict):
            return False
        if not isinstance(function.get("name"), str):
            return False
        if not isinstance(function.get("arguments"), str):
            return False

    return True


def has_valid_function_call(function_call: dict[str, Any] | None) -> bool:
    if function_call is None:
        return True
    return isinstance(function_call.get("name"), str) and isinstance(function_call.get("arguments"), str)


def infer_finish_reason(
    raw_finish_reason: str,
    *,
    tool_calls: list[dict[str, Any]] | None,
    function_call: dict[str, Any] | None,
) -> str:
    normalized_tool_calls = normalize_tool_calls(tool_calls)
    if normalized_tool_calls is not None:
        return "tool_calls"
    if function_call is not None:
        return "function_call"
    if raw_finish_reason in {"length", "content_filter"}:
        return raw_finish_reason
    return "stop"
