from __future__ import annotations

import json
import re
from typing import Any, cast

EDIT_BLOCK_RE = re.compile(
    r"<<<<<<<\s*SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>>\s*REPLACE",
    flags=re.DOTALL,
)

ADAPTER_DRAFT_PAYLOAD_RE = re.compile(
    r"\A<ADAPTER_DRAFT_CONTENT>\n(?P<content>.*?)\n</ADAPTER_DRAFT_CONTENT>\n"
    r"<ADAPTER_DRAFT_TOOL_CALLS>\n(?P<tool_calls>.*?)\n</ADAPTER_DRAFT_TOOL_CALLS>\n"
    r"<ADAPTER_DRAFT_FUNCTION_CALL>\n(?P<function_call>.*?)\n</ADAPTER_DRAFT_FUNCTION_CALL>\Z",
    flags=re.DOTALL,
)


def apply_adapter_output(draft: str, adapter_output: str) -> str:
    stripped = adapter_output.strip().lower()
    if stripped == "lgtm":
        return draft

    matches = EDIT_BLOCK_RE.findall(adapter_output)
    if not matches:
        raise ValueError("malformed adapter edits")

    updated = draft
    for search_text, replace_text in matches:
        if search_text not in updated:
            raise ValueError("search text not found in draft")
        updated = updated.replace(search_text, replace_text, 1)

    return updated


def build_adapter_draft_payload(
    content: str,
    tool_calls: list[dict[str, Any]] | None,
    function_call: dict[str, Any] | None,
) -> str:
    tool_calls_json = json.dumps(tool_calls or [], indent=2, sort_keys=True)
    function_call_json = json.dumps(function_call, indent=2, sort_keys=True)
    return (
        "<ADAPTER_DRAFT_CONTENT>\n"
        f"{content}\n"
        "</ADAPTER_DRAFT_CONTENT>\n"
        "<ADAPTER_DRAFT_TOOL_CALLS>\n"
        f"{tool_calls_json}\n"
        "</ADAPTER_DRAFT_TOOL_CALLS>\n"
        "<ADAPTER_DRAFT_FUNCTION_CALL>\n"
        f"{function_call_json}\n"
        "</ADAPTER_DRAFT_FUNCTION_CALL>"
    )


def parse_adapter_draft_payload(payload: str) -> tuple[str, list[dict[str, Any]] | None, dict[str, Any] | None]:
    match = ADAPTER_DRAFT_PAYLOAD_RE.fullmatch(payload)
    if match is None:
        raise ValueError("malformed adapter draft payload")

    content = match.group("content")

    tool_calls_value = json.loads(match.group("tool_calls"))
    if not isinstance(tool_calls_value, list) or any(not isinstance(item, dict) for item in tool_calls_value):
        raise ValueError("adapter draft tool_calls must be a list of objects")

    function_call_value = json.loads(match.group("function_call"))
    if function_call_value is not None and not isinstance(function_call_value, dict):
        raise ValueError("adapter draft function_call must be an object")

    tool_calls = cast(list[dict[str, Any]], tool_calls_value)
    function_call = cast(dict[str, Any] | None, function_call_value)
    return content, (tool_calls if len(tool_calls) > 0 else None), function_call


def apply_adapter_output_to_draft(
    content: str,
    tool_calls: list[dict[str, Any]] | None,
    function_call: dict[str, Any] | None,
    adapter_output: str,
) -> tuple[str, list[dict[str, Any]] | None, dict[str, Any] | None]:
    draft_payload = build_adapter_draft_payload(content=content, tool_calls=tool_calls, function_call=function_call)
    stripped = adapter_output.strip().lower()
    if stripped == "lgtm":
        return content, tool_calls, function_call

    matches = EDIT_BLOCK_RE.findall(adapter_output)
    if not matches:
        raise ValueError("malformed adapter edits")

    updated_payload = draft_payload
    for search_text, replace_text in matches:
        if search_text not in updated_payload:
            return content, tool_calls, function_call
        updated_payload = updated_payload.replace(search_text, replace_text, 1)

    return parse_adapter_draft_payload(updated_payload)
