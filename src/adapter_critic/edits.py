from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

ADAPTER_DRAFT_PAYLOAD_RE = re.compile(
    r"\A<ADAPTER_DRAFT_CONTENT>\n(?P<content>.*?)\n</ADAPTER_DRAFT_CONTENT>\n"
    r"<ADAPTER_DRAFT_TOOL_CALLS>\n(?P<tool_calls>.*?)\n</ADAPTER_DRAFT_TOOL_CALLS>\Z",
    flags=re.DOTALL,
)

ALLOWED_PATCH_PATH_RE = re.compile(r"^/(content|tool_calls|tool_calls/[0-9]+/function/(name|arguments))$")


class AdapterPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: str
    path: str
    value: Any


class AdapterStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["lgtm", "patch"]
    patches: list[AdapterPatch] = Field(default_factory=list)


def _decode_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _parse_array_index(token: str, *, path: str, size: int) -> int:
    if not token.isdigit():
        raise ValueError(f"path not found: {path}")
    index = int(token)
    if index < 0 or index >= size:
        raise ValueError(f"path not found: {path}")
    return index


def _resolve_patch_target(document: Any, path: str) -> tuple[list[Any] | dict[str, Any], int | str]:
    if not path.startswith("/"):
        raise ValueError(f"unsupported patch path: {path}")

    tokens = [_decode_pointer_token(token) for token in path.lstrip("/").split("/")]
    current = document
    for token in tokens[:-1]:
        if isinstance(current, list):
            current = current[_parse_array_index(token, path=path, size=len(current))]
            continue
        if isinstance(current, dict):
            if token not in current:
                raise ValueError(f"path not found: {path}")
            current = current[token]
            continue
        raise ValueError(f"path not found: {path}")

    final_token = tokens[-1]
    if isinstance(current, list):
        index = _parse_array_index(final_token, path=path, size=len(current))
        return current, index
    if isinstance(current, dict):
        if final_token not in current:
            raise ValueError(f"path not found: {path}")
        return current, final_token
    raise ValueError(f"path not found: {path}")


def _parse_adapter_output(adapter_output: str) -> AdapterStructuredOutput:
    parsed_output = json.loads(adapter_output)
    if not isinstance(parsed_output, dict):
        raise ValueError("adapter output must be a JSON object")

    structured_output = AdapterStructuredOutput.model_validate(parsed_output)
    if structured_output.decision == "lgtm" and len(structured_output.patches) > 0:
        raise ValueError("lgtm decision must not include patches")
    if structured_output.decision == "patch" and len(structured_output.patches) == 0:
        raise ValueError("patch decision requires non-empty patches")
    return structured_output


def _apply_replace_patch(document: dict[str, Any], patch: AdapterPatch) -> None:
    if patch.op != "replace":
        raise ValueError(f"unsupported patch op: {patch.op}")
    if ALLOWED_PATCH_PATH_RE.fullmatch(patch.path) is None:
        raise ValueError(f"unsupported patch path: {patch.path}")

    target, key = _resolve_patch_target(document, patch.path)
    if isinstance(target, list):
        if not isinstance(key, int):
            raise ValueError(f"path not found: {patch.path}")
        target[key] = patch.value
        return

    if not isinstance(key, str):
        raise ValueError(f"path not found: {patch.path}")
    target[key] = patch.value


def _coerce_patched_draft(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None]:
    content_value = payload.get("content")
    if not isinstance(content_value, str):
        raise ValueError("adapter draft content must be a string")

    tool_calls_value = payload.get("tool_calls")
    if tool_calls_value is None:
        tool_calls: list[dict[str, Any]] | None = None
    elif isinstance(tool_calls_value, list) and all(isinstance(item, dict) for item in tool_calls_value):
        tool_calls = cast(list[dict[str, Any]], tool_calls_value)
    else:
        raise ValueError("adapter draft tool_calls must be a list of objects or null")

    if tool_calls is not None and len(tool_calls) == 0:
        tool_calls = None

    return content_value, tool_calls


def apply_adapter_output(draft: str, adapter_output: str) -> str:
    updated_draft, _ = apply_adapter_output_to_draft(
        content=draft,
        tool_calls=None,
        adapter_output=adapter_output,
    )
    return updated_draft


def build_adapter_draft_payload(
    content: str,
    tool_calls: list[dict[str, Any]] | None,
) -> str:
    tool_calls_json = json.dumps(tool_calls or [], indent=2, sort_keys=True)
    return (
        "<ADAPTER_DRAFT_CONTENT>\n"
        f"{content}\n"
        "</ADAPTER_DRAFT_CONTENT>\n"
        "<ADAPTER_DRAFT_TOOL_CALLS>\n"
        f"{tool_calls_json}\n"
        "</ADAPTER_DRAFT_TOOL_CALLS>"
    )


def parse_adapter_draft_payload(payload: str) -> tuple[str, list[dict[str, Any]] | None]:
    match = ADAPTER_DRAFT_PAYLOAD_RE.fullmatch(payload)
    if match is None:
        raise ValueError("malformed adapter draft payload")

    content = match.group("content")

    tool_calls_value = json.loads(match.group("tool_calls"))
    if not isinstance(tool_calls_value, list) or any(not isinstance(item, dict) for item in tool_calls_value):
        raise ValueError("adapter draft tool_calls must be a list of objects")

    tool_calls = cast(list[dict[str, Any]], tool_calls_value)
    return content, (tool_calls if len(tool_calls) > 0 else None)


def apply_adapter_output_to_draft(
    content: str,
    tool_calls: list[dict[str, Any]] | None,
    adapter_output: str,
) -> tuple[str, list[dict[str, Any]] | None]:
    structured_output = _parse_adapter_output(adapter_output)
    if structured_output.decision == "lgtm":
        return content, tool_calls

    patched_payload: dict[str, Any] = {
        "content": content,
        "tool_calls": deepcopy(tool_calls),
    }
    for patch in structured_output.patches:
        _apply_replace_patch(patched_payload, patch)

    return _coerce_patched_draft(patched_payload)
