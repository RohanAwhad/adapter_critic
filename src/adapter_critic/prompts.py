from __future__ import annotations

import json
from typing import Any

from .contracts import ChatMessage

ADAPTER_SYSTEM_PROMPT = (
    "You are a response editor running in JSON mode. Respond with valid JSON only. "
    'Return {"decision":"lgtm"} if the draft is good, or return '
    '{"decision":"patch","patches":[{"op":"replace","path":"/content","value":"..."}]} '
    "to apply RFC6902-style replace patches. Never emit tool calls in your own output."
)

ADAPTER_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "adapter_patch_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["lgtm", "patch"],
                },
                "patches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "op": {"type": "string", "enum": ["replace"]},
                            "path": {"type": "string"},
                            "value": {},
                        },
                        "required": ["op", "path", "value"],
                    },
                },
            },
            "required": ["decision"],
        },
    },
}

CRITIC_SYSTEM_PROMPT = (
    "You are a critique generator. Explain what is correct, what is wrong/missing, and exact fix instructions."
)


def _render_history(messages: list[ChatMessage]) -> str:
    rendered = []
    for message in messages:
        rendered.append(f"[{message.role}] {message.content or ''}")
    return "\n".join(rendered)


def _render_tool_contract(request_options: dict[str, Any] | None) -> str | None:
    if request_options is None:
        return None

    contract: dict[str, Any] = {}
    tools = request_options.get("tools")
    if isinstance(tools, list) and len(tools) > 0:
        contract["tools"] = tools

    tool_choice = request_options.get("tool_choice")
    if tool_choice is not None:
        contract["tool_choice"] = tool_choice

    if len(contract) == 0:
        return None
    return json.dumps(contract, indent=2, sort_keys=True, default=str)


def build_adapter_messages(
    messages: list[ChatMessage],
    draft: str,
    adapter_system_prompt: str = ADAPTER_SYSTEM_PROMPT,
    request_options: dict[str, Any] | None = None,
) -> list[ChatMessage]:
    tool_contract = _render_tool_contract(request_options)
    system_prompt_content = adapter_system_prompt
    if tool_contract is not None:
        system_prompt_content = (
            f"{adapter_system_prompt}\n\n"
            "Authoritative tool contract for this request:\n"
            f"{tool_contract}\n\n"
            "Never emit tool calls directly. Return only the structured JSON adapter response."
        )

    return [
        ChatMessage(role="system", content=system_prompt_content),
        ChatMessage(
            role="user",
            content=(f"Conversation history:\n{_render_history(messages)}\n\nLatest API draft:\n{draft}"),
        ),
    ]


def build_critic_messages(
    messages: list[ChatMessage],
    system_prompt: str,
    draft: str,
    critic_system_prompt: str = CRITIC_SYSTEM_PROMPT,
    request_options: dict[str, Any] | None = None,
) -> list[ChatMessage]:
    tool_contract = _render_tool_contract(request_options)
    system_prompt_content = critic_system_prompt
    if tool_contract is not None:
        system_prompt_content = (
            f"{critic_system_prompt}\n\n"
            "Authoritative tool contract for this request:\n"
            f"{tool_contract}\n\n"
            "Evaluate tool usage against this contract. Never emit tool calls yourself."
        )

    return [
        ChatMessage(role="system", content=system_prompt_content),
        ChatMessage(
            role="user",
            content=(
                "System instructions:\n"
                f"{system_prompt}\n\n"
                "Conversation history:\n"
                f"{_render_history(messages)}\n\n"
                "Latest API draft:\n"
                f"{draft}"
            ),
        ),
    ]


def build_critic_second_pass_messages(messages: list[ChatMessage], draft: str, critique: str) -> list[ChatMessage]:
    return [
        *messages,
        ChatMessage(
            role="system",
            content=(f"Critique for improving prior draft:\n{critique}\n\nPrior draft:\n{draft}"),
        ),
    ]
