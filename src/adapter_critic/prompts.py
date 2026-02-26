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

ADVISOR_SYSTEM_PROMPT = (
    "You are an expert advisor for another language model. "
    "Provide concise, actionable guidance on how to solve the user's request: where to look, "
    "what steps/tools to use, what pitfalls to avoid, and what the final answer must include. "
    "Do not answer the user directly. Do not emit tool calls. Return guidance only."
)

ADVISOR_GUIDANCE_OPEN_TAG = "[ADVISOR_GUIDANCE]"
ADVISOR_GUIDANCE_CLOSE_TAG = "[/ADVISOR_GUIDANCE]"


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
            role="user",
            content=(
                f"Below is your prior draft and feedback from a small critic model.\n"
                f"The critic is less capable than you — use your own judgment about whether to act on its feedback.\n\n"
                f"Critic feedback:\n{critique}\n\n"
                f"Prior draft:\n{draft}"
            ),
        ),
    ]


def build_advisor_messages(
    messages: list[ChatMessage],
    advisor_system_prompt: str = ADVISOR_SYSTEM_PROMPT,
    request_options: dict[str, Any] | None = None,
) -> list[ChatMessage]:
    tool_contract = _render_tool_contract(request_options)
    system_prompt_content = advisor_system_prompt
    if tool_contract is not None:
        system_prompt_content = (
            f"{advisor_system_prompt}\n\n"
            "Authoritative tool contract for this request:\n"
            f"{tool_contract}\n\n"
            "Use this contract only as planning context. Never emit tool calls directly."
        )

    return [
        ChatMessage(role="system", content=system_prompt_content),
        *messages,
    ]


def _build_advisor_guidance_block(advisor_guidance: str) -> str:
    return f"{ADVISOR_GUIDANCE_OPEN_TAG}\n{advisor_guidance}\n{ADVISOR_GUIDANCE_CLOSE_TAG}"


def append_advisor_guidance_to_last_user_message(
    messages: list[ChatMessage],
    advisor_guidance: str,
) -> list[ChatMessage]:
    guidance_block = _build_advisor_guidance_block(advisor_guidance)
    updated_messages = [message.model_copy(deep=True) for message in messages]

    for index in range(len(updated_messages) - 1, -1, -1):
        message = updated_messages[index]
        if message.role != "user":
            continue

        current_content = message.content or ""
        next_content = guidance_block if current_content == "" else f"{current_content}\n\n{guidance_block}"
        updated_messages[index] = message.model_copy(update={"content": next_content})
        return updated_messages

    updated_messages.append(ChatMessage(role="user", content=guidance_block))
    return updated_messages
