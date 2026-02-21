from __future__ import annotations

import json
from typing import Any

from ..config import RuntimeConfig
from ..contracts import ChatMessage
from ..edits import build_adapter_draft_payload
from ..prompts import build_critic_messages, build_critic_second_pass_messages
from ..response_shape import normalize_tool_calls
from ..upstream import UpstreamGateway
from .direct import WorkflowOutput


def _first_system_prompt(messages: list[ChatMessage]) -> str:
    for message in messages:
        if message.role == "system":
            content = message.content
            if content is None:
                return ""
            return content
    return ""


async def run_critic(
    runtime: RuntimeConfig,
    messages: list[ChatMessage],
    gateway: UpstreamGateway,
    request_options: dict[str, Any],
) -> WorkflowOutput:
    if runtime.critic is None:
        raise ValueError("critic runtime is missing critic target")

    api_draft = await gateway.complete(
        model=runtime.api.model,
        base_url=runtime.api.base_url,
        messages=messages,
        api_key_env=runtime.api.api_key_env,
        request_options=request_options,
    )
    api_tool_calls = normalize_tool_calls(api_draft.tool_calls)
    api_function_call = api_draft.function_call if api_tool_calls is None else None

    draft_payload = build_adapter_draft_payload(
        content=api_draft.content,
        tool_calls=api_tool_calls,
        function_call=api_function_call,
    )

    critic_messages = build_critic_messages(
        messages=messages,
        system_prompt=_first_system_prompt(messages),
        draft=draft_payload,
        critic_system_prompt=runtime.critic_system_prompt,
        request_options=request_options,
    )
    critic_feedback = await gateway.complete(
        model=runtime.critic.model,
        base_url=runtime.critic.base_url,
        messages=critic_messages,
        api_key_env=runtime.critic.api_key_env,
    )
    second_pass_messages = build_critic_second_pass_messages(
        messages=messages,
        draft=draft_payload,
        critique=critic_feedback.content,
    )
    final_response = await gateway.complete(
        model=runtime.api.model,
        base_url=runtime.api.base_url,
        messages=second_pass_messages,
        api_key_env=runtime.api.api_key_env,
        request_options=request_options,
    )
    intermediate: dict[str, str] = {
        "api_draft": api_draft.content,
        "critic": critic_feedback.content,
        "final": final_response.content,
    }
    if api_tool_calls is not None:
        intermediate["api_draft_tool_calls"] = json.dumps(api_tool_calls, sort_keys=True)
    if api_function_call is not None:
        intermediate["api_draft_function_call"] = json.dumps(api_function_call, sort_keys=True)

    return WorkflowOutput(
        final_text=final_response.content,
        intermediate=intermediate,
        stage_usage={
            "api_draft": api_draft.usage,
            "critic": critic_feedback.usage,
            "api_final": final_response.usage,
        },
        final_tool_calls=final_response.tool_calls,
        final_function_call=final_response.function_call,
        finish_reason=final_response.finish_reason,
    )
