from __future__ import annotations

from typing import Any

from ..config import RuntimeConfig
from ..contracts import ChatMessage
from ..prompts import append_advisor_guidance_to_last_user_message, build_advisor_messages
from ..upstream import UpstreamGateway
from .direct import WorkflowOutput


async def run_advisor(
    runtime: RuntimeConfig,
    messages: list[ChatMessage],
    gateway: UpstreamGateway,
    request_options: dict[str, Any],
) -> WorkflowOutput:
    if runtime.advisor is None:
        raise ValueError("advisor runtime is missing advisor target")

    advisor_messages = build_advisor_messages(
        messages=messages,
        advisor_system_prompt=runtime.advisor_system_prompt,
        request_options=request_options,
    )
    advisor_feedback = await gateway.complete(
        model=runtime.advisor.model,
        base_url=runtime.advisor.base_url,
        messages=advisor_messages,
        api_key_env=runtime.advisor.api_key_env,
    )

    api_messages = append_advisor_guidance_to_last_user_message(
        messages=messages,
        advisor_guidance=advisor_feedback.content,
    )
    api_response = await gateway.complete(
        model=runtime.api.model,
        base_url=runtime.api.base_url,
        messages=api_messages,
        api_key_env=runtime.api.api_key_env,
        request_options=request_options,
    )

    return WorkflowOutput(
        final_text=api_response.content,
        intermediate={
            "advisor": advisor_feedback.content,
            "final": api_response.content,
        },
        stage_usage={
            "advisor": advisor_feedback.usage,
            "api": api_response.usage,
        },
        final_tool_calls=api_response.tool_calls,
        finish_reason=api_response.finish_reason,
    )
