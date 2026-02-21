from __future__ import annotations

import json
from typing import Any

import httpx
from loguru import logger

from ..config import RuntimeConfig
from ..contracts import ChatMessage
from ..edits import build_adapter_draft_payload
from ..http_gateway import UpstreamResponseFormatError
from ..prompts import build_critic_messages, build_critic_second_pass_messages
from ..response_shape import normalize_tool_calls
from ..upstream import TokenUsage, UpstreamGateway, UpstreamResult
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

    draft_payload = build_adapter_draft_payload(
        content=api_draft.content,
        tool_calls=api_tool_calls,
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
    final_response: UpstreamResult | None = None
    final_fallback_reason: str | None = None
    final_attempts = 2
    for attempt in range(1, final_attempts + 1):
        try:
            final_response = await gateway.complete(
                model=runtime.api.model,
                base_url=runtime.api.base_url,
                messages=second_pass_messages,
                api_key_env=runtime.api.api_key_env,
                request_options=request_options,
            )
            break
        except (UpstreamResponseFormatError, httpx.HTTPError) as exc:
            logger.warning(
                "critic final pass attempt failed model={} base_url={} attempt={}/{} error_type={} detail={}",
                runtime.api.model,
                runtime.api.base_url,
                attempt,
                final_attempts,
                type(exc).__name__,
                str(exc),
            )
            if attempt == final_attempts:
                final_fallback_reason = (
                    f"api_final failed after {final_attempts} attempts: {type(exc).__name__}: {str(exc)}"
                )

    if final_response is None:
        final_text = api_draft.content
        final_tool_calls = api_tool_calls
        finish_reason = api_draft.finish_reason
        api_final_usage = TokenUsage()
    else:
        final_text = final_response.content
        final_tool_calls = final_response.tool_calls
        finish_reason = final_response.finish_reason
        api_final_usage = final_response.usage

    intermediate: dict[str, str] = {
        "api_draft": api_draft.content,
        "critic": critic_feedback.content,
        "final": final_text,
    }
    if api_tool_calls is not None:
        intermediate["api_draft_tool_calls"] = json.dumps(api_tool_calls, sort_keys=True)
    if final_fallback_reason is not None:
        intermediate["final_fallback_reason"] = final_fallback_reason

    return WorkflowOutput(
        final_text=final_text,
        intermediate=intermediate,
        stage_usage={
            "api_draft": api_draft.usage,
            "critic": critic_feedback.usage,
            "api_final": api_final_usage,
        },
        final_tool_calls=final_tool_calls,
        finish_reason=finish_reason,
    )
