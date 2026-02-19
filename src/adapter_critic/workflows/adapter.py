from __future__ import annotations

from typing import Any

from ..config import RuntimeConfig
from ..contracts import ChatMessage
from ..edits import apply_adapter_output_to_draft, build_adapter_draft_payload
from ..prompts import build_adapter_messages
from ..upstream import UpstreamGateway
from .direct import WorkflowOutput


async def run_adapter(
    runtime: RuntimeConfig,
    messages: list[ChatMessage],
    gateway: UpstreamGateway,
    request_options: dict[str, Any],
) -> WorkflowOutput:
    if runtime.adapter is None:
        raise ValueError("adapter runtime is missing adapter target")

    api_draft = await gateway.complete(
        model=runtime.api.model,
        base_url=runtime.api.base_url,
        messages=messages,
        api_key_env=runtime.api.api_key_env,
        request_options=request_options,
    )

    draft_payload = build_adapter_draft_payload(
        content=api_draft.content,
        tool_calls=api_draft.tool_calls,
        function_call=api_draft.function_call,
    )

    adapter_messages = build_adapter_messages(
        messages=messages,
        draft=draft_payload,
        adapter_system_prompt=runtime.adapter_system_prompt,
    )
    adapter_review = await gateway.complete(
        model=runtime.adapter.model,
        base_url=runtime.adapter.base_url,
        messages=adapter_messages,
        api_key_env=runtime.adapter.api_key_env,
    )

    final_text, final_tool_calls, final_function_call = apply_adapter_output_to_draft(
        content=api_draft.content,
        tool_calls=api_draft.tool_calls,
        function_call=api_draft.function_call,
        adapter_output=adapter_review.content,
    )
    finish_reason = (
        "tool_calls"
        if final_tool_calls is not None
        else ("function_call" if final_function_call is not None else "stop")
    )

    return WorkflowOutput(
        final_text=final_text,
        intermediate={
            "api_draft": api_draft.content,
            "adapter": adapter_review.content,
            "final": final_text,
        },
        stage_usage={"api": api_draft.usage, "adapter": adapter_review.usage},
        final_tool_calls=final_tool_calls,
        final_function_call=final_function_call,
        finish_reason=finish_reason,
    )
