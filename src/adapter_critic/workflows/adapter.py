from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..config import RuntimeConfig
from ..contracts import ChatMessage
from ..edits import apply_adapter_output_to_draft, build_adapter_draft_payload
from ..prompts import ADAPTER_RESPONSE_FORMAT, build_adapter_messages
from ..response_shape import (
    has_valid_function_call,
    has_valid_tool_calls,
    infer_finish_reason,
    normalize_tool_calls,
)
from ..upstream import TokenUsage, UpstreamGateway
from .direct import WorkflowOutput


def _add_usage(total: TokenUsage, current: TokenUsage) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=total.prompt_tokens + current.prompt_tokens,
        completion_tokens=total.completion_tokens + current.completion_tokens,
        total_tokens=total.total_tokens + current.total_tokens,
    )


def _is_adapter_candidate_usable(
    *,
    content: str,
    tool_calls: list[dict[str, Any]] | None,
    function_call: dict[str, Any] | None,
    require_call: bool,
) -> bool:
    normalized_tool_calls = normalize_tool_calls(tool_calls)
    has_call = normalized_tool_calls is not None or function_call is not None
    if normalized_tool_calls is not None and not has_valid_tool_calls(normalized_tool_calls):
        return False
    if function_call is not None and not has_valid_function_call(function_call):
        return False
    if normalized_tool_calls is not None and function_call is not None:
        return False
    if content == "" and not has_call:
        return False
    return not (require_call and not has_call)


def _requires_tool_call(request_options: dict[str, Any]) -> bool:
    tool_choice = request_options.get("tool_choice")
    if tool_choice == "required":
        return True
    if isinstance(tool_choice, dict):
        return tool_choice.get("type") == "function"

    function_call = request_options.get("function_call")
    return bool(isinstance(function_call, dict))


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
    api_tool_calls = normalize_tool_calls(api_draft.tool_calls)
    api_function_call = api_draft.function_call if api_tool_calls is None else None
    requested_requires_call = _requires_tool_call(request_options)

    draft_payload = build_adapter_draft_payload(
        content=api_draft.content,
        tool_calls=api_tool_calls,
        function_call=api_function_call,
    )

    adapter_messages = build_adapter_messages(
        messages=messages,
        draft=draft_payload,
        adapter_system_prompt=runtime.adapter_system_prompt,
    )
    adapter_usage = TokenUsage()
    adapter_output = ""
    adapter_request_options = {"response_format": deepcopy(ADAPTER_RESPONSE_FORMAT)}

    final_text = api_draft.content
    final_tool_calls = api_tool_calls
    final_function_call = api_function_call

    max_attempts = runtime.max_adapter_retries + 1
    for _ in range(max_attempts):
        adapter_review = await gateway.complete(
            model=runtime.adapter.model,
            base_url=runtime.adapter.base_url,
            messages=adapter_messages,
            api_key_env=runtime.adapter.api_key_env,
            request_options=adapter_request_options,
        )
        adapter_usage = _add_usage(adapter_usage, adapter_review.usage)
        adapter_output = adapter_review.content
        try:
            candidate_text, candidate_tool_calls, candidate_function_call = apply_adapter_output_to_draft(
                content=api_draft.content,
                tool_calls=api_tool_calls,
                function_call=api_function_call,
                adapter_output=adapter_review.content,
            )
        except ValueError:
            continue

        candidate_tool_calls = normalize_tool_calls(candidate_tool_calls)
        candidate_function_call = candidate_function_call if candidate_tool_calls is None else None
        if not _is_adapter_candidate_usable(
            content=candidate_text,
            tool_calls=candidate_tool_calls,
            function_call=candidate_function_call,
            require_call=requested_requires_call,
        ):
            continue

        final_text = candidate_text
        final_tool_calls = candidate_tool_calls
        final_function_call = candidate_function_call
        break

    finish_reason = infer_finish_reason(
        "stop",
        tool_calls=final_tool_calls,
        function_call=final_function_call,
    )

    return WorkflowOutput(
        final_text=final_text,
        intermediate={
            "api_draft": api_draft.content,
            "adapter": adapter_output,
            "final": final_text,
        },
        stage_usage={"api": api_draft.usage, "adapter": adapter_usage},
        final_tool_calls=final_tool_calls,
        final_function_call=final_function_call,
        finish_reason=finish_reason,
    )
