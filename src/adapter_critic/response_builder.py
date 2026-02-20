from __future__ import annotations

from typing import Any

from .contracts import ChatCompletionRequest, Mode
from .response_shape import infer_finish_reason, normalize_tool_calls
from .usage import TokenBreakdown


def build_response(
    request: ChatCompletionRequest,
    *,
    mode: Mode,
    final_text: str,
    intermediate: dict[str, str],
    tokens: TokenBreakdown,
    response_id: str,
    created: int,
    final_tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    normalized_tool_calls = normalize_tool_calls(final_tool_calls)
    response_finish_reason = infer_finish_reason(
        finish_reason,
        tool_calls=normalized_tool_calls,
    )

    message: dict[str, Any] = {"role": "assistant", "content": final_text}
    if normalized_tool_calls is not None:
        message["tool_calls"] = normalized_tool_calls

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": response_finish_reason,
            }
        ],
        "usage": tokens.total.model_dump(),
        "adapter_critic": {
            "mode": mode,
            "intermediate": intermediate,
            "tokens": {
                "stages": {name: usage.model_dump() for name, usage in tokens.stages.items()},
                "total": tokens.total.model_dump(),
            },
        },
    }
