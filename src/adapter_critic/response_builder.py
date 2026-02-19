from __future__ import annotations

from typing import Any

from .contracts import ChatCompletionRequest, Mode
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
    final_function_call: dict[str, Any] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": final_text}
    if final_tool_calls is not None:
        message["tool_calls"] = final_tool_calls
    if final_function_call is not None:
        message["function_call"] = final_function_call

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
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
