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
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": final_text},
                "finish_reason": "stop",
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
