from __future__ import annotations

import time
import uuid
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
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
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
