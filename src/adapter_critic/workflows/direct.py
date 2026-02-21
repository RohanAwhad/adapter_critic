from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..config import RuntimeConfig
from ..contracts import ChatMessage
from ..upstream import TokenUsage, UpstreamGateway


class WorkflowOutput(BaseModel):
    final_text: str
    intermediate: dict[str, str]
    stage_usage: dict[str, TokenUsage]
    final_tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str = "stop"


async def run_direct(
    runtime: RuntimeConfig,
    messages: list[ChatMessage],
    gateway: UpstreamGateway,
    request_options: dict[str, Any],
) -> WorkflowOutput:
    response = await gateway.complete(
        model=runtime.api.model,
        base_url=runtime.api.base_url,
        messages=messages,
        api_key_env=runtime.api.api_key_env,
        request_options=request_options,
    )
    return WorkflowOutput(
        final_text=response.content,
        intermediate={"api": response.content},
        stage_usage={"api": response.usage},
        final_tool_calls=response.tool_calls,
        finish_reason=response.finish_reason,
    )
