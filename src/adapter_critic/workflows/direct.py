from __future__ import annotations

from pydantic import BaseModel

from ..config import RuntimeConfig
from ..contracts import ChatMessage
from ..upstream import TokenUsage, UpstreamGateway


class WorkflowOutput(BaseModel):
    final_text: str
    intermediate: dict[str, str]
    stage_usage: dict[str, TokenUsage]


async def run_direct(runtime: RuntimeConfig, messages: list[ChatMessage], gateway: UpstreamGateway) -> WorkflowOutput:
    response = await gateway.complete(model=runtime.api.model, base_url=runtime.api.base_url, messages=messages)
    return WorkflowOutput(
        final_text=response.content,
        intermediate={"api": response.content},
        stage_usage={"api": response.usage},
    )
