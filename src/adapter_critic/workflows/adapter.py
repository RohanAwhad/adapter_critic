from __future__ import annotations

from ..config import RuntimeConfig
from ..contracts import ChatMessage
from ..edits import apply_adapter_output
from ..prompts import build_adapter_messages
from ..upstream import UpstreamGateway
from .direct import WorkflowOutput


async def run_adapter(runtime: RuntimeConfig, messages: list[ChatMessage], gateway: UpstreamGateway) -> WorkflowOutput:
    if runtime.adapter is None:
        raise ValueError("adapter runtime is missing adapter target")

    api_draft = await gateway.complete(model=runtime.api.model, base_url=runtime.api.base_url, messages=messages)
    adapter_messages = build_adapter_messages(messages=messages, draft=api_draft.content)
    adapter_review = await gateway.complete(
        model=runtime.adapter.model,
        base_url=runtime.adapter.base_url,
        messages=adapter_messages,
    )
    final_text = apply_adapter_output(draft=api_draft.content, adapter_output=adapter_review.content)
    return WorkflowOutput(
        final_text=final_text,
        intermediate={
            "api_draft": api_draft.content,
            "adapter": adapter_review.content,
            "final": final_text,
        },
        stage_usage={"api": api_draft.usage, "adapter": adapter_review.usage},
    )
