from __future__ import annotations

from typing import Any

from .config import RuntimeConfig
from .contracts import ChatMessage
from .upstream import UpstreamGateway
from .workflows import run_adapter, run_advisor, run_critic, run_direct
from .workflows.direct import WorkflowOutput


async def dispatch(
    runtime: RuntimeConfig,
    messages: list[ChatMessage],
    gateway: UpstreamGateway,
    request_options: dict[str, Any],
) -> WorkflowOutput:
    if runtime.mode == "direct":
        return await run_direct(
            runtime=runtime,
            messages=messages,
            gateway=gateway,
            request_options=request_options,
        )
    if runtime.mode == "adapter":
        return await run_adapter(
            runtime=runtime,
            messages=messages,
            gateway=gateway,
            request_options=request_options,
        )
    if runtime.mode == "advisor":
        return await run_advisor(
            runtime=runtime,
            messages=messages,
            gateway=gateway,
            request_options=request_options,
        )
    return await run_critic(
        runtime=runtime,
        messages=messages,
        gateway=gateway,
        request_options=request_options,
    )
