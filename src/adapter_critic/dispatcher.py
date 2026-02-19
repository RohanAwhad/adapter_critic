from __future__ import annotations

from .config import RuntimeConfig
from .contracts import ChatMessage
from .upstream import UpstreamGateway
from .workflows import run_adapter, run_critic, run_direct
from .workflows.direct import WorkflowOutput


async def dispatch(runtime: RuntimeConfig, messages: list[ChatMessage], gateway: UpstreamGateway) -> WorkflowOutput:
    if runtime.mode == "direct":
        return await run_direct(runtime=runtime, messages=messages, gateway=gateway)
    if runtime.mode == "adapter":
        return await run_adapter(runtime=runtime, messages=messages, gateway=gateway)
    return await run_critic(runtime=runtime, messages=messages, gateway=gateway)
