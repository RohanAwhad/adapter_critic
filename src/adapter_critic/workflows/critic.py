from __future__ import annotations

from ..config import RuntimeConfig
from ..contracts import ChatMessage
from ..prompts import build_critic_messages, build_critic_second_pass_messages
from ..upstream import UpstreamGateway
from .direct import WorkflowOutput


def _first_system_prompt(messages: list[ChatMessage]) -> str:
    for message in messages:
        if message.role == "system":
            return message.content
    return ""


async def run_critic(runtime: RuntimeConfig, messages: list[ChatMessage], gateway: UpstreamGateway) -> WorkflowOutput:
    if runtime.critic is None:
        raise ValueError("critic runtime is missing critic target")

    api_draft = await gateway.complete(model=runtime.api.model, base_url=runtime.api.base_url, messages=messages)
    critic_messages = build_critic_messages(
        messages=messages,
        system_prompt=_first_system_prompt(messages),
        draft=api_draft.content,
    )
    critic_feedback = await gateway.complete(
        model=runtime.critic.model,
        base_url=runtime.critic.base_url,
        messages=critic_messages,
    )
    second_pass_messages = build_critic_second_pass_messages(
        messages=messages,
        draft=api_draft.content,
        critique=critic_feedback.content,
    )
    final_response = await gateway.complete(
        model=runtime.api.model,
        base_url=runtime.api.base_url,
        messages=second_pass_messages,
    )
    return WorkflowOutput(
        final_text=final_response.content,
        intermediate={
            "api_draft": api_draft.content,
            "critic": critic_feedback.content,
            "final": final_response.content,
        },
        stage_usage={
            "api_draft": api_draft.usage,
            "critic": critic_feedback.usage,
            "api_final": final_response.usage,
        },
    )
