from __future__ import annotations

from .contracts import ChatMessage

ADAPTER_SYSTEM_PROMPT = (
    "You are a response editor. Return lgtm if draft is good, or return one or more search/replace blocks only."
)

CRITIC_SYSTEM_PROMPT = (
    "You are a critique generator. Explain what is correct, what is wrong/missing, and exact fix instructions."
)


def _render_history(messages: list[ChatMessage]) -> str:
    rendered = []
    for message in messages:
        rendered.append(f"[{message.role}] {message.content}")
    return "\n".join(rendered)


def build_adapter_messages(
    messages: list[ChatMessage], draft: str, adapter_system_prompt: str = ADAPTER_SYSTEM_PROMPT
) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=adapter_system_prompt),
        ChatMessage(
            role="user",
            content=(f"Conversation history:\n{_render_history(messages)}\n\nLatest API draft:\n{draft}"),
        ),
    ]


def build_critic_messages(
    messages: list[ChatMessage],
    system_prompt: str,
    draft: str,
    critic_system_prompt: str = CRITIC_SYSTEM_PROMPT,
) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=critic_system_prompt),
        ChatMessage(
            role="user",
            content=(
                "System instructions:\n"
                f"{system_prompt}\n\n"
                "Conversation history:\n"
                f"{_render_history(messages)}\n\n"
                "Latest API draft:\n"
                f"{draft}"
            ),
        ),
    ]


def build_critic_second_pass_messages(messages: list[ChatMessage], draft: str, critique: str) -> list[ChatMessage]:
    return [
        *messages,
        ChatMessage(
            role="system",
            content=(f"Critique for improving prior draft:\n{critique}\n\nPrior draft:\n{draft}"),
        ),
    ]
