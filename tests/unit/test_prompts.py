from __future__ import annotations

from adapter_critic.contracts import ChatMessage
from adapter_critic.prompts import build_critic_messages


def test_critic_prompt_contains_required_inputs() -> None:
    messages = [
        ChatMessage(role="system", content="sys rule"),
        ChatMessage(role="user", content="question"),
    ]
    prompt_messages = build_critic_messages(messages=messages, system_prompt="sys rule", draft="api draft")
    assert "sys rule" in prompt_messages[1].content
    assert "question" in prompt_messages[1].content
    assert "api draft" in prompt_messages[1].content
