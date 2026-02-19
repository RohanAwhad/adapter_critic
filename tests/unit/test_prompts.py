from __future__ import annotations

from adapter_critic.contracts import ChatMessage
from adapter_critic.prompts import build_critic_messages


def test_critic_prompt_contains_required_inputs() -> None:
    messages = [
        ChatMessage(role="system", content="sys rule"),
        ChatMessage(role="user", content="question"),
    ]
    prompt_messages = build_critic_messages(messages=messages, system_prompt="sys rule", draft="api draft")
    content = prompt_messages[1].content
    assert content is not None
    assert "sys rule" in content
    assert "question" in content
    assert "api draft" in content
