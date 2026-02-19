from __future__ import annotations

from adapter_critic.contracts import ChatMessage
from adapter_critic.prompts import ADAPTER_RESPONSE_FORMAT, ADAPTER_SYSTEM_PROMPT, build_critic_messages


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


def test_default_adapter_prompt_mentions_json_mode() -> None:
    assert "json" in ADAPTER_SYSTEM_PROMPT.lower()


def test_adapter_response_format_uses_strict_json_schema() -> None:
    assert ADAPTER_RESPONSE_FORMAT["type"] == "json_schema"
    json_schema = ADAPTER_RESPONSE_FORMAT["json_schema"]
    assert json_schema["strict"] is True
    assert json_schema["schema"]["properties"]["decision"]["enum"] == ["lgtm", "patch"]
