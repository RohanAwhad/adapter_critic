from __future__ import annotations

from adapter_critic.contracts import ChatMessage
from adapter_critic.prompts import (
    ADAPTER_RESPONSE_FORMAT,
    ADAPTER_SYSTEM_PROMPT,
    build_adapter_messages,
    build_critic_messages,
)


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


def test_adapter_prompt_includes_tool_contract_when_tools_are_provided() -> None:
    messages = [ChatMessage(role="user", content="cancel reservation EHGLP3")]
    prompt_messages = build_adapter_messages(
        messages=messages,
        draft="<ADAPTER_DRAFT_CONTENT>\n\n</ADAPTER_DRAFT_CONTENT>",
        request_options={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "cancel_reservation",
                        "parameters": {
                            "type": "object",
                            "properties": {"reservation_id": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )

    system_content = prompt_messages[0].content
    assert system_content is not None
    assert "Authoritative tool contract for this request" in system_content
    assert '"name": "cancel_reservation"' in system_content
    assert '"tool_choice": "auto"' in system_content
