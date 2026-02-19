from __future__ import annotations

from adapter_critic.contracts import ChatCompletionRequest
from adapter_critic.response_builder import build_response
from adapter_critic.upstream import TokenUsage
from adapter_critic.usage import TokenBreakdown


def test_response_keeps_openai_shape_and_extension() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "served-direct",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    tokens = TokenBreakdown(
        stages={"api": TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)},
        total=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    response = build_response(
        request,
        mode="direct",
        final_text="answer",
        intermediate={"api": "answer"},
        tokens=tokens,
        response_id="chatcmpl-fixed",
        created=1700000000,
    )
    assert response["id"] == "chatcmpl-fixed"
    assert response["created"] == 1700000000
    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"]["content"] == "answer"
    assert response["adapter_critic"]["mode"] == "direct"
    assert "intermediate" in response["adapter_critic"]
    assert "tokens" in response["adapter_critic"]
    assert response["usage"] == response["adapter_critic"]["tokens"]["total"]


def test_response_supports_tool_calls_with_empty_content() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "served-direct",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    tokens = TokenBreakdown(
        stages={"api": TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)},
        total=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    response = build_response(
        request,
        mode="direct",
        final_text="",
        intermediate={"api": ""},
        tokens=tokens,
        response_id="chatcmpl-fixed",
        created=1700000000,
        finish_reason="tool_calls",
        final_tool_calls=[
            {
                "id": "call_cancel",
                "type": "function",
                "function": {"name": "cancel_reservation", "arguments": '{"reservation_id":"EHGLP3"}'},
            }
        ],
    )

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "cancel_reservation"


def test_response_normalizes_empty_tool_calls_to_stop() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "served-direct",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    tokens = TokenBreakdown(
        stages={"api": TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)},
        total=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    response = build_response(
        request,
        mode="direct",
        final_text="plain text",
        intermediate={"api": "plain text"},
        tokens=tokens,
        response_id="chatcmpl-fixed",
        created=1700000000,
        finish_reason="tool_calls",
        final_tool_calls=[],
    )

    choice = response["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]


def test_response_drops_function_call_when_tool_calls_exist() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "served-direct",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    tokens = TokenBreakdown(
        stages={"api": TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)},
        total=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    response = build_response(
        request,
        mode="direct",
        final_text="",
        intermediate={"api": ""},
        tokens=tokens,
        response_id="chatcmpl-fixed",
        created=1700000000,
        finish_reason="tool_calls",
        final_tool_calls=[
            {
                "id": "call_cancel",
                "type": "function",
                "function": {"name": "cancel_reservation", "arguments": '{"reservation_id":"EHGLP3"}'},
            }
        ],
        final_function_call={"response": "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."},
    )

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "cancel_reservation"
    assert "function_call" not in choice["message"]
