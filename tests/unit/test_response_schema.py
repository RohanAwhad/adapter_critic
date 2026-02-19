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
