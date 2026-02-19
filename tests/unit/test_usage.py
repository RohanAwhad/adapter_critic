from __future__ import annotations

from adapter_critic.upstream import TokenUsage
from adapter_critic.usage import aggregate_usage


def test_aggregate_usage_is_deterministic() -> None:
    stages = {
        "api": TokenUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
        "adapter": TokenUsage(prompt_tokens=2, completion_tokens=2, total_tokens=4),
    }
    result = aggregate_usage(stages)
    assert result.total.prompt_tokens == 5
    assert result.total.completion_tokens == 6
    assert result.total.total_tokens == 11


def test_aggregate_usage_clamps_negative_values() -> None:
    stages = {"api": TokenUsage(prompt_tokens=-5, completion_tokens=2, total_tokens=-1)}
    result = aggregate_usage(stages)
    assert result.total.prompt_tokens == 0
    assert result.total.completion_tokens == 2
    assert result.total.total_tokens == 0
