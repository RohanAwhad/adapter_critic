from __future__ import annotations

from pydantic import BaseModel

from .upstream import TokenUsage


class TokenBreakdown(BaseModel):
    stages: dict[str, TokenUsage]
    total: TokenUsage


def aggregate_usage(stages: dict[str, TokenUsage]) -> TokenBreakdown:
    prompt_tokens = sum(max(stage.prompt_tokens, 0) for stage in stages.values())
    completion_tokens = sum(max(stage.completion_tokens, 0) for stage in stages.values())
    total_tokens = sum(max(stage.total_tokens, 0) for stage in stages.values())
    total = TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    return TokenBreakdown(stages=stages, total=total)
