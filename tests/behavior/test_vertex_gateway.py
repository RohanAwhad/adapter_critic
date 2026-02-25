from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pytest

from adapter_critic.contracts import ChatMessage
from adapter_critic.vertex_gateway import VertexAICompatibleHttpGateway


@dataclass(frozen=True)
class LiveVertexConfig:
    project_id: str
    region: str
    model: str


def _project_id_from_env() -> str:
    project_id = (os.getenv("ANTHROPIC_VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
    assert project_id != "", "missing project env var: set ANTHROPIC_VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT"
    return project_id


def _live_vertex_config() -> LiveVertexConfig:
    return LiveVertexConfig(
        project_id=_project_id_from_env(),
        region=(os.getenv("CLOUD_ML_REGION") or os.getenv("VERTEX_LOCATION") or "us-east5").strip(),
        model=(os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-5@20250929").strip(),
    )


def _vertex_base_url(config: LiveVertexConfig) -> str:
    return (
        f"https://{config.region}-aiplatform.googleapis.com/v1/projects/{config.project_id}/locations/{config.region}"
    )


@pytest.mark.anyio
async def test_vertex_gateway_live_text_response() -> None:
    config = _live_vertex_config()
    gateway = VertexAICompatibleHttpGateway(timeout_seconds=5.0)
    result = await gateway.complete(
        model=config.model,
        base_url=_vertex_base_url(config),
        messages=[
            ChatMessage(role="system", content="Reply with exactly the word PONG."),
            ChatMessage(role="user", content="Ping"),
        ],
        request_options={"max_tokens": 32, "temperature": 0},
    )

    assert "pong" in result.content.lower()
    assert result.finish_reason in {"stop", "length"}
    assert result.usage.total_tokens > 0
    assert result.tool_calls is None


@pytest.mark.anyio
async def test_vertex_gateway_live_tool_use_mapping() -> None:
    config = _live_vertex_config()
    gateway = VertexAICompatibleHttpGateway(timeout_seconds=5.0)
    result = await gateway.complete(
        model=config.model,
        base_url=_vertex_base_url(config),
        messages=[ChatMessage(role="user", content="cancel reservation EHGLP3")],
        request_options={
            "max_tokens": 128,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "cancel_reservation",
                        "description": "Cancel reservation by id",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "reservation_id": {"type": "string"},
                            },
                            "required": ["reservation_id"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    )

    assert result.finish_reason == "tool_calls"
    assert result.tool_calls is not None
    first_tool_call = result.tool_calls[0]
    assert first_tool_call["function"]["name"] == "cancel_reservation"
    parsed_arguments = json.loads(first_tool_call["function"]["arguments"])
    assert parsed_arguments["reservation_id"] == "EHGLP3"
