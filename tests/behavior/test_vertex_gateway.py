from __future__ import annotations

import subprocess
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request

from adapter_critic.contracts import ChatMessage
from adapter_critic.vertex_gateway import VertexAICompatibleHttpGateway


def _patch_async_client(monkeypatch: pytest.MonkeyPatch, app: FastAPI) -> None:
    transport = httpx.ASGITransport(app=app)
    original_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)


@pytest.mark.anyio
async def test_vertex_gateway_uses_gcloud_token_and_maps_text_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = FastAPI()

    @upstream.post("/{path:path}")
    async def raw_predict(path: str, request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        assert path == (
            "v1/projects/test-project/locations/us-east5/publishers/anthropic/models/"
            "claude-sonnet-4-5@20250929:rawPredict"
        )
        assert request.headers.get("authorization") == "Bearer gcloud-token"
        assert payload["anthropic_version"] == "vertex-2023-10-16"
        assert payload["system"] == "system instructions"
        assert payload["messages"] == [{"role": "user", "content": "ping"}]
        assert payload["max_tokens"] == 32
        assert payload["temperature"] == 0
        return {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "pong"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 11, "output_tokens": 3},
        }

    _patch_async_client(monkeypatch, upstream)

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert command == ["gcloud", "auth", "print-access-token"]
        assert check is True
        assert capture_output is True
        assert text is True
        return subprocess.CompletedProcess(command, 0, stdout="gcloud-token\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    gateway = VertexAICompatibleHttpGateway(timeout_seconds=5.0)
    result = await gateway.complete(
        model="claude-sonnet-4-5@20250929",
        base_url="http://testserver/v1/projects/test-project/locations/us-east5",
        messages=[
            ChatMessage(role="system", content="system instructions"),
            ChatMessage(role="user", content="ping"),
        ],
        request_options={"max_tokens": 32, "temperature": 0},
    )

    assert result.content == "pong"
    assert result.finish_reason == "stop"
    assert result.usage.prompt_tokens == 11
    assert result.usage.completion_tokens == 3
    assert result.usage.total_tokens == 14
    assert result.tool_calls is None


@pytest.mark.anyio
async def test_vertex_gateway_maps_tool_use_to_openai_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = FastAPI()

    @upstream.post("/{path:path}")
    async def raw_predict(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert path == (
            "v1/projects/test-project/locations/us-east5/publishers/anthropic/models/"
            "claude-sonnet-4-5@20250929:rawPredict"
        )
        assert payload["messages"] == [{"role": "user", "content": "cancel reservation EHGLP3"}]
        assert payload["max_tokens"] == 64
        return {
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "cancel_reservation",
                    "input": {"reservation_id": "EHGLP3"},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 4},
        }

    _patch_async_client(monkeypatch, upstream)

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text
        return subprocess.CompletedProcess(command, 0, stdout="gcloud-token\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    gateway = VertexAICompatibleHttpGateway(timeout_seconds=5.0)
    result = await gateway.complete(
        model="anthropic/claude-sonnet-4-5@20250929",
        base_url=(
            "http://testserver/v1/projects/test-project/locations/us-east5/"
            "publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict"
        ),
        messages=[ChatMessage(role="user", content="cancel reservation EHGLP3")],
        request_options={"max_tokens": 64},
    )

    assert result.content == ""
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls is not None
    assert result.tool_calls[0]["function"]["name"] == "cancel_reservation"
    assert result.tool_calls[0]["function"]["arguments"] == '{"reservation_id":"EHGLP3"}'
