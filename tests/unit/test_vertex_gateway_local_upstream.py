from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import uvicorn
from anthropic.lib.vertex._client import AsyncAnthropicVertex as SDKAsyncAnthropicVertex
from fastapi import FastAPI, Request

from adapter_critic.contracts import ChatMessage
from adapter_critic.vertex_gateway import VertexAICompatibleHttpGateway


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_server(app: FastAPI) -> tuple[uvicorn.Server, threading.Thread, str]:
    port = _reserve_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 5.0
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started is True
    return server, thread, f"http://127.0.0.1:{port}"


def _build_upstream_app(*, capture: dict[str, Any], response_payload: dict[str, Any]) -> FastAPI:
    app = FastAPI()

    @app.post("/{path:path}")
    async def raw_predict(path: str, request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        capture["path"] = path
        capture["authorization"] = request.headers.get("authorization")
        capture["payload"] = payload
        return response_payload

    return app


@pytest.fixture
def upstream_server_factory() -> Iterator[Callable[[dict[str, Any], dict[str, Any]], str]]:
    running_servers: list[tuple[uvicorn.Server, threading.Thread]] = []

    def start(response_payload: dict[str, Any], capture: dict[str, Any]) -> str:
        app = _build_upstream_app(capture=capture, response_payload=response_payload)
        server, thread, base_url = _start_server(app)
        running_servers.append((server, thread))
        return base_url

    yield start

    for server, thread in running_servers:
        server.should_exit = True
        thread.join(timeout=3)


@pytest.fixture
def fake_vertex_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_ensure_access_token(self: SDKAsyncAnthropicVertex) -> str:
        del self
        return "unit-token"

    monkeypatch.setattr(SDKAsyncAnthropicVertex, "_ensure_access_token", _fake_ensure_access_token)


@pytest.mark.anyio
async def test_vertex_gateway_local_text_response_mapping(
    fake_vertex_auth_token: None,
    upstream_server_factory: Callable[[dict[str, Any], dict[str, Any]], str],
) -> None:
    del fake_vertex_auth_token
    capture: dict[str, Any] = {}
    base_url = upstream_server_factory(
        {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5@20250929",
            "content": [{"type": "text", "text": "pong"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 11, "output_tokens": 3},
        },
        capture,
    )

    gateway = VertexAICompatibleHttpGateway(timeout_seconds=5.0)
    result = await gateway.complete(
        model="claude-sonnet-4-5@20250929",
        base_url=f"{base_url}/v1/projects/test-project/locations/us-east5",
        messages=[
            ChatMessage(role="system", content="system instructions"),
            ChatMessage(role="user", content="ping"),
        ],
        request_options={"max_tokens": 32, "temperature": 0},
    )

    assert capture["path"] == (
        "v1/projects/test-project/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict"
    )
    assert capture["authorization"] == "Bearer unit-token"
    payload = capture["payload"]
    assert payload["anthropic_version"] == "vertex-2023-10-16"
    assert payload["system"] == "system instructions"
    assert payload["messages"] == [{"role": "user", "content": "ping"}]
    assert payload["max_tokens"] == 32
    assert payload["temperature"] == 0

    assert result.content == "pong"
    assert result.finish_reason == "stop"
    assert result.usage.prompt_tokens == 11
    assert result.usage.completion_tokens == 3
    assert result.usage.total_tokens == 14
    assert result.tool_calls is None


@pytest.mark.anyio
async def test_vertex_gateway_local_tool_use_mapping(
    fake_vertex_auth_token: None,
    upstream_server_factory: Callable[[dict[str, Any], dict[str, Any]], str],
) -> None:
    del fake_vertex_auth_token
    capture: dict[str, Any] = {}
    base_url = upstream_server_factory(
        {
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5@20250929",
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
        },
        capture,
    )

    gateway = VertexAICompatibleHttpGateway(timeout_seconds=5.0)
    result = await gateway.complete(
        model="anthropic/claude-sonnet-4-5@20250929",
        base_url=(
            f"{base_url}/v1/projects/test-project/locations/us-east5/"
            "publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict"
        ),
        messages=[ChatMessage(role="user", content="cancel reservation EHGLP3")],
        request_options={"max_tokens": 64},
    )

    payload = capture["payload"]
    assert payload["messages"] == [{"role": "user", "content": "cancel reservation EHGLP3"}]
    assert payload["max_tokens"] == 64

    assert result.content == ""
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls is not None
    assert result.tool_calls[0]["function"]["name"] == "cancel_reservation"
    assert result.tool_calls[0]["function"]["arguments"] == '{"reservation_id":"EHGLP3"}'


@pytest.mark.anyio
async def test_vertex_gateway_accepts_empty_content_list_end_turn(
    fake_vertex_auth_token: None,
    upstream_server_factory: Callable[[dict[str, Any], dict[str, Any]], str],
) -> None:
    del fake_vertex_auth_token
    capture: dict[str, Any] = {}
    base_url = upstream_server_factory(
        {
            "id": "msg_789",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5@20250929",
            "content": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 18, "output_tokens": 2},
        },
        capture,
    )

    gateway = VertexAICompatibleHttpGateway(timeout_seconds=5.0)
    result = await gateway.complete(
        model="claude-sonnet-4-5@20250929",
        base_url=f"{base_url}/v1/projects/test-project/locations/us-east5",
        messages=[ChatMessage(role="user", content="hello")],
        request_options={"max_tokens": 32},
    )

    assert capture["path"] == (
        "v1/projects/test-project/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict"
    )
    assert result.content == ""
    assert result.finish_reason == "stop"
    assert result.tool_calls is None
