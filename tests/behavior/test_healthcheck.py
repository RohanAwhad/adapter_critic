from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from adapter_critic.app import create_app
from adapter_critic.config import AppConfig
from tests.helpers import FakeGateway


def _health_config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "served_models": {
                "served-direct": {
                    "mode": "direct",
                    "api": {"model": "api-model", "base_url": "http://testserver/v1"},
                },
                "served-adapter": {
                    "mode": "adapter",
                    "api": {"model": "api-model", "base_url": "http://testserver/v1"},
                    "adapter": {"model": "adapter-model", "base_url": "http://testserver/v1"},
                },
                "served-critic": {
                    "mode": "critic",
                    "api": {"model": "api-model", "base_url": "http://testserver/v1"},
                    "critic": {"model": "critic-model", "base_url": "http://testserver/v1"},
                },
            }
        }
    )


@pytest.mark.anyio
async def test_healthz_returns_200_when_all_targets_available(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = FastAPI()

    @upstream.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": "api-model"},
                {"id": "adapter-model"},
                {"id": "critic-model"},
            ],
        }

    transport = httpx.ASGITransport(app=upstream)
    original_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

    app = create_app(config=_health_config(), gateway=FakeGateway([]))
    client = TestClient(app)

    response = client.get("/healthz")
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["checked"] == 3
    assert payload["healthy"] == 3
    assert all(target["ok"] is True for target in payload["targets"])


@pytest.mark.anyio
async def test_healthz_returns_503_when_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = FastAPI()

    @upstream.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": "api-model"},
                {"id": "adapter-model"},
            ],
        }

    transport = httpx.ASGITransport(app=upstream)
    original_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

    app = create_app(config=_health_config(), gateway=FakeGateway([]))
    client = TestClient(app)

    response = client.get("/healthz")
    payload = response.json()

    assert response.status_code == 503
    assert payload["status"] == "degraded"
    assert payload["checked"] == 3
    assert payload["healthy"] == 2
    assert any(target["ok"] is False for target in payload["targets"])


@pytest.mark.anyio
async def test_healthz_dedupes_shared_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    request_count = {"models": 0}
    upstream = FastAPI()

    @upstream.get("/v1/models")
    async def models() -> dict[str, Any]:
        request_count["models"] += 1
        return {
            "object": "list",
            "data": [
                {"id": "api-model"},
                {"id": "adapter-model"},
                {"id": "critic-model"},
            ],
        }

    transport = httpx.ASGITransport(app=upstream)
    original_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

    app = create_app(config=_health_config(), gateway=FakeGateway([]))
    client = TestClient(app)

    response = client.get("/healthz")
    payload = response.json()

    assert response.status_code == 200
    assert payload["checked"] == 3
    assert request_count["models"] == 3


@pytest.mark.anyio
async def test_healthz_uses_stage_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = FastAPI()

    @upstream.get("/v1/models")
    async def models(request: Request) -> dict[str, Any]:
        assert request.headers.get("authorization") == "Bearer health-secret"
        return {
            "object": "list",
            "data": [{"id": "api-model"}],
        }

    transport = httpx.ASGITransport(app=upstream)
    original_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)
    monkeypatch.setenv("HEALTH_API_KEY", "health-secret")

    config = AppConfig.model_validate(
        {
            "served_models": {
                "served-direct": {
                    "mode": "direct",
                    "api": {
                        "model": "api-model",
                        "base_url": "http://testserver/v1",
                        "api_key_env": "HEALTH_API_KEY",
                    },
                }
            }
        }
    )

    app = create_app(config=config, gateway=FakeGateway([]))
    client = TestClient(app)

    response = client.get("/healthz")
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["checked"] == 1
    assert payload["healthy"] == 1
