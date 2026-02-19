from __future__ import annotations

import httpx
import pytest
from openai import AsyncOpenAI

from adapter_critic.app import create_app
from adapter_critic.config import AppConfig
from adapter_critic.upstream import UpstreamResult
from tests.helpers import FakeGateway, usage


@pytest.mark.anyio
async def test_openai_sdk_smoke_with_extension(base_config: AppConfig) -> None:
    gateway = FakeGateway([UpstreamResult(content="ok", usage=usage(1, 1, 2))])
    app = create_app(config=base_config, gateway=gateway)
    transport = httpx.ASGITransport(app=app)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    client = AsyncOpenAI(api_key="test", base_url="http://testserver/v1", http_client=http_client)

    response = await client.chat.completions.create(
        model="served-direct",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"x_adapter_critic": {"mode": "direct"}},
    )
    await client.close()

    adapter_critic = response.model_extra.get("adapter_critic") if response.model_extra is not None else None
    assert adapter_critic is not None
    assert adapter_critic["mode"] == "direct"
    assert "tokens" in adapter_critic
