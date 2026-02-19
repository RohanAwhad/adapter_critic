from __future__ import annotations

from typing import Any

from adapter_critic.config import AppConfig
from adapter_critic.contracts import ChatMessage
from adapter_critic.runtime import build_runtime_state
from adapter_critic.upstream import UpstreamResult


class DummyGateway:
    async def complete(
        self,
        *,
        model: str,
        base_url: str,
        messages: list[ChatMessage],
        api_key_env: str | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> UpstreamResult:
        del api_key_env, request_options
        raise NotImplementedError


def _config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "served_models": {
                "served-direct": {
                    "mode": "direct",
                    "api": {"model": "api-model", "base_url": "https://api.example"},
                }
            }
        }
    )


def test_runtime_state_keeps_only_required_long_lived_dependencies() -> None:
    gateway = DummyGateway()
    state = build_runtime_state(
        config=_config(),
        gateway=gateway,
        id_provider=lambda: "chatcmpl-fixed",
        time_provider=lambda: 1700000000,
    )
    assert state.config.served_models["served-direct"].api.model == "api-model"
    assert state.gateway is gateway
    assert state.id_provider() == "chatcmpl-fixed"
    assert state.time_provider() == 1700000000
