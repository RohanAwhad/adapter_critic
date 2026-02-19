from __future__ import annotations

from adapter_critic.config import AppConfig, resolve_runtime_config
from adapter_critic.contracts import AdapterCriticOverrides


def _config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "served_models": {
                "served-adapter": {
                    "mode": "adapter",
                    "api": {"model": "api-default", "base_url": "https://api.example"},
                    "adapter": {"model": "adapter-default", "base_url": "https://adapter.example"},
                },
                "served-direct": {
                    "mode": "direct",
                    "api": {"model": "api-direct", "base_url": "https://api.example"},
                },
            }
        }
    )


def test_served_model_resolves_defaults() -> None:
    runtime = resolve_runtime_config(_config(), "served-adapter", AdapterCriticOverrides())
    assert runtime is not None
    assert runtime.mode == "adapter"
    assert runtime.api.model == "api-default"
    assert runtime.adapter is not None
    assert runtime.adapter.model == "adapter-default"


def test_request_override_has_precedence() -> None:
    overrides = AdapterCriticOverrides(
        mode="adapter",
        adapter_model="adapter-override",
        adapter_base_url="https://override.example",
    )
    runtime = resolve_runtime_config(_config(), "served-adapter", overrides)
    assert runtime is not None
    assert runtime.adapter is not None
    assert runtime.adapter.model == "adapter-override"


def test_invalid_mode_resolution_returns_none() -> None:
    overrides = AdapterCriticOverrides(mode="adapter")
    runtime = resolve_runtime_config(_config(), "served-direct", overrides)
    assert runtime is None
