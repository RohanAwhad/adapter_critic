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


def test_mode_override_without_secondary_target_falls_back_to_api_target() -> None:
    overrides = AdapterCriticOverrides(mode="adapter")
    runtime = resolve_runtime_config(_config(), "served-direct", overrides)
    assert runtime is not None
    assert runtime.mode == "adapter"
    assert runtime.adapter is not None
    assert runtime.adapter.model == "api-direct"
    assert runtime.adapter.base_url == "https://api.example"


def test_critic_mode_override_without_secondary_target_falls_back_to_api_target() -> None:
    overrides = AdapterCriticOverrides(mode="critic")
    runtime = resolve_runtime_config(_config(), "served-direct", overrides)
    assert runtime is not None
    assert runtime.mode == "critic"
    assert runtime.critic is not None
    assert runtime.critic.model == "api-direct"
    assert runtime.critic.base_url == "https://api.example"


def test_partial_secondary_override_is_rejected() -> None:
    overrides = AdapterCriticOverrides(mode="adapter", adapter_model="adapter-only")
    runtime = resolve_runtime_config(_config(), "served-direct", overrides)
    assert runtime is None


def test_served_model_custom_prompts_are_resolved() -> None:
    config = AppConfig.model_validate(
        {
            "served_models": {
                "served-direct": {
                    "mode": "direct",
                    "api": {"model": "api-direct", "base_url": "https://api.example"},
                    "adapter_system_prompt": "adapter prompt from config",
                    "critic_system_prompt": "critic prompt from config",
                }
            }
        }
    )
    adapter_runtime = resolve_runtime_config(config, "served-direct", AdapterCriticOverrides(mode="adapter"))
    critic_runtime = resolve_runtime_config(config, "served-direct", AdapterCriticOverrides(mode="critic"))

    assert adapter_runtime is not None
    assert critic_runtime is not None
    assert adapter_runtime.adapter_system_prompt == "adapter prompt from config"
    assert critic_runtime.critic_system_prompt == "critic prompt from config"
