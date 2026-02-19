from __future__ import annotations

import pytest

from adapter_critic.config import AppConfig


@pytest.fixture
def base_config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "served_models": {
                "served-direct": {
                    "mode": "direct",
                    "api": {"model": "api-model", "base_url": "https://api.example"},
                },
                "served-adapter": {
                    "mode": "adapter",
                    "api": {"model": "api-model", "base_url": "https://api.example"},
                    "adapter": {"model": "adapter-model", "base_url": "https://adapter.example"},
                },
                "served-critic": {
                    "mode": "critic",
                    "api": {"model": "api-model", "base_url": "https://api.example"},
                    "critic": {"model": "critic-model", "base_url": "https://critic.example"},
                },
            }
        }
    )
