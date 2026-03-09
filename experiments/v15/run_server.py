from __future__ import annotations

import os

import uvicorn
from loguru import logger

from adapter_critic.app import create_app
from adapter_critic.config import AppConfig
from adapter_critic.http_gateway import OpenAICompatibleHttpGateway
from adapter_critic.logging_setup import configure_logging
from adapter_critic.runtime import build_runtime_state
from experiments.v15.upstream_resolution import build_upstream_base_url, resolve_upstream_host

os.environ.setdefault("LOGGING_LEVEL", "DEBUG")
configure_logging()

API_MODEL = "qwen3.5-4b"
API_PORT = 8100
ADAPTER_MODEL = "qwen3.5-0.8b"
ADAPTER_PORT = 8102


def build_experiment_config(upstream_host: str) -> AppConfig:
    api_base_url = build_upstream_base_url(host=upstream_host, port=API_PORT)
    adapter_base_url = build_upstream_base_url(host=upstream_host, port=ADAPTER_PORT)

    logger.info("selected upstream host={}", upstream_host)
    logger.info("resolved upstream URL for API model={}", api_base_url)
    logger.info("resolved upstream URL for interventionist model={}", adapter_base_url)

    return AppConfig.model_validate(
        {
            "served_models": {
                "served-adapter": {
                    "mode": "adapter",
                    "api": {"model": API_MODEL, "base_url": api_base_url},
                    "adapter": {"model": ADAPTER_MODEL, "base_url": adapter_base_url},
                }
            }
        }
    )


def main() -> None:
    upstream_host = resolve_upstream_host()
    config = build_experiment_config(upstream_host=upstream_host)
    gateway = OpenAICompatibleHttpGateway(api_key="dummy")
    state = build_runtime_state(config=config, gateway=gateway)
    app = create_app(config=config, gateway=gateway, state=state)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
