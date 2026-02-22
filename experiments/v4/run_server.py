import os
from pathlib import Path

import uvicorn
from loguru import logger

from adapter_critic.app import create_app
from adapter_critic.config import AppConfig
from adapter_critic.http_gateway import OpenAICompatibleHttpGateway
from adapter_critic.logging_setup import configure_logging
from adapter_critic.runtime import build_runtime_state
from experiments.v4.upstream_resolution import (
    build_upstream_base_url,
    resolve_upstream_host,
)

os.environ.setdefault("LOGGING_LEVEL", "DEBUG")
configure_logging()

CRITIC_PROMPT = Path(__file__).resolve().with_name("critic_system_prompt.txt").read_text().strip()


def build_experiment_config(upstream_host: str) -> AppConfig:
    adapter_base_url = build_upstream_base_url(host=upstream_host, port=8100)
    api_base_url = build_upstream_base_url(host=upstream_host, port=8101)

    logger.info("selected upstream host={}", upstream_host)
    logger.info("resolved upstream URL for API model={}", api_base_url)
    logger.info("resolved upstream URL for adapter/critic model={}", adapter_base_url)

    return AppConfig.model_validate(
        {
            "served_models": {
                "served-direct": {
                    "mode": "direct",
                    "api": {"model": "gpt-oss-120b", "base_url": api_base_url},
                },
                "served-adapter": {
                    "mode": "adapter",
                    "api": {"model": "gpt-oss-120b", "base_url": api_base_url},
                    "adapter": {"model": "glm-4.7-flash", "base_url": adapter_base_url},
                },
                "served-critic": {
                    "mode": "critic",
                    "api": {"model": "gpt-oss-120b", "base_url": api_base_url},
                    "critic": {"model": "glm-4.7-flash", "base_url": adapter_base_url},
                    "critic_system_prompt": CRITIC_PROMPT,
                },
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
