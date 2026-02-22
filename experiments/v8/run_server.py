import os
from pathlib import Path

import uvicorn
from loguru import logger

from adapter_critic.app import create_app
from adapter_critic.config import AppConfig
from adapter_critic.http_gateway import OpenAICompatibleHttpGateway
from adapter_critic.logging_setup import configure_logging
from adapter_critic.runtime import build_runtime_state
from experiments.v8.upstream_resolution import (
    build_upstream_base_url,
    resolve_upstream_host,
)

os.environ.setdefault("LOGGING_LEVEL", "DEBUG")
configure_logging()

CRITIC_PROMPT = Path(__file__).resolve().with_name("critic_system_prompt.txt").read_text().strip()


OPENAI_BASE_URL = "https://api.openai.com/v1"


def build_experiment_config(upstream_host: str) -> AppConfig:
    adapter_base_url = build_upstream_base_url(host=upstream_host, port=8100)

    logger.info("selected upstream host={}", upstream_host)
    logger.info("API model base_url={}", OPENAI_BASE_URL)
    logger.info("resolved upstream URL for adapter/critic model={}", adapter_base_url)

    return AppConfig.model_validate(
        {
            "served_models": {
                "served-direct": {
                    "mode": "direct",
                    "api": {"model": "gpt-4.1", "base_url": OPENAI_BASE_URL},
                },
                "served-adapter": {
                    "mode": "adapter",
                    "api": {"model": "gpt-4.1", "base_url": OPENAI_BASE_URL},
                    "adapter": {"model": "glm-4.7-flash", "base_url": adapter_base_url},
                },
                "served-critic": {
                    "mode": "critic",
                    "api": {"model": "gpt-4.1", "base_url": OPENAI_BASE_URL},
                    "critic": {"model": "glm-4.7-flash", "base_url": adapter_base_url},
                    "critic_system_prompt": CRITIC_PROMPT,
                },
            }
        }
    )


def main() -> None:
    upstream_host = resolve_upstream_host()

    config = build_experiment_config(upstream_host=upstream_host)
    gateway = OpenAICompatibleHttpGateway(api_key=os.environ["OPENAI_API_KEY"])
    state = build_runtime_state(config=config, gateway=gateway)
    app = create_app(config=config, gateway=gateway, state=state)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
