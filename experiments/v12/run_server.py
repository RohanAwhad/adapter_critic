from __future__ import annotations

import os

import uvicorn
from loguru import logger

from adapter_critic.app import create_app
from adapter_critic.config import AppConfig
from adapter_critic.http_gateway import OpenAICompatibleHttpGateway
from adapter_critic.logging_setup import configure_logging
from adapter_critic.routing_gateway import RoutingGateway
from adapter_critic.runtime import build_runtime_state
from adapter_critic.vertex_gateway import VertexAICompatibleHttpGateway
from experiments.v12.upstream_resolution import (
    build_openai_base_url,
    build_vertex_base_url,
    resolve_model_name,
    resolve_project_id,
    resolve_region,
    resolve_upstream_host,
)

os.environ.setdefault("LOGGING_LEVEL", "DEBUG")
configure_logging()


INTERVENTIONIST_MODEL = "gpt-oss-120b"
INTERVENTIONIST_PORT = 8101


def build_experiment_config(*, project_id: str, region: str, model_name: str, upstream_host: str) -> AppConfig:
    api_base_url = build_vertex_base_url(project_id=project_id, region=region)
    adapter_base_url = build_openai_base_url(host=upstream_host, port=INTERVENTIONIST_PORT)

    logger.info("selected project_id={}", project_id)
    logger.info("selected region={}", region)
    logger.info("selected model_name={}", model_name)
    logger.info("selected upstream_host={}", upstream_host)
    logger.info("resolved upstream URL for API model={}", api_base_url)
    logger.info("resolved upstream URL for interventionist model={}", adapter_base_url)

    return AppConfig.model_validate(
        {
            "served_models": {
                "served-direct": {
                    "mode": "direct",
                    "api": {"model": model_name, "base_url": api_base_url},
                },
                "served-adapter": {
                    "mode": "adapter",
                    "api": {"model": model_name, "base_url": api_base_url},
                    "adapter": {"model": INTERVENTIONIST_MODEL, "base_url": adapter_base_url},
                },
            }
        }
    )


def main() -> None:
    project_id = resolve_project_id()
    region = resolve_region()
    model_name = resolve_model_name()
    upstream_host = resolve_upstream_host()

    config = build_experiment_config(
        project_id=project_id,
        region=region,
        model_name=model_name,
        upstream_host=upstream_host,
    )
    openai_gateway = OpenAICompatibleHttpGateway(api_key="dummy")
    vertex_gateway = VertexAICompatibleHttpGateway()
    gateway = RoutingGateway(openai_gateway=openai_gateway, vertex_gateway=vertex_gateway)
    state = build_runtime_state(config=config, gateway=gateway)
    app = create_app(config=config, gateway=gateway, state=state)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
