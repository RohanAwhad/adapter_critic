"""
v10: served-adapter with its_hub best-of-n + llm-judge as adapter endpoint.

Starts two servers:
  - its_hub IaaS on port 8108 (best-of-n with llm-judge, hardcoded config)
  - adapter_critic on port 8000 (served-adapter wired to IaaS for adapter stage)

Requires: localhost:8100 running qwen3-30b-a3b-thinking (vLLM).
"""

import os
import threading
import time
from pathlib import Path

import httpx
import uvicorn
from its_hub.integration.iaas import app as iaas_app
from loguru import logger

from adapter_critic.app import create_app
from adapter_critic.config import AppConfig
from adapter_critic.http_gateway import OpenAICompatibleHttpGateway
from adapter_critic.logging_setup import configure_logging
from adapter_critic.runtime import build_runtime_state
from experiments.v10.upstream_resolution import (
    build_upstream_base_url,
    resolve_upstream_host,
)

os.environ.setdefault("LOGGING_LEVEL", "DEBUG")
configure_logging()

ADAPTER_PROMPT = Path(__file__).resolve().with_name("adapter_system_prompt.txt").read_text().strip()

# ITS Hub IaaS config
IAAS_HOST = "127.0.0.1"
IAAS_PORT = 8108
IAAS_BASE = f"http://{IAAS_HOST}:{IAAS_PORT}"

# Adapter critic config
ADAPTER_CRITIC_HOST = "0.0.0.0"
ADAPTER_CRITIC_PORT = 8000

MODEL_NAME = "qwen3-30b-a3b-thinking"


def start_iaas_server() -> tuple[uvicorn.Server, threading.Thread]:
    cfg = uvicorn.Config(iaas_app, host=IAAS_HOST, port=IAAS_PORT, log_level="warning")
    server = uvicorn.Server(cfg)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


def wait_ready(url: str, timeout_s: float = 30.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code < 500:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"service did not become ready: {url}")


def configure_iaas(model_base_url: str) -> None:
    """Configure its_hub IaaS with best-of-n + llm-judge."""
    its_cfg = {
        "provider": "openai",
        "endpoint": model_base_url,
        "api_key": "dummy",
        "model": MODEL_NAME,
        "alg": "best-of-n",
        "rm_name": "llm-judge",
        "judge_model": f"openai/{MODEL_NAME}",
        "judge_base_url": model_base_url,
        "judge_mode": "groupwise",
        "judge_criterion": "overall_quality",
        "judge_api_key": "dummy",
        "judge_temperature": 0.0,
        "judge_max_tokens": 4096,
    }
    r = httpx.post(f"{IAAS_BASE}/configure", json=its_cfg, timeout=60.0)
    if r.status_code != 200:
        raise RuntimeError(f"its_hub /configure failed: {r.status_code} {r.text}")
    logger.info("its_hub configured: best-of-n + llm-judge on {}", model_base_url)


def build_experiment_config(upstream_host: str) -> AppConfig:
    model_base_url = build_upstream_base_url(host=upstream_host, port=8100)

    logger.info("selected upstream host={}", upstream_host)
    logger.info("resolved upstream URL for model={}", model_base_url)
    logger.info("its_hub IaaS adapter endpoint={}/v1", IAAS_BASE)

    return AppConfig.model_validate(
        {
            "served_models": {
                "served-direct": {
                    "mode": "direct",
                    "api": {"model": MODEL_NAME, "base_url": model_base_url},
                },
                "served-adapter": {
                    "mode": "adapter",
                    "api": {"model": MODEL_NAME, "base_url": model_base_url},
                    "adapter": {"model": MODEL_NAME, "base_url": f"{IAAS_BASE}/v1"},
                    "adapter_system_prompt": ADAPTER_PROMPT,
                },
            }
        }
    )


def main() -> None:
    upstream_host = resolve_upstream_host()
    model_base_url = build_upstream_base_url(host=upstream_host, port=8100)

    # 1. Start its_hub IaaS
    logger.info("starting its_hub IaaS on {}:{}", IAAS_HOST, IAAS_PORT)
    iaas_server, iaas_thread = start_iaas_server()
    wait_ready(f"{IAAS_BASE}/docs")
    logger.info("its_hub IaaS ready")

    # 2. Configure IaaS
    configure_iaas(model_base_url)

    # 3. Start adapter_critic
    config = build_experiment_config(upstream_host=upstream_host)
    gateway = OpenAICompatibleHttpGateway(api_key="dummy")
    state = build_runtime_state(config=config, gateway=gateway)
    app = create_app(config=config, gateway=gateway, state=state)

    logger.info("starting adapter_critic on {}:{}", ADAPTER_CRITIC_HOST, ADAPTER_CRITIC_PORT)
    uvicorn.run(app, host=ADAPTER_CRITIC_HOST, port=ADAPTER_CRITIC_PORT)


if __name__ == "__main__":
    main()
