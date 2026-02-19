import os
from pathlib import Path

import uvicorn

from adapter_critic.app import create_app
from adapter_critic.config import AppConfig
from adapter_critic.http_gateway import OpenAICompatibleHttpGateway
from adapter_critic.logging_setup import configure_logging
from adapter_critic.runtime import build_runtime_state

os.environ.setdefault("LOGGING_LEVEL", "DEBUG")
configure_logging()

ADAPTER_PROMPT = Path(__file__).resolve().with_name("adapter_system_prompt.txt").read_text().strip()

config = AppConfig.model_validate(
    {
        "served_models": {
            "served-direct": {
                "mode": "direct",
                "api": {"model": "gpt-oss-120b", "base_url": "http://localhost:8101/v1"},
            },
            "served-adapter": {
                "mode": "adapter",
                "api": {"model": "gpt-oss-120b", "base_url": "http://localhost:8101/v1"},
                "adapter": {"model": "gpt-oss-20b", "base_url": "http://localhost:8100/v1"},
                "adapter_system_prompt": ADAPTER_PROMPT,
            },
            "served-critic": {
                "mode": "critic",
                "api": {"model": "gpt-oss-120b", "base_url": "http://localhost:8101/v1"},
                "critic": {"model": "gpt-oss-20b", "base_url": "http://localhost:8100/v1"},
            },
        }
    }
)

gateway = OpenAICompatibleHttpGateway(api_key="dummy")
state = build_runtime_state(config=config, gateway=gateway)
app = create_app(config=config, gateway=gateway, state=state)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
