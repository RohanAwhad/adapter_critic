from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import uvicorn

from .app import create_app
from .config import AppConfig
from .http_gateway import OpenAICompatibleHttpGateway
from .runtime import build_runtime_state


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adapter_critic server")
    parser.add_argument("--config", type=Path, default=Path("config.json"), help="Path to app config JSON")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable name for API key")
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="Upstream HTTP timeout in seconds")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload")
    return parser.parse_args()


def _load_config(config_path: Path) -> AppConfig:
    config_data = json.loads(config_path.read_text())
    return AppConfig.model_validate(config_data)


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    api_key = os.environ.get(args.api_key_env)
    gateway = OpenAICompatibleHttpGateway(api_key=api_key, timeout_seconds=args.timeout_seconds)
    state = build_runtime_state(config=config, gateway=gateway)
    app = create_app(config=config, gateway=gateway, state=state)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
