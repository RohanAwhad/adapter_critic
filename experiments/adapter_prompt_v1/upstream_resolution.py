from __future__ import annotations

import os
from collections.abc import Mapping

DEFAULT_UPSTREAM_HOST = "localhost"
UPSTREAM_HOST_ENV_VAR = "UPSTREAM_HOST"


def build_upstream_base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1"


def resolve_upstream_host(
    env: Mapping[str, str] | None = None,
    env_var: str = UPSTREAM_HOST_ENV_VAR,
    default_host: str = DEFAULT_UPSTREAM_HOST,
) -> str:
    env_source = env if env is not None else os.environ
    host = env_source.get(env_var, default_host).strip()
    return host or default_host
