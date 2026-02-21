from __future__ import annotations

import os
from collections.abc import Mapping

DEFAULT_UPSTREAM_HOST = "localhost"
UPSTREAM_HOST_ENV_VAR = "UPSTREAM_HOST"


def build_upstream_base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1"


def validate_upstream_host(host: str, env_var: str = UPSTREAM_HOST_ENV_VAR) -> None:
    if "://" in host:
        raise ValueError(f"{env_var} must be a bare host without scheme")
    if any(separator in host for separator in ("/", "?", "#")):
        raise ValueError(f"{env_var} must be a bare host without path, query, or fragment")
    if ":" in host:
        raise ValueError(f"{env_var} must not include a port")
    if any(character.isspace() for character in host):
        raise ValueError(f"{env_var} must not contain whitespace")


def resolve_upstream_host(
    env: Mapping[str, str] | None = None,
    env_var: str = UPSTREAM_HOST_ENV_VAR,
    default_host: str = DEFAULT_UPSTREAM_HOST,
) -> str:
    env_source = env if env is not None else os.environ
    host = env_source.get(env_var, default_host).strip()
    resolved_host = host or default_host
    validate_upstream_host(resolved_host, env_var=env_var)
    return resolved_host
