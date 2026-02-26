from __future__ import annotations

import os
from collections.abc import Mapping

DEFAULT_UPSTREAM_HOST = "localhost"
UPSTREAM_HOST_ENV_VAR = "UPSTREAM_HOST"


def resolve_project_id() -> str:
    project_id = (os.getenv("ANTHROPIC_VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
    if project_id == "":
        raise ValueError("missing project env var: set ANTHROPIC_VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT")
    return project_id


def resolve_region() -> str:
    region = (os.getenv("CLOUD_ML_REGION") or os.getenv("VERTEX_LOCATION") or "us-east5").strip()
    if region == "":
        raise ValueError("vertex region must not be empty")
    return region


def resolve_model_name() -> str:
    model_name = (os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-5@20250929").strip()
    if model_name == "":
        raise ValueError("vertex model must not be empty")
    return model_name


def build_vertex_base_url(*, project_id: str, region: str) -> str:
    return f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}"


def build_openai_base_url(*, host: str, port: int) -> str:
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
