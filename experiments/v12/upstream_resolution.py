from __future__ import annotations

import os


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
