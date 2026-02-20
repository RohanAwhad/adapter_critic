from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import AppConfig, StageTarget

DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"


@dataclass(frozen=True)
class HealthTarget:
    model: str
    base_url: str
    api_key_env: str | None
    used_by: tuple[str, ...] = field(default_factory=tuple)


def _target_key(target: StageTarget) -> tuple[str, str, str | None]:
    return (target.base_url.rstrip("/"), target.model, target.api_key_env)


def collect_health_targets(config: AppConfig) -> list[HealthTarget]:
    by_key: dict[tuple[str, str, str | None], dict[str, Any]] = {}

    for served_model, served in config.served_models.items():
        for stage_name in ("api", "adapter", "critic"):
            stage = getattr(served, stage_name)
            if stage is None:
                continue

            key = _target_key(stage)
            entry = by_key.get(key)
            used_by = f"{served_model}.{stage_name}"
            if entry is None:
                by_key[key] = {
                    "model": stage.model,
                    "base_url": stage.base_url.rstrip("/"),
                    "api_key_env": stage.api_key_env,
                    "used_by": [used_by],
                }
            else:
                entry["used_by"].append(used_by)

    targets: list[HealthTarget] = []
    for item in by_key.values():
        targets.append(
            HealthTarget(
                model=item["model"],
                base_url=item["base_url"],
                api_key_env=item["api_key_env"],
                used_by=tuple(sorted(item["used_by"])),
            )
        )
    return targets


def _resolve_api_key(api_key_env: str | None) -> str | None:
    if api_key_env is not None and api_key_env != "":
        key: str = api_key_env
        return os.environ.get(key)
    return os.environ.get(DEFAULT_API_KEY_ENV)


async def _check_target(target: HealthTarget, timeout_seconds: float) -> dict[str, Any]:
    started = time.perf_counter()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = _resolve_api_key(target.api_key_env)
    if api_key not in {None, ""}:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.get(
            f"{target.base_url}/models",
            headers=headers,
        )

    duration_ms = int((time.perf_counter() - started) * 1000)
    if response.status_code < 200 or response.status_code >= 300:
        return {
            "model": target.model,
            "base_url": target.base_url,
            "api_key_env": target.api_key_env,
            "used_by": list(target.used_by),
            "ok": False,
            "status_code": response.status_code,
            "error": f"/models returned status {response.status_code}",
            "duration_ms": duration_ms,
        }

    payload = response.json()
    if not isinstance(payload, dict):
        return {
            "model": target.model,
            "base_url": target.base_url,
            "api_key_env": target.api_key_env,
            "used_by": list(target.used_by),
            "ok": False,
            "status_code": response.status_code,
            "error": "/models response is not a JSON object",
            "duration_ms": duration_ms,
        }

    data = payload.get("data")
    if not isinstance(data, list):
        return {
            "model": target.model,
            "base_url": target.base_url,
            "api_key_env": target.api_key_env,
            "used_by": list(target.used_by),
            "ok": False,
            "status_code": response.status_code,
            "error": "/models response missing data list",
            "duration_ms": duration_ms,
        }

    model_found = any(
        isinstance(item, dict) and (item.get("id") == target.model or item.get("root") == target.model) for item in data
    )
    if not model_found:
        return {
            "model": target.model,
            "base_url": target.base_url,
            "api_key_env": target.api_key_env,
            "used_by": list(target.used_by),
            "ok": False,
            "status_code": response.status_code,
            "error": "configured model not found in /models",
            "duration_ms": duration_ms,
        }

    return {
        "model": target.model,
        "base_url": target.base_url,
        "api_key_env": target.api_key_env,
        "used_by": list(target.used_by),
        "ok": True,
        "status_code": response.status_code,
        "duration_ms": duration_ms,
    }


async def run_healthcheck(config: AppConfig, *, timeout_seconds: float = 5.0) -> dict[str, Any]:
    started = time.perf_counter()
    targets = collect_health_targets(config)
    raw_results = await asyncio.gather(
        *[_check_target(target, timeout_seconds=timeout_seconds) for target in targets],
        return_exceptions=True,
    )

    results: list[dict[str, Any]] = []
    for target, raw in zip(targets, raw_results, strict=False):
        if isinstance(raw, BaseException):
            results.append(
                {
                    "model": target.model,
                    "base_url": target.base_url,
                    "api_key_env": target.api_key_env,
                    "used_by": list(target.used_by),
                    "ok": False,
                    "status_code": 0,
                    "error": f"health probe failed: {type(raw).__name__}: {raw}",
                    "duration_ms": 0,
                }
            )
        else:
            results.append(raw)

    healthy_count = sum(1 for item in results if item.get("ok") is True)
    total_count = len(results)
    status = "ok" if healthy_count == total_count else "degraded"

    return {
        "status": status,
        "checked": total_count,
        "healthy": healthy_count,
        "duration_ms": int((time.perf_counter() - started) * 1000),
        "targets": results,
    }
