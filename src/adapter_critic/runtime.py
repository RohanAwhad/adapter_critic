from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from .config import AppConfig
from .upstream import UpstreamGateway


@dataclass(frozen=True)
class RuntimeState:
    config: AppConfig
    gateway: UpstreamGateway
    id_provider: Callable[[], str]
    time_provider: Callable[[], int]


def default_id_provider() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def default_time_provider() -> int:
    return int(time.time())


def build_runtime_state(
    config: AppConfig,
    gateway: UpstreamGateway,
    id_provider: Callable[[], str] = default_id_provider,
    time_provider: Callable[[], int] = default_time_provider,
) -> RuntimeState:
    return RuntimeState(
        config=config,
        gateway=gateway,
        id_provider=id_provider,
        time_provider=time_provider,
    )
