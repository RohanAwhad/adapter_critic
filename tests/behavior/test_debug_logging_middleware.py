from __future__ import annotations

from typing import Any

import pytest
from loguru import logger

from adapter_critic.config import AppConfig
from adapter_critic.upstream import UpstreamResult
from tests.helpers import build_client, usage


def test_debug_middleware_logs_request_and_response(base_config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGGING_LEVEL", "DEBUG")
    records: list[str] = []

    def capture(message: Any) -> None:
        records.append(message.record["message"])

    sink_id = logger.add(capture, level="DEBUG")
    try:
        client, _ = build_client(
            base_config,
            [UpstreamResult(content="direct-answer", usage=usage(2, 3, 5))],
        )
        response = client.post(
            "/v1/chat/completions",
            json={"model": "served-direct", "messages": [{"role": "user", "content": "hello"}]},
        )
    finally:
        logger.remove(sink_id)
        monkeypatch.delenv("LOGGING_LEVEL", raising=False)

    assert response.status_code == 200
    assert any(
        "incoming request method=POST path=/v1/chat/completions" in record and '"model":"served-direct"' in record
        for record in records
    )
    assert any(
        "outgoing response method=POST path=/v1/chat/completions status_code=200" in record
        and '"object":"chat.completion"' in record
        for record in records
    )
