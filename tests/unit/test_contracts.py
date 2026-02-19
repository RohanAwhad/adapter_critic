from __future__ import annotations

import pytest
from pydantic import ValidationError

from adapter_critic.contracts import parse_request_payload


def _base_payload() -> dict[str, object]:
    return {
        "model": "served-direct",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_parse_override_from_extra_body() -> None:
    payload = _base_payload()
    payload["extra_body"] = {
        "x_adapter_critic": {
            "mode": "adapter",
            "adapter_model": "adapter-v1",
            "adapter_base_url": "https://adapter.example",
        }
    }
    parsed = parse_request_payload(payload)
    assert parsed.overrides.mode == "adapter"
    assert parsed.overrides.adapter_model == "adapter-v1"


def test_parse_override_from_top_level() -> None:
    payload = _base_payload()
    payload["x_adapter_critic"] = {"mode": "critic"}
    parsed = parse_request_payload(payload)
    assert parsed.overrides.mode == "critic"


def test_top_level_override_has_precedence_over_extra_body() -> None:
    payload = _base_payload()
    payload["x_adapter_critic"] = {"mode": "critic", "critic_model": "critic-top"}
    payload["extra_body"] = {
        "x_adapter_critic": {
            "mode": "adapter",
            "adapter_model": "adapter-extra",
            "adapter_base_url": "https://adapter.example",
        }
    }
    parsed = parse_request_payload(payload)
    assert parsed.overrides.mode == "critic"
    assert parsed.overrides.critic_model == "critic-top"
    assert parsed.overrides.adapter_model is None


def test_unknown_override_field_is_rejected() -> None:
    payload = _base_payload()
    payload["x_adapter_critic"] = {"unknown": "x"}
    with pytest.raises(ValidationError):
        parse_request_payload(payload)
