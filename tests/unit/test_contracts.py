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
    payload["x_adapter_critic"] = {"mode": "critic", "max_adapter_retries": 1}
    parsed = parse_request_payload(payload)
    assert parsed.overrides.mode == "critic"
    assert parsed.overrides.max_adapter_retries == 1


def test_parse_advisor_override_from_top_level() -> None:
    payload = _base_payload()
    payload["x_adapter_critic"] = {
        "mode": "advisor",
        "advisor_model": "advisor-v1",
        "advisor_base_url": "https://advisor.example",
    }
    parsed = parse_request_payload(payload)
    assert parsed.overrides.mode == "advisor"
    assert parsed.overrides.advisor_model == "advisor-v1"
    assert parsed.overrides.advisor_base_url == "https://advisor.example"


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


def test_negative_max_adapter_retries_is_rejected() -> None:
    payload = _base_payload()
    payload["x_adapter_critic"] = {"mode": "adapter", "max_adapter_retries": -1}
    with pytest.raises(ValidationError):
        parse_request_payload(payload)


def test_tool_related_fields_are_preserved_for_direct_upstream_calls() -> None:
    payload = _base_payload()
    payload["messages"] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_cancel",
                    "type": "function",
                    "function": {
                        "name": "cancel_reservation",
                        "arguments": '{"reservation_id":"EHGLP3"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": "{}", "tool_call_id": "call_cancel"},
    ]
    payload["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "cancel_reservation",
                "parameters": {"type": "object", "properties": {"reservation_id": {"type": "string"}}},
            },
        }
    ]
    payload["tool_choice"] = "auto"
    payload["x_adapter_critic"] = {"mode": "direct"}

    parsed = parse_request_payload(payload)

    assert "x_adapter_critic" not in parsed.request_options
    assert parsed.request_options["tool_choice"] == "auto"
    assert parsed.request_options["tools"][0]["function"]["name"] == "cancel_reservation"

    assistant_extra = parsed.request.messages[0].model_extra
    tool_extra = parsed.request.messages[1].model_extra
    assert assistant_extra is not None
    assert tool_extra is not None
    assert assistant_extra["tool_calls"][0]["id"] == "call_cancel"
    assert tool_extra["tool_call_id"] == "call_cancel"
