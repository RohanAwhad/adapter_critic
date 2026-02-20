from __future__ import annotations

from adapter_critic.response_shape import (
    has_valid_function_call,
    has_valid_tool_calls,
    infer_finish_reason,
    normalize_tool_calls,
)


def test_normalize_tool_calls_empty_list_to_none() -> None:
    assert normalize_tool_calls([]) is None


def test_has_valid_tool_calls_requires_openai_function_fields() -> None:
    assert has_valid_tool_calls(
        [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "cancel_reservation",
                    "arguments": '{"reservation_id":"EHGLP3"}',
                },
            }
        ]
    )
    assert not has_valid_tool_calls(
        [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "cancel_reservation",
                },
            }
        ]
    )


def test_has_valid_tool_calls_requires_arguments_json_object() -> None:
    assert has_valid_tool_calls(
        [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "cancel_reservation",
                    "arguments": '{"reservation_id":"EHGLP3"}',
                },
            }
        ]
    )
    assert not has_valid_tool_calls(
        [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "cancel_reservation",
                    "arguments": '"{}"',
                },
            }
        ]
    )
    assert not has_valid_tool_calls(
        [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "cancel_reservation",
                    "arguments": "{bad",
                },
            }
        ]
    )


def test_has_valid_function_call_requires_name_and_arguments_strings() -> None:
    assert has_valid_function_call({"name": "cancel_reservation", "arguments": '{"reservation_id":"EHGLP3"}'})
    assert not has_valid_function_call({"name": "cancel_reservation", "arguments": None})


def test_infer_finish_reason_uses_message_fields() -> None:
    assert (
        infer_finish_reason(
            "stop",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "cancel_reservation", "arguments": '{"reservation_id":"EHGLP3"}'},
                }
            ],
            function_call=None,
        )
        == "tool_calls"
    )
    assert infer_finish_reason("tool_calls", tool_calls=[], function_call=None) == "stop"
    assert infer_finish_reason("length", tool_calls=None, function_call=None) == "length"
