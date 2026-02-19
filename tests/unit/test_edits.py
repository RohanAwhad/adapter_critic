from __future__ import annotations

import pytest

from adapter_critic.edits import (
    apply_adapter_output,
    apply_adapter_output_to_draft,
)


def test_lgtm_returns_original_draft() -> None:
    draft = "hello world"
    assert apply_adapter_output(draft, '{"decision":"lgtm"}') == draft


def test_single_replace_patch_updates_content() -> None:
    draft = "hello world"
    edits = '{"decision":"patch","patches":[{"op":"replace","path":"/content","value":"hello universe"}]}'
    assert apply_adapter_output(draft, edits) == "hello universe"


def test_patch_can_update_content_and_function_call() -> None:
    final_text, final_tool_calls, final_function_call = apply_adapter_output_to_draft(
        content="Draft answer",
        tool_calls=None,
        function_call={"name": "transfer_to_human", "arguments": "{}"},
        adapter_output=(
            '{"decision":"patch","patches":['
            '{"op":"replace","path":"/content","value":"Final answer"},'
            '{"op":"replace","path":"/function_call/arguments","value":"{\\"reason\\":\\"user_requested\\"}"}'
            "]}"
        ),
    )

    assert final_text == "Final answer"
    assert final_tool_calls is None
    assert final_function_call == {
        "name": "transfer_to_human",
        "arguments": '{"reason":"user_requested"}',
    }


def test_invalid_json_raises_error() -> None:
    with pytest.raises(ValueError):
        apply_adapter_output("hello", "not-valid")


def test_unsupported_patch_op_raises_error() -> None:
    with pytest.raises(ValueError, match="unsupported patch op"):
        apply_adapter_output(
            "hello",
            '{"decision":"patch","patches":[{"op":"remove","path":"/content","value":null}]}',
        )


def test_disallowed_patch_path_raises_error() -> None:
    with pytest.raises(ValueError, match="unsupported patch path"):
        apply_adapter_output(
            "hello",
            '{"decision":"patch","patches":[{"op":"replace","path":"/unknown","value":"x"}]}',
        )


def test_out_of_range_tool_call_index_raises_error() -> None:
    with pytest.raises(ValueError, match="path not found"):
        apply_adapter_output_to_draft(
            content="",
            tool_calls=[
                {
                    "id": "call_cancel",
                    "type": "function",
                    "function": {
                        "name": "cancel_reservation",
                        "arguments": '{"reservation_id":"WRONG"}',
                    },
                }
            ],
            function_call=None,
            adapter_output=(
                '{"decision":"patch","patches":['
                '{"op":"replace","path":"/tool_calls/2/function/arguments","value":"{\\"reservation_id\\":\\"EHGLP3\\"}"}'
                "]}"
            ),
        )


def test_apply_adapter_output_to_draft_can_edit_tool_call_arguments() -> None:
    final_text, final_tool_calls, final_function_call = apply_adapter_output_to_draft(
        content="",
        tool_calls=[
            {
                "id": "call_cancel",
                "type": "function",
                "function": {
                    "name": "cancel_reservation",
                    "arguments": '{"reservation_id":"WRONG"}',
                },
            }
        ],
        function_call=None,
        adapter_output=(
            '{"decision":"patch","patches":['
            '{"op":"replace","path":"/tool_calls/0/function/arguments","value":"{\\"reservation_id\\":\\"EHGLP3\\"}"}'
            "]}"
        ),
    )

    assert final_text == ""
    assert final_function_call is None
    assert final_tool_calls is not None
    assert final_tool_calls[0]["function"]["arguments"] == '{"reservation_id":"EHGLP3"}'


def test_patch_decision_requires_non_empty_patches() -> None:
    with pytest.raises(ValueError, match="non-empty patches"):
        apply_adapter_output("hello", '{"decision":"patch","patches":[]}')
