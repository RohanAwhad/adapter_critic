from __future__ import annotations

import pytest

from adapter_critic.edits import (
    apply_adapter_output,
    apply_adapter_output_to_draft,
)


def test_lgtm_returns_original_draft() -> None:
    draft = "hello world"
    assert apply_adapter_output(draft, "lgtm") == draft


def test_single_replace_block() -> None:
    draft = "hello world"
    edits = "<<<<<<< SEARCH\nworld\n=======\nuniverse\n>>>>>>> REPLACE"
    assert apply_adapter_output(draft, edits) == "hello universe"


def test_multi_replace_blocks() -> None:
    draft = "alpha beta gamma"
    edits = "<<<<<<< SEARCH\nalpha\n=======\nA\n>>>>>>> REPLACE\n<<<<<<< SEARCH\ngamma\n=======\nG\n>>>>>>> REPLACE"
    assert apply_adapter_output(draft, edits) == "A beta G"


def test_missing_search_raises_error() -> None:
    with pytest.raises(ValueError, match="search text"):
        apply_adapter_output("hello", "<<<<<<< SEARCH\nmissing\n=======\nnew\n>>>>>>> REPLACE")


def test_malformed_block_raises_error() -> None:
    with pytest.raises(ValueError, match="malformed"):
        apply_adapter_output("hello", "not-valid")


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
            "<<<<<<< SEARCH\n"
            '{\\"reservation_id\\":\\"WRONG\\"}\n'
            "=======\n"
            '{\\"reservation_id\\":\\"EHGLP3\\"}\n'
            ">>>>>>> REPLACE"
        ),
    )

    assert final_text == ""
    assert final_function_call is None
    assert final_tool_calls is not None
    assert final_tool_calls[0]["function"]["arguments"] == '{"reservation_id":"EHGLP3"}'
