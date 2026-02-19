from __future__ import annotations

import pytest

from adapter_critic.edits import apply_adapter_output


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
