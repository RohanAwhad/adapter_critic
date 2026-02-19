from __future__ import annotations

import re

EDIT_BLOCK_RE = re.compile(
    r"<<<<<<<\s*SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>>\s*REPLACE",
    flags=re.DOTALL,
)


def apply_adapter_output(draft: str, adapter_output: str) -> str:
    stripped = adapter_output.strip().lower()
    if stripped == "lgtm":
        return draft

    matches = EDIT_BLOCK_RE.findall(adapter_output)
    if not matches:
        raise ValueError("malformed adapter edits")

    updated = draft
    for search_text, replace_text in matches:
        if search_text not in updated:
            raise ValueError("search text not found in draft")
        updated = updated.replace(search_text, replace_text, 1)

    return updated
