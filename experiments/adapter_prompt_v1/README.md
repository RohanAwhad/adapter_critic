## Adapter Prompt V1

Built after reviewing prior adapter/refiner prompt attempts in:

- `../api_adapter/new_api.py` (`REFINER_SYS_PROMPT`)
- `../api_adapter_v2026/main.py` (`MODEL2_SYSTEM_PROMPT`)
- `../api_adapter_revamped/llms.py` (`SURGICAL_ADAPTER_SYSTEM_PROMPT`)
- `../api_adapter/api.py` (`REFINER_SYS_PROMPT`)

This version keeps the strongest shared ideas and matches `adapter_critic` runtime behavior:

- strict output contract (`lgtm` or SEARCH/REPLACE blocks only)
- surgical edits instead of full rewrites
- preservation-first policy
- focus on latest user turn while using history for context
- exact-match replacement semantics compatible with `src/adapter_critic/edits.py`

Prompt text is in `adapter_system_prompt.txt`.
