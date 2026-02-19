# Work Plan

## Operating Rules (all milestones)
- Human comments check cadence:
  - check `plans/HUMAN_COMMENTS.md` at iteration start
  - check `plans/HUMAN_COMMENTS.md` before each verifier run
  - log checks and actions in `devlogs.md`
- Git/PR cadence (stacked diffs):
  - branch `M(n)` from latest active stack branch (`M(n-1)`), or from base branch for first PR
  - push every branch to remote (`git push -u origin <branch>`)
  - open one PR per branch when ready
  - if PR is not first in stack, include parent merge checkbox in PR body

## M1 - Scaffold service + toolchain
- Files/paths likely touched:
  - `pyproject.toml`
  - `.pre-commit-config.yaml`
  - `src/adapter_critic/__init__.py`
  - `src/adapter_critic/app.py`
  - `tests/unit/test_smoke.py`
- I/O contracts:
  - FastAPI app exposes `POST /v1/chat/completions`.
  - Minimal OpenAI-shaped response contract type exists.
- Acceptance checks:
  - `uv run ruff format --check .`
  - `uv run ruff check .`
  - `uv run mypy .`
  - `uv run pytest -q tests/unit/test_smoke.py`
- Stop condition (stuck 3 iterations):
  - Roll back to last checkpoint that had green lint/type checks.

## M2 - Contracts + config resolution (tests first)
- Files/paths likely touched:
  - `src/adapter_critic/contracts.py`
  - `src/adapter_critic/config.py`
  - `tests/unit/test_contracts.py`
  - `tests/unit/test_config_resolution.py`
- I/O contracts:
  - Request parsing captures `model`, `messages`, known OpenAI args, and `x_adapter_critic` controls from `extra_body`.
  - Startup config maps served model id to mode + model/base-url settings.
- Acceptance checks:
  - Unit tests cover resolution precedence and validation errors.
- Stop condition (stuck 3 iterations):
  - Freeze feature work; reduce contract surface and re-derive from failing tests.

## M3 - Direct mode workflow
- Files/paths likely touched:
  - `src/adapter_critic/workflows/direct.py`
  - `src/adapter_critic/dispatcher.py`
  - `tests/behavior/test_direct_mode.py`
- I/O contracts:
  - Input: resolved runtime config + chat request payload.
  - Output: OpenAI-compatible chat completion + `adapter_critic` stage metadata.
- Acceptance checks:
  - Behavior test proves request is forwarded to API model/base_url from resolved config.
  - Usage/token metadata for single stage are present.
- Stop condition (stuck 3 iterations):
  - Lock direct mode as baseline and defer multi-stage orchestration.

## M4 - Adapter mode (API -> Adapter)
- Files/paths likely touched:
  - `src/adapter_critic/prompts.py`
  - `src/adapter_critic/workflows/adapter.py`
  - `src/adapter_critic/edits.py`
  - `tests/unit/test_edits.py`
  - `tests/behavior/test_adapter_mode.py`
- I/O contracts:
  - Adapter sees full conversation + latest API draft.
  - Adapter output is either `lgtm` or search/replace blocks.
  - Final response is draft (lgtm) or edited draft.
- Acceptance checks:
  - Behavior test verifies two calls in order: API then Adapter.
  - Unit tests validate search/replace parsing/application and tool-call preservation.
- Stop condition (stuck 3 iterations):
  - Fallback to lgtm-only path and open BB item for richer edit grammar.

## M5 - Critic mode sandwich (API -> Critic -> API)
- Files/paths likely touched:
  - `src/adapter_critic/workflows/critic.py`
  - `src/adapter_critic/prompts.py`
  - `tests/behavior/test_critic_mode.py`
- I/O contracts:
  - Critic sees full conversation, latest API draft, and system prompt.
  - Second API call receives critic feedback in prompt scaffolding and returns final response.
- Acceptance checks:
  - Behavior test verifies three calls in strict order with expected models/base_urls.
  - Intermediate stage payload includes draft, critic output, and final response.
- Stop condition (stuck 3 iterations):
  - Freeze prompt complexity; reduce to deterministic critic feedback envelope.

## M6 - Token accounting + response telemetry
- Files/paths likely touched:
  - `src/adapter_critic/usage.py`
  - `src/adapter_critic/response_builder.py`
  - `tests/unit/test_usage.py`
  - `tests/behavior/test_response_telemetry.py`
- I/O contracts:
  - Each stage has token usage.
  - Top-level `adapter_critic.tokens` includes per-stage and total values.
- Acceptance checks:
  - Unit tests verify deterministic token aggregation from stage usage fixtures.
  - Behavior tests assert token telemetry fields survive SDK parsing.
- Stop condition (stuck 3 iterations):
  - Keep raw stage usage only, block merge until token schema decision is explicit in devlogs.

## M7 - Startup serve config + per-request override behavior
- Files/paths likely touched:
  - `src/adapter_critic/config.py`
  - `src/adapter_critic/dispatcher.py`
  - `tests/behavior/test_served_model_routing.py`
- I/O contracts:
  - Startup config binds served model ids to mode/model/base-url defaults.
  - `extra_body.x_adapter_critic` can override mode and secondary model/base-url.
- Acceptance checks:
  - Behavior tests cover:
    - startup-bound direct call with no request overrides
    - startup-bound adapter call
    - startup-bound critic call
    - request override of mode/model/base_url
- Stop condition (stuck 3 iterations):
  - Disable request override for base_url and keep only model/mode overrides.

## M8 - Docs and wrap-up
- Files/paths likely touched:
  - `README.md`
  - `devlogs.md`
  - `plans/FEEDBACK.md`
- I/O contracts:
  - README includes startup command, config file example, and request examples for direct/adapter/critic.
- Acceptance checks:
  - All verifier checks pass and DoD checklist is fully checked.
- Stop condition (stuck 3 iterations):
  - Publish minimal README with direct mode + config and track missing sections in BACKBURNER.
