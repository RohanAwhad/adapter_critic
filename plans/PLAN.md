# Plan Entry Point

## Current Objective
Build a FastAPI OpenAI Chat Completions wrapper that supports three workflows:
- direct: API model only
- adapter: API -> Adapter (search/replace refinement)
- critic: API -> Critic -> API (sandwich)

The server must support:
- startup model-id routing config (served model id -> workflow + model/base-url config)
- per-request workflow/model selection via OpenAI `extra_body`
- OpenAI-compatible response shape with extra telemetry fields for intermediate responses and token usage breakdowns
- unit + behavior tests, TDD style
- toolchain guardrails (`ruff format`, `ruff`, `mypy`, `pytest`) via pre-commit, line-length 120

## Scope
In scope:
- New Python service implementation in this repo
- Workflow orchestrators and prompt templates
- Startup config loader + request override resolver
- Response extension payload (`adapter_critic`) with intermediate outputs and token usage
- Unit and behavior tests
- README usage docs for direct/adapter/critic modes

Out of scope (for this objective):
- Streaming/SSE support
- Auth and multi-tenant rate limiting
- Persistent storage for traces/usage

## Success Criteria
- All verifier commands in `plans/VERIFIER.md` pass.
- Behavior tests prove the three workflow paths and config routing behavior.
- Response includes intermediate stage outputs and token totals/per-stage token usage.
- README documents startup and request usage for direct/adapter/critic paths.

## Constraints
- Strong typing and mypy-friendly code.
- No broad refactors, keep surface area minimal.
- No `try/except` unless explicitly requested.
- No attribution/watermarks.
- TDD incremental flow (tests written milestone-by-milestone, not all upfront).

## Key Decisions
- Request control namespace: `extra_body.x_adapter_critic`.
- Response extension namespace: top-level `adapter_critic` object.
- Token strategy: no estimated cost reporting; expose per-stage usage and total usage only.
- Mode resolution order:
  1. startup served model config
  2. request `extra_body.x_adapter_critic` overrides
  3. explicit validation error if unresolved/invalid
- Human loop communication file: `plans/HUMAN_COMMENTS.md` in repo root, checked every iteration.
- Git strategy: stacked-diff branches, each pushed to remote, PR per branch with parent-merge checkbox when not first.

## Prompt Drafts (implementation target)

### Adapter system prompt (API -> Adapter)
- Role: response editor over API draft.
- Inputs:
  - full conversation history
  - latest API draft (text and/or tool calls)
  - available tool schemas
- Output contract:
  - `lgtm`, or
  - one or more search/replace blocks over the draft
- Guardrails:
  - preserve valid content and valid tool calls
  - only make minimal surgical edits
  - no extra prose outside edit grammar

### Critic system prompt (API -> Critic -> API)
- Role: critique generator, not final responder.
- Inputs:
  - full conversation history
  - system instructions
  - latest API draft
- Output contract:
  - structured critique with:
    - what is correct
    - what is wrong or missing
    - exact actionable fix instructions for a second API pass
- Guardrails:
  - no direct user-facing final answer
  - focus on factual/tool-call correctness and policy alignment

## Working Docs
- `plans/WORK.md` - ordered milestones for implementer
- `plans/TESTPLAN.md` - tests to write and expected behavior
- `plans/VERIFIER.md` - exact command chain and pass/fail
- `plans/RISKS.md` - gotchas and rollback triggers
- `plans/CHECKLIST.md` - definition of done
- `plans/BACKBURNER.md` - non-critical tasks with trigger rules
- `plans/IMPLEMENTER.md` - execution guardrails
- `plans/DEVLOG_CONTRACT.md` - devlogs/checkpoint policy
- `plans/FEEDBACK.md` - loop feedback sink
- `plans/HUMAN_COMMENTS.md` - async human instructions to loop
