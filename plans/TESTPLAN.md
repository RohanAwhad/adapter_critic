# Test Plan

## Strategy
- Use TDD incrementally by milestone.
- Do not write all tests upfront; write tests immediately before each feature slice.
- Keep a strict split:
  - unit tests: pure logic/transform/contract checks
  - behavior tests: end-to-end API behavior with mocked upstream models

## Unit Tests

### U-001 Request control parsing
- Validate extraction of `x_adapter_critic` from request extra-body payload.
- Validate unknown controls are rejected with clear validation failure.

### U-002 Startup config resolution
- Validate served model-id lookup from startup config.
- Validate precedence when request override exists.

### U-003 Adapter edit parser
- Cases: `lgtm`, single replace, multi replace, missing match, malformed block.

### U-004 Critic prompt composition
- Ensure critic prompt includes:
  - full conversation history
  - system prompt content
  - latest API draft response

### U-005 Token usage aggregator
- Input: usage fixtures per stage.
- Output: per-stage token counts + total token counts.
- Assert deterministic aggregation and non-negative totals.

### U-006 Response extension schema
- Ensure top-level OpenAI fields remain valid.
- Ensure extension object includes `mode`, `intermediate`, `tokens`.

## Behavior Tests

### B-001 Direct mode path
- One upstream call (API model/base_url).
- Response shape compatible with `/v1/chat/completions`.

### B-002 Adapter mode path
- Call order: API draft -> Adapter refine.
- Adapter sees full history + draft.
- Final response is draft-or-edited draft.

### B-003 Critic mode path
- Call order: API draft -> Critic -> API final.
- Critic sees full history + system prompt + latest draft.
- Final API call includes critic guidance.

### B-004 Startup served model routing
- Startup config registers model-id aliases for direct/adapter/critic.
- Client only sends configured served model-id and request works without extra overrides.

### B-005 Request override via extra_body
- Request can override mode and adapter/critic model/base_url.
- Verify override precedence over startup defaults.

### B-006 Intermediate response visibility
- Response includes stage-level intermediate outputs for all invoked stages.

### B-007 Token visibility
- Response includes `adapter_critic.tokens.total` and per-stage token usage.

### B-008 SDK compatibility smoke
- Use OpenAI SDK against local server with `extra_body` in request.
- Verify SDK parse succeeds and extension fields remain accessible.

## Test Infrastructure
- Use deterministic upstream stubs (transport-level mocking) to avoid live network.
- Keep behavior tests fast; no external dependencies.

## Exit Criteria
- All unit tests pass.
- All behavior tests pass.
- Verifier chain in `plans/VERIFIER.md` passes without manual patching.
