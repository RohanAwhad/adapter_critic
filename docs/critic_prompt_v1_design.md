# Critic Prompt V1 Design

## Context

The adapter_critic server has three workflows: `direct`, `adapter`, `critic`.

The **adapter** workflow uses a small model to surgically patch the API model's draft via structured JSON patches. The small model is the final editor — it directly mutates content, tool calls, and function calls.

The **critic** workflow uses a small model differently: it generates natural language feedback on the API model's draft, then the big API model gets a second pass with that feedback to self-correct. The big model is the final editor — it decides what to act on.

Currently the critic has a minimal default prompt (`prompts.py:CRITIC_SYSTEM_PROMPT`) and does not receive tool contract information. This design doc covers the v1 critic prompt and the plumbing changes needed to support it.

## Design Goals

1. The critic prompt should cover the same review surface as the adapter prompt (content, tool calls, tool arguments, missing/extra calls, formatting) but output natural language feedback instead of JSON patches.
2. The critic model does NOT need structured output. Its feedback is consumed by the 120b API model which can parse natural language.
3. The critic model MUST receive the tool contract (available tools, tool_choice, function_call) so it can validate tool usage against the request schema. Today this info is missing from the critic path.
4. Reuse the same draft presentation format (`build_adapter_draft_payload` XML tags) that the adapter workflow already uses, so the critic sees content, tool_calls, and function_call in a structured way.

## Current Critic Flow (what exists today)

```
1. API model produces draft
2. build_adapter_draft_payload(content, tool_calls, function_call) -> tagged XML string
3. build_critic_messages(messages, system_prompt, draft, critic_system_prompt) -> critic input
4. Critic model generates feedback (plain text)
5. build_critic_second_pass_messages(messages, draft, critique) -> second pass input
6. API model produces final response
```

### Gaps in current flow

- `build_critic_messages` does NOT receive `request_options`, so the critic model has no visibility into available tools/schemas.
- The default `CRITIC_SYSTEM_PROMPT` is a single generic sentence with no guidance on what to review or how to structure feedback.
- The second pass prompt includes the raw draft but does not explicitly surface tool call information in a reviewable format. (However, the second pass API call does receive `request_options` via `gateway.complete`, so the API model has tool schemas through the normal OpenAI `tools` parameter — no change needed there.)
- The critic `intermediate` dict only stores `.content` strings for each stage. If the API draft included tool calls or function calls, they are invisible in the intermediate debugging payload. The adapter workflow handles this by conditionally adding `api_draft_tool_calls` and `api_draft_function_call` (JSON-serialized) to its intermediate dict — the critic workflow does not.

## Planned Changes

### 1. Critic system prompt (`experiments/v1/critic_system_prompt.txt`)

The prompt will instruct the critic model to:

- Review the API draft for correctness, completeness, and proper tool usage
- Review categories:
  - Content accuracy and completeness
  - Tool call selection (right tool for the task)
  - Tool call arguments (correct format, correct parameters, matches schema)
  - Missing or extra tool calls
  - Confirmation/safety ordering (e.g., ask before destructive actions)
  - Formatting and structure
- Output clear natural language feedback:
  - What is correct in the draft
  - What is wrong or missing, with specific fix instructions
  - If the draft is acceptable, say "LGTM" (or similar short signal)
- NOT output JSON patches, structured responses, or tool calls of its own

### 2. Tool contract in critic messages (`prompts.py`)

- `build_critic_messages` will accept an optional `request_options` parameter
- Use the existing `_render_tool_contract(request_options)` helper to render tool schemas into the critic's system prompt, same pattern as `build_adapter_messages`
- This gives the critic visibility into what tools are available and what the tool_choice constraints are

### 3. Workflow wiring (`workflows/critic.py`)

- Pass `request_options` from `run_critic` into `build_critic_messages`
- Add `api_draft_tool_calls` and `api_draft_function_call` to the critic `intermediate` dict when present, same pattern as the adapter workflow (`workflows/adapter.py:174-177`). This surfaces draft tool call info in the `adapter_critic.intermediate` response payload for debugging.
- No changes needed for the second pass — it already receives `request_options` via `gateway.complete`

### 4. Experiment setup (`experiments/v1/`)

- `critic_system_prompt.txt`: the v1 critic prompt file (colocated with adapter prompt in same experiment dir)
- `run_server.py`: single experiment runner that loads both adapter and critic prompts and wires up all served models

## What does NOT change

- The second pass message construction (`build_critic_second_pass_messages`) — no changes needed. The API model already gets tool schemas via `request_options` in the `gateway.complete` call.
- The critic model is NOT called with `response_format` / JSON mode. Natural language output only.
- No new structured output parsing for critic responses.
- The `WorkflowOutput` shape and response builder remain unchanged.
- The adapter workflow and its prompt are untouched.

## Draft Presentation Format (reference)

The critic will receive the API draft in the same XML-tagged format used by the adapter:

```
<ADAPTER_DRAFT_CONTENT>
{content text}
</ADAPTER_DRAFT_CONTENT>
<ADAPTER_DRAFT_TOOL_CALLS>
{tool_calls JSON array}
</ADAPTER_DRAFT_TOOL_CALLS>
<ADAPTER_DRAFT_FUNCTION_CALL>
{function_call JSON or null}
</ADAPTER_DRAFT_FUNCTION_CALL>
```

This format is produced by `edits.build_adapter_draft_payload`.

## Testing Plan

### Unit tests (`tests/unit/test_prompts.py`)

These verify the plumbing changes to `build_critic_messages`.

1. **Tool contract rendered when `request_options` has tools** — call `build_critic_messages` with `request_options` containing `tools` and `tool_choice`. Assert the critic system prompt (`messages[0].content`) contains "Authoritative tool contract for this request", the tool name, and `tool_choice` value. Mirrors the existing `test_adapter_prompt_includes_tool_contract_when_tools_are_provided`.

2. **No tool contract when `request_options` is None or has no tools** — call `build_critic_messages` without `request_options` (or with empty tools). Assert the system prompt does NOT contain "Authoritative tool contract". Ensures backward compatibility.

### Behavior tests

These verify end-to-end critic workflow behavior through the HTTP boundary using `FakeGateway`.

3. **Critic receives tool contract in system prompt** (`tests/behavior/test_tool_passthrough.py`) — extend or add alongside the existing `test_served_critic_forwards_tool_options_and_returns_tool_calls`. Assert that `gateway.calls[1]["messages"][0].content` (the critic's system prompt) contains "Authoritative tool contract", the tool name, and `tool_choice`. The existing test already checks the draft content but not the system prompt tool contract.

4. **Critic intermediate includes draft tool call info** (`tests/behavior/test_critic_mode.py`) — new test: send a critic-mode request where the API draft returns tool calls. Assert that `adapter_critic.intermediate` in the response contains `api_draft_tool_calls` key with the serialized tool call data. Optionally test `api_draft_function_call` for legacy function_call coverage.

### What we do NOT test

- Second pass tool forwarding — already works and already tested (`gateway.calls[2]["request_options"]` assertions exist in `test_tool_passthrough.py:258-264`).
- Prompt content quality — that's the experiment's job (`experiments/v1/`), not unit/behavior tests. The experiment uses live models to validate the prompt produces useful feedback.
- Critic structured output parsing — there is none. The critic returns natural language.

## Critic Prompt Design Principles

1. **Review breadth over precision** — the critic should flag anything suspicious. False positives are OK because the big API model will filter them on the second pass.
2. **Specific over vague** — "tool_calls[0].function.arguments is missing required field 'units'" is better than "the tool call looks wrong".
3. **Natural language over structure** — no JSON, no patches, no special formats. Just clear feedback text.
4. **Short-circuit signal** — if the draft is good, the critic should say so briefly (e.g., "LGTM") so the second pass can reproduce the draft without unnecessary changes.
