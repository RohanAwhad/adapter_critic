# System Design

## Goal

Provide an OpenAI-compatible `POST /v1/chat/completions` wrapper that can run one of three workflows:

- `direct`: single API model call
- `adapter`: API draft then adapter review/edit
- `critic`: API draft, critic feedback, second API pass

## High-Level Architecture

- Request entry: `src/adapter_critic/app.py`
- Request contracts/parsing: `src/adapter_critic/contracts.py`
- Startup config and runtime resolution: `src/adapter_critic/config.py`
- Workflow dispatch: `src/adapter_critic/dispatcher.py`
- Workflow implementations:
  - `src/adapter_critic/workflows/direct.py`
  - `src/adapter_critic/workflows/adapter.py`
  - `src/adapter_critic/workflows/critic.py`
- Prompt templates: `src/adapter_critic/prompts.py`
- Adapter edit application: `src/adapter_critic/edits.py`
- Token aggregation: `src/adapter_critic/usage.py`
- OpenAI-compatible response builder: `src/adapter_critic/response_builder.py`
- Upstream transport abstraction: `src/adapter_critic/upstream.py`
- Built-in OpenAI-compatible HTTP gateway: `src/adapter_critic/http_gateway.py`
- Built-in server entrypoint: `src/adapter_critic/server.py`

## Core Contracts

- Incoming request must include:
  - `model: str`
  - `messages: [{role, content}]`
- Workflow overrides are read from either:
  - top-level `x_adapter_critic`
  - `extra_body.x_adapter_critic`
- Startup routing config maps served model ids to workflow + targets:
  - `served_models.<model_id>.mode`
  - `served_models.<model_id>.api`
  - optional `served_models.<model_id>.adapter`
  - optional `served_models.<model_id>.critic`

## Runtime Resolution

1. Parse request and extract overrides.
2. Look up served model in startup config.
3. Resolve effective mode and stage targets with precedence:
   - request override
   - startup default
4. Validate required targets exist for selected mode.
5. Dispatch to selected workflow.

## Workflow Behavior

### Direct

- One call to API target.
- Final response is API response content.

### Adapter

1. Call API target for draft.
2. Build adapter prompt using full history + draft.
3. Call adapter target.
4. Apply adapter output:
   - `lgtm` => keep draft
   - search/replace blocks => apply edits sequentially
5. Return edited or original draft.

### Critic

1. Call API target for draft.
2. Build critic prompt using full history + system prompt + draft.
3. Call critic target.
4. Build second-pass API prompt with critique + prior draft context.
5. Call API target again for final output.

## Response Shape

Response keeps OpenAI chat completion shape (`id`, `object`, `created`, `model`, `choices`, `usage`) and adds:

- `adapter_critic.mode`
- `adapter_critic.intermediate` (stage outputs)
- `adapter_critic.tokens.stages` (per-stage token usage)
- `adapter_critic.tokens.total` (aggregated token usage)

## Upstream Integration Model

- All model calls go through `UpstreamGateway.complete(model, base_url, messages)`.
- Built-in gateway implementation posts to `<base_url>/chat/completions`.
- This keeps workflow logic independent from transport/provider details.

## Operations

- Start with built-in CLI:
  - `uv run adapter-critic-server --config config.json --port 8000`
- API key is optional and read from env (`OPENAI_API_KEY` by default).
- Per-request overrides can change mode/model/base_url without restarting server.

## Testing Strategy

- Unit tests (`tests/unit`): contracts, config resolution, edit parser, prompts, token aggregation, response schema.
- Behavior tests (`tests/behavior`): direct/adapter/critic flows, served-model routing, override precedence, SDK compatibility, telemetry visibility, HTTP gateway behavior.
- Verification chain:
  - `ruff format --check`
  - `ruff check`
  - `mypy`
  - `pytest`

## Current Boundaries

- Non-streaming only (`stream=true` not implemented).
- Message contract currently assumes text `content` (no multimodal structure).
- Tool-call passthrough/transforms are not yet modeled.
