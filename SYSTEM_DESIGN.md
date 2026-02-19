# System Design

## Purpose

`adapter_critic` exposes an OpenAI-compatible `POST /v1/chat/completions` endpoint and routes each request into one of three workflows:

- `direct`: API only
- `adapter`: API -> Adapter -> final text (via `lgtm` or SEARCH/REPLACE edits)
- `critic`: API -> Critic -> API second pass

## Design Shape

- Functional-first pipeline for request parsing, config resolution, dispatch, usage aggregation, and response assembly.
- One minimal runtime state holder: `RuntimeState` in `src/adapter_critic/runtime.py`.
- Side effects isolated to:
  - FastAPI HTTP boundary in `src/adapter_critic/app.py`
  - upstream model calls via `UpstreamGateway` (`src/adapter_critic/upstream.py`)

`RuntimeState` fields:

- `config`
- `gateway`
- `id_provider`
- `time_provider`

## Core Modules

- `src/adapter_critic/server.py`: CLI entrypoint, config load, app boot.
- `src/adapter_critic/app.py`: route handler and top-level orchestration.
- `src/adapter_critic/contracts.py`: request models + override extraction.
- `src/adapter_critic/config.py`: served-model routing + override resolution.
- `src/adapter_critic/dispatcher.py`: mode-to-workflow dispatch.
- `src/adapter_critic/workflows/*.py`: mode implementations.
- `src/adapter_critic/prompts.py`: adapter/critic prompt composition.
- `src/adapter_critic/edits.py`: adapter SEARCH/REPLACE application.
- `src/adapter_critic/usage.py`: token aggregation.
- `src/adapter_critic/response_builder.py`: OpenAI-shaped response + extension payload.
- `src/adapter_critic/http_gateway.py`: built-in OpenAI-compatible upstream transport.

## Request/Response Contracts

Request minimum:

- `model: str`
- `messages: [{role, content}]`

Workflow overrides:

- top-level `x_adapter_critic`
- or `extra_body.x_adapter_critic`
- precedence: top-level wins if both exist
- mode-stage fallback: if adapter/critic target is missing, that stage falls back to resolved API target
- partial secondary override without enough data to resolve a target is rejected

Per-served-model prompt config (startup):

- `adapter_system_prompt`
- `critic_system_prompt`
- if not set, defaults from `src/adapter_critic/prompts.py` are used

Per-stage API key env config (startup):

- each stage target supports `api_key_env` (`api_key_var` alias accepted)
- gateway resolves bearer token from stage env var per call
- if unset, gateway uses default env name (CLI `--api-key-env`, default `OPENAI_API_KEY`)

Response:

- standard OpenAI chat-completions shape (`id`, `object`, `created`, `model`, `choices`, `usage`)
- extension payload at `adapter_critic`:
  - `mode`
  - `intermediate`
  - `tokens.stages`
  - `tokens.total`

Invariant:

- top-level `usage == adapter_critic.tokens.total`

## ASCII System Diagram

```text
                               +-------------------------+
                               |     RuntimeState        |
                               |-------------------------|
                               | config                  |
                               | gateway (UpstreamGateway)|
                               | id_provider             |
                               | time_provider           |
                               +-----------+-------------+
                                           |
Client                                      |
  |                                         |
  v                                         v
+-------------------------------+   +--------------------------+
| FastAPI /v1/chat/completions |-->| app.py orchestration     |
| (create_app in app.py)       |   | parse -> resolve -> run  |
+---------------+---------------+   +------------+-------------+
                |                                |
                v                                v
      +--------------------+           +-----------------------+
      | contracts.py       |           | config.py             |
      | parse_request_*    |           | resolve_runtime_*     |
      +--------------------+           +-----------------------+
                           \           /
                            v         v
                        +------------------+
                        | dispatcher.py    |
                        | mode dispatch    |
                        +----+--------+----+
                             |        |
                             |        +-------------------------+
                             v                                  v
                   +-------------------+             +-------------------+
                   | workflows/direct  |             | workflows/adapter |
                   | (1 API call)      |             | API->Adapter->edit|
                   +-------------------+             +-------------------+
                              \                           /
                               \                         /
                                v                       v
                               +-------------------------+
                               | workflows/critic        |
                               | API->Critic->API        |
                               +-----------+-------------+
                                           |
                                           v
                               +-------------------------+
                               | usage.py + response_*   |
                               | tokens + final payload  |
                               +-----------+-------------+
                                           |
                                           v
                                       HTTP response

Upstream calls go through:
app/workflows -> UpstreamGateway.complete(...) -> http_gateway.py -> <base_url>/chat/completions
```

## ASCII Sequence Diagram

```text
Common entry path
-----------------
Client
  -> app.chat_completions
  -> contracts.parse_request_payload
  -> config.resolve_runtime_config
  -> dispatcher.dispatch(mode)

Mode: direct
------------
dispatcher
  -> workflows.run_direct
  -> gateway.complete(api)
  <- api response
  -> usage.aggregate_usage
  -> response_builder.build_response
  <- response (mode=direct)

Mode: adapter
-------------
dispatcher
  -> workflows.run_adapter
  -> gateway.complete(api)            # draft
  <- api_draft
  -> prompts.build_adapter_messages
  -> gateway.complete(adapter)
  <- adapter_review
  -> edits.apply_adapter_output
  -> usage.aggregate_usage
  -> response_builder.build_response
  <- response (mode=adapter)

Mode: critic
------------
dispatcher
  -> workflows.run_critic
  -> gateway.complete(api)            # draft
  <- api_draft
  -> prompts.build_critic_messages
  -> gateway.complete(critic)
  <- critic_feedback
  -> prompts.build_critic_second_pass_messages
  -> gateway.complete(api)            # second pass
  <- api_final
  -> usage.aggregate_usage
  -> response_builder.build_response
  <- response (mode=critic)
```

## Current Boundaries

- No streaming path (`stream=true`) in current implementation.
- Message content contract is text-only (`content: str`).
- Built-in gateway expects OpenAI-compatible response shape with `choices[0].message.content` and `usage`.
