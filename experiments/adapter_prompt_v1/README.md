# Adapter Prompt V1 Play Server

## What this experiment is

`experiments/adapter_prompt_v1/` is a local playground for running `adapter_critic` with a file-based adapter system prompt.

It is useful when you want to:
- iterate on adapter prompt text quickly
- run real chat requests through `served-adapter`
- inspect `adapter_critic.intermediate` fields and token usage

## Files in this folder

- `adapter_system_prompt.txt`: adapter prompt used by this experiment
- `run_server.py`: starts FastAPI app with local model routing and this prompt
- `upstream_resolution.py`: `UPSTREAM_HOST` resolution and validation helpers
- `README.md`: this doc

## Runtime wiring used by `run_server.py`

- server binds to `0.0.0.0:8000`
- upstream host comes from `UPSTREAM_HOST` (defaults to `localhost`)
- `UPSTREAM_HOST` must be a bare host value (no scheme/path/query/fragment/port)
- `served-direct` -> API model at `http://<upstream-host>:8101/v1` (`gpt-oss-120b`)
- `served-adapter` ->
  - API draft model at `http://<upstream-host>:8101/v1` (`gpt-oss-120b`)
  - adapter model at `http://<upstream-host>:8100/v1` (`gpt-oss-20b`)
- `served-critic` -> API on `8101`, critic on `8100`

The adapter stage is configured to return structured JSON patches.

## Prerequisites

1. `uv` environment is ready in repo root.
2. OpenAI-compatible upstreams are running on the selected upstream host on ports `8100` and `8101`.

Quick check:

```bash
UPSTREAM_HOST=${UPSTREAM_HOST:-localhost}
curl -sS "http://${UPSTREAM_HOST}:8100/v1/models"
curl -sS "http://${UPSTREAM_HOST}:8101/v1/models"
```

## Start the play server

From repo root:

```bash
UPSTREAM_HOST=host.docker.internal uv run experiments/adapter_prompt_v1/run_server.py
```

Or default to localhost:

```bash
uv run experiments/adapter_prompt_v1/run_server.py
```

Server endpoint:

- `POST http://127.0.0.1:8000/v1/chat/completions`

## Example request (plain text)

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "served-adapter",
    "messages": [{"role": "user", "content": "Write one sentence about Paris."}]
  }'
```

## Example request (tool calling)

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "served-adapter",
    "messages": [{"role": "user", "content": "Get weather for Paris in celsius."}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"},
              "units": {"type": "string"}
            },
            "required": ["city", "units"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'
```

## What to inspect in response

Look at:

- `choices[0].message`
- `choices[0].finish_reason`
- `adapter_critic.intermediate`
- `adapter_critic.tokens`

In adapter mode, `adapter_critic.intermediate` includes:

- `api_draft`: API draft text content
- `api_draft_tool_calls`: JSON string of API draft tool calls (when present)
- `api_draft_function_call`: JSON string of API draft function call (when present)
- `adapter`: raw adapter model structured response
- `final`: final text returned after adapter processing
- `adapter_rejection_reason`: present when adapter candidate is rejected

## Logs

- app log file: `logs/adapter_critic.log`
- default logging level in this experiment: `DEBUG`

If debugging a request, check the latest incoming/outgoing request lines in `logs/adapter_critic.log`.

## Stop server

- if running in foreground: `Ctrl+C`
- if running in tmux: stop/kill that window or send `Ctrl+C` in-pane
