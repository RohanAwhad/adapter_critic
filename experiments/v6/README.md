# V6 Experiment Server

## What this experiment is

`experiments/v6/` is a local playground for running `adapter_critic` with file-based adapter and critic system prompts.

Use it to:
- iterate quickly on prompt text
- run real requests through `served-adapter` or `served-critic`
- inspect `adapter_critic.intermediate` and token usage

## Files in this folder

- `critic_system_prompt.txt`: critic prompt used by this experiment
- `run_server.py`: starts FastAPI app with local routing and both prompts
- `upstream_resolution.py`: `UPSTREAM_HOST` resolution and validation helpers
- `README.md`: this doc

## V2-specific critic behavior

The critic prompt is MAST-based and returns natural language feedback.
- It evaluates the draft against all 14 UCB MAST failure modes.
- If no failures are found, the critic must return exactly: `looks good to me`.
- No structured response format is used for critic output.

## Runtime wiring used by `run_server.py`

- server binds to `0.0.0.0:8000`
- upstream host comes from `UPSTREAM_HOST` (defaults to `localhost`)
- `UPSTREAM_HOST` must be a bare host value (no scheme/path/query/fragment/port)
- `served-direct` -> API model at `http://<upstream-host>:8101/v1` (`gpt-oss-120b`)
- `served-adapter` ->
  - API draft model at `http://<upstream-host>:8101/v1` (`gpt-oss-120b`)
  - adapter model at `http://<upstream-host>:8100/v1` (`qwen3-30b-a3b-thinking`)
- `served-critic` ->
  - API draft + second pass model at `http://<upstream-host>:8101/v1` (`gpt-oss-120b`)
  - critic model at `http://<upstream-host>:8100/v1` (`qwen3-30b-a3b-thinking`)

The adapter stage is configured to return structured JSON patches.
The critic stage returns natural language feedback.

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
UPSTREAM_HOST=host.docker.internal uv run python -m experiments.v6.run_server
```

Or default to localhost:

```bash
uv run python -m experiments.v6.run_server
```

Server endpoint:

- `POST http://127.0.0.1:8000/v1/chat/completions`

## Example request (critic)

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "served-critic",
    "messages": [{"role": "user", "content": "Write one sentence about Paris."}]
  }'
```

## What to inspect in response

Look at:

- `choices[0].message`
- `choices[0].finish_reason`
- `adapter_critic.intermediate`
- `adapter_critic.tokens`

In critic mode, `adapter_critic.intermediate` includes:

- `api_draft`: API draft text content
- `api_draft_tool_calls`: JSON string of API draft tool calls (when present)
- `critic`: critic model feedback
- `final`: final text from second API pass

## Logs

- app log file: `logs/adapter_critic.log`
- default logging level in this experiment: `DEBUG`

## Stop server

- if running in foreground: `Ctrl+C`
- if running in tmux: stop/kill that window or send `Ctrl+C` in-pane
