# V15 Experiment: served-adapter with Qwen3.5 4B + 0.8B

## What this experiment is

`experiments/v15/` runs `adapter_critic` with a single served model:

- `served-adapter`: API draft from `qwen3.5-4b` then intervention from `qwen3.5-0.8b`

## Runtime wiring used by `run_server.py`

- server binds to `0.0.0.0:8000`
- upstream host comes from `UPSTREAM_HOST` (defaults to `localhost`)
- `served-adapter.api`:
  - model: `qwen3.5-4b`
  - base URL: `http://<UPSTREAM_HOST>:8100/v1`
- `served-adapter.adapter`:
  - model: `qwen3.5-0.8b`
  - base URL: `http://<UPSTREAM_HOST>:8102/v1`

## Prerequisites

1. `uv` environment is ready in repo root.
2. OpenAI-compatible upstreams are running on:
   - `http://<UPSTREAM_HOST>:8100/v1`
   - `http://<UPSTREAM_HOST>:8102/v1`

Quick check:

```bash
UPSTREAM_HOST=${UPSTREAM_HOST:-localhost}
curl -sS "http://${UPSTREAM_HOST}:8100/v1/models"
curl -sS "http://${UPSTREAM_HOST}:8102/v1/models"
```

## Start the server

```bash
uv run python -m experiments.v15.run_server
```

With explicit host override:

```bash
UPSTREAM_HOST=localhost uv run python -m experiments.v15.run_server
```

## Endpoint

- `POST http://127.0.0.1:8000/v1/chat/completions` with `model: "served-adapter"`

## Example request

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "served-adapter",
    "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

## What to inspect in response

- `choices[0].message`
- `choices[0].finish_reason`
- `adapter_critic.intermediate`
- `adapter_critic.tokens`

## Logs

- app log file: `logs/adapter_critic.log`
- default logging level in this experiment: `DEBUG`
