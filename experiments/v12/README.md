# V12 Experiment: served-direct on Vertex Claude Sonnet 4.5

## What this experiment is

`experiments/v12/` runs `adapter_critic` in `served-direct` mode only, with the
API stage pointed at Vertex Anthropic Claude.

Default model matches current Vertex tests:

- `claude-sonnet-4-5@20250929`

## Runtime wiring used by `run_server.py`

- server binds to `0.0.0.0:8000`
- `served-direct` -> Vertex Anthropic model using:
  - project: `ANTHROPIC_VERTEX_PROJECT_ID` or `GOOGLE_CLOUD_PROJECT`
  - region: `CLOUD_ML_REGION` or `VERTEX_LOCATION` (default `us-east5`)
  - model: `ANTHROPIC_MODEL` (default `claude-sonnet-4-5@20250929`)

## Prerequisites

1. `uv` environment is ready in repo root.
2. Vertex project access + auth is available (`gcloud auth application-default login` or equivalent ADC).
3. Project/region/model env vars are set (or defaults apply).

## Start the server

```bash
ANTHROPIC_VERTEX_PROJECT_ID=<your-project> \
CLOUD_ML_REGION=us-east5 \
ANTHROPIC_MODEL=claude-sonnet-4-5@20250929 \
uv run python -m experiments.v12.run_server
```

## Endpoint

- `POST http://127.0.0.1:8000/v1/chat/completions`

## Example request

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "served-direct",
    "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
    "temperature": 0,
    "max_tokens": 32
  }'
```

## Probable output

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "served-direct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "pong"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 2,
    "total_tokens": 12
  },
  "adapter_critic": {
    "mode": "direct",
    "intermediate": {
      "api": "pong"
    },
    "tokens": {
      "stages": {
        "api": {
          "prompt_tokens": 10,
          "completion_tokens": 2,
          "total_tokens": 12
        }
      },
      "total": {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12
      }
    }
  }
}
```

## Logs

- app log file: `logs/adapter_critic.log`
- default logging level in this experiment: `DEBUG`
