# V12 Experiment: Vertex Claude API + GPT-OSS interventionist

## What this experiment is

`experiments/v12/` runs `adapter_critic` with:

- `served-direct`: Vertex Anthropic Claude only
- `served-adapter`: Vertex Anthropic Claude API draft + GPT-OSS interventionist

Default API model matches current Vertex tests:

- `claude-sonnet-4-5@20250929`

Interventionist model:

- `gpt-oss-120b`

## Runtime wiring used by `run_server.py`

- server binds to `0.0.0.0:8000`
- Vertex API target (`served-direct` + `served-adapter.api`):
  - project: `ANTHROPIC_VERTEX_PROJECT_ID` or `GOOGLE_CLOUD_PROJECT`
  - region: `CLOUD_ML_REGION` or `VERTEX_LOCATION` (default `us-east5`)
  - model: `ANTHROPIC_MODEL` (default `claude-sonnet-4-5@20250929`)
- interventionist target (`served-adapter.adapter`):
  - model: `gpt-oss-120b`
  - base URL: `http://<UPSTREAM_HOST>:8101/v1` (`UPSTREAM_HOST` defaults to `localhost`)

## Prerequisites

1. `uv` environment is ready in repo root.
2. Vertex project access + auth is available (`gcloud auth application-default login` or equivalent ADC).
3. GPT-OSS OpenAI-compatible endpoint is running on `http://<UPSTREAM_HOST>:8101/v1`.

Quick check:

```bash
UPSTREAM_HOST=${UPSTREAM_HOST:-localhost}
curl -sS "http://${UPSTREAM_HOST}:8101/v1/models"
```

## Start the server

```bash
ANTHROPIC_VERTEX_PROJECT_ID=<your-project> \
CLOUD_ML_REGION=us-east5 \
ANTHROPIC_MODEL=claude-sonnet-4-5@20250929 \
UPSTREAM_HOST=localhost \
uv run python -m experiments.v12.run_server
```

## Endpoint

- `POST http://127.0.0.1:8000/v1/chat/completions` with `model: "served-direct"` or `"served-adapter"`

## Example request (served-adapter)

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

## Probable output (served-adapter)

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "served-adapter",
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
    "prompt_tokens": 14,
    "completion_tokens": 4,
    "total_tokens": 18
  },
  "adapter_critic": {
    "mode": "adapter",
    "intermediate": {
      "api_draft": "pong",
      "adapter": {
        "decision": "lgtm"
      },
      "final": "pong",
      "adapter_rejection_reason": null
    },
    "tokens": {
      "stages": {
        "api": {
          "prompt_tokens": 10,
          "completion_tokens": 3,
          "total_tokens": 13
        },
        "adapter": {
          "prompt_tokens": 4,
          "completion_tokens": 1,
          "total_tokens": 5
        }
      },
      "total": {
        "prompt_tokens": 14,
        "completion_tokens": 4,
        "total_tokens": 18
      }
    }
  }
}
```

## Logs

- app log file: `logs/adapter_critic.log`
- default logging level in this experiment: `DEBUG`
