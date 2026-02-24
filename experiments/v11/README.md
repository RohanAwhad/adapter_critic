# V11 Experiment: served-adapter with ITS Hub (best-of-n + gpt-oss-120b judge)

## What this experiment is

`experiments/v11/` runs `adapter_critic` with the adapter stage routed through
`its_hub` IaaS (inference-time scaling). The IaaS endpoint generates multiple
adapter candidates via best-of-n using `qwen3-30b-a3b-thinking`, then scores
them with `gpt-oss-120b` as the LLM judge.

## Architecture

```
Client
  └─► adapter_critic (:8000)
        ├─ API stage ──────► qwen3-30b-a3b-thinking (:8100)  (draft generation)
        └─ Adapter stage ──► its_hub (:8108)
                               ├─ generation ──► qwen3-30b-a3b-thinking (:8100)
                               └─ judge ───────► gpt-oss-120b (:8101)
```

## ITS Hub configuration (hardcoded)

- Algorithm: `best-of-n`
- Budget: `8` (default per-request)
- Reward model: `llm-judge` (groupwise, `overall_quality` criterion)
- Judge model: `gpt-oss-120b` on port `8101`
- Judge max tokens: `4096`

## Prerequisites

1. `uv` environment is ready in repo root.
2. `qwen3-30b-a3b-thinking` running on port `8100`.
3. `gpt-oss-120b` running on port `8101`.

Quick check:

```bash
curl -sS "http://localhost:8100/v1/models"
curl -sS "http://localhost:8101/v1/models"
```

## Start the server

```bash
uv run python -m experiments.v11.run_server
```

Or with a remote upstream host:

```bash
UPSTREAM_HOST=host.docker.internal uv run python -m experiments.v11.run_server
```

## Endpoints

- `POST http://127.0.0.1:8000/v1/chat/completions` with `model: "served-adapter"` or `"served-direct"`

## Example requests

### served-adapter (with ITS Hub best-of-n + 120b judge)

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "served-adapter",
    "messages": [{"role": "user", "content": "Explain the difference between TCP and UDP."}]
  }'
```

### served-direct (baseline, no adapter)

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "served-direct",
    "messages": [{"role": "user", "content": "Explain the difference between TCP and UDP."}]
  }'
```

## What to inspect

- `adapter_critic.intermediate.adapter`: adapter model's structured JSON output
- `adapter_critic.intermediate.adapter_rejection_reason`: `null` when adapter succeeded
- `choices[0].message.content`: final response text
- `adapter_critic.tokens`: token usage breakdown by stage

## Logs

- App log file: `logs/adapter_critic.log`
- Default logging level: `DEBUG`
