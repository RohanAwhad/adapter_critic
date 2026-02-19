# adapter_critic

FastAPI wrapper for OpenAI-compatible `/v1/chat/completions` with 3 workflows:

- `direct`: API only
- `adapter`: API draft -> adapter search/replace review -> final
- `critic`: API draft -> critic feedback -> second API pass -> final

## Built-in Gateway

This repo now ships with a built-in HTTP gateway (`src/adapter_critic/http_gateway.py`) that calls upstream OpenAI-compatible servers.

- Upstream URL is taken from per-stage `base_url` in your config
- Request path used upstream is `<base_url>/chat/completions`
- If `OPENAI_API_KEY` (or configured env var) is set, it sends `Authorization: Bearer ...`

## Quick Start

### 1) Install

```bash
uv sync --dev
```

### 2) Create `config.json`

```json
{
  "served_models": {
    "served-direct": {
      "mode": "direct",
      "api": {"model": "gpt-4.1-mini", "base_url": "https://api.openai.com/v1"}
    },
    "served-adapter": {
      "mode": "adapter",
      "api": {"model": "gpt-4.1-mini", "base_url": "https://api.openai.com/v1"},
      "adapter": {"model": "gpt-4.1-mini", "base_url": "https://api.openai.com/v1"}
    },
    "served-critic": {
      "mode": "critic",
      "api": {"model": "gpt-4.1-mini", "base_url": "https://api.openai.com/v1"},
      "critic": {"model": "gpt-4.1-mini", "base_url": "https://api.openai.com/v1"}
    }
  }
}
```

### 3) Start server (one command)

```bash
OPENAI_API_KEY=<your_key_if_needed> uv run adapter-critic-server --config config.json --host 0.0.0.0 --port 8000
```

Alternative command:

```bash
OPENAI_API_KEY=<your_key_if_needed> uv run python -m adapter_critic.server --config config.json
```

Server endpoint:

- `http://localhost:8000/v1/chat/completions`

## CLI Options

```bash
uv run adapter-critic-server --help
```

Key options:

- `--config` path to startup routing config JSON (default `config.json`)
- `--api-key-env` env var name for bearer token (default `OPENAI_API_KEY`)
- `--timeout-seconds` upstream request timeout (default `120.0`)
- `--reload` for local dev autoreload

## Request Overrides

Control overrides are accepted in either location:

- top-level `x_adapter_critic` (how OpenAI SDK `extra_body` arrives over the wire)
- nested `extra_body.x_adapter_critic` (raw JSON callers)

## Request Examples

### Direct (startup routing)

```json
{
  "model": "served-direct",
  "messages": [{"role": "user", "content": "hello"}]
}
```

### Adapter (request override)

```json
{
  "model": "served-direct",
  "messages": [{"role": "user", "content": "hello"}],
  "extra_body": {
    "x_adapter_critic": {
      "mode": "adapter",
      "adapter_model": "adapter-model",
      "adapter_base_url": "https://adapter.example/v1"
    }
  }
}
```

### Critic (request override)

```json
{
  "model": "served-direct",
  "messages": [{"role": "user", "content": "hello"}],
  "extra_body": {
    "x_adapter_critic": {
      "mode": "critic",
      "critic_model": "critic-model",
      "critic_base_url": "https://critic.example/v1"
    }
  }
}
```

## OpenAI SDK Example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
resp = client.chat.completions.create(
    model="served-direct",
    messages=[{"role": "user", "content": "hello"}],
    extra_body={"x_adapter_critic": {"mode": "adapter"}},
)

print(resp.choices[0].message.content)
print(resp.model_extra["adapter_critic"])
```

## Response Extension

Responses keep standard OpenAI fields and add:

- `adapter_critic.mode`
- `adapter_critic.intermediate`
- `adapter_critic.tokens.stages`
- `adapter_critic.tokens.total`

## Dev Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest -q
```
