# adapter_critic

FastAPI wrapper for OpenAI-compatible Chat Completions with three workflows:

- `direct`: API only
- `adapter`: API draft -> adapter review/edit -> final
- `critic`: API draft -> critic feedback -> second API pass -> final

## How It Works

1. `POST /v1/chat/completions` is parsed in `src/adapter_critic/app.py`.
2. Routing config + request overrides are resolved in `src/adapter_critic/config.py`.
3. Workflow dispatch runs from `src/adapter_critic/dispatcher.py`:
   - `src/adapter_critic/workflows/direct.py`
   - `src/adapter_critic/workflows/adapter.py`
   - `src/adapter_critic/workflows/critic.py`
4. Response is OpenAI-compatible plus telemetry from `src/adapter_critic/response_builder.py`.

## Why Startup Felt Unclear

This repo is currently library-first. It has `create_app(...)`, but no built-in CLI server entrypoint or built-in upstream HTTP gateway.

Use the launcher below.

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

### 3) Create `run_server.py`

```python
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import uvicorn

from adapter_critic.app import create_app
from adapter_critic.config import AppConfig
from adapter_critic.contracts import ChatMessage
from adapter_critic.upstream import TokenUsage, UpstreamResult


class HttpGateway:
    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key

    async def complete(self, *, model: str, base_url: str, messages: list[ChatMessage]) -> UpstreamResult:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": model,
            "messages": [message.model_dump() for message in messages],
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        usage = data.get("usage", {})
        content = data["choices"][0]["message"].get("content") or ""
        return UpstreamResult(
            content=content,
            usage=TokenUsage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
            ),
        )


def main() -> None:
    config_data = json.loads(Path("config.json").read_text())
    config = AppConfig.model_validate(config_data)
    gateway = HttpGateway(api_key=os.environ.get("OPENAI_API_KEY"))
    app = create_app(config=config, gateway=gateway)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
```

### 4) Start

```bash
OPENAI_API_KEY=<your_key_if_needed> uv run python run_server.py
```

Server endpoint:

- `http://localhost:8000/v1/chat/completions`

## Request Examples

Control overrides are accepted in either location:

- top-level `x_adapter_critic` (how OpenAI SDK `extra_body` arrives on wire)
- nested `extra_body.x_adapter_critic` (for direct raw JSON requests)

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
