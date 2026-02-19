# adapter_critic

OpenAI-compatible FastAPI wrapper for `POST /v1/chat/completions` with 3 workflows:

- `direct`: API model only
- `adapter`: API draft -> adapter review/edit -> final
- `critic`: API draft -> critic feedback -> second API pass -> final

## Quick Start

### 1) Install

```bash
uv sync
```

### 2) Create `config.json`

You can configure per-served-model adapter/critic system prompts in startup config.

```json
{
  "served_models": {
    "served-direct": {
      "mode": "direct",
      "api": {"model": "gpt-4.1-mini", "base_url": "https://api.openai.com/v1"},
      "adapter_system_prompt": "You are a strict editor. Return lgtm or SEARCH/REPLACE blocks only.",
      "critic_system_prompt": "You are a critique generator. List issues and exact fix instructions."
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

### 3) Start server

```bash
OPENAI_API_KEY=<your_key_if_needed> uv run adapter-critic-server --config config.json --host 0.0.0.0 --port 8000
```

Alternative:

```bash
OPENAI_API_KEY=<your_key_if_needed> uv run python -m adapter_critic.server --config config.json
```

Endpoint:

- `http://localhost:8000/v1/chat/completions`

## Python script setup example

```python
from adapter_critic.app import create_app
from adapter_critic.config import AppConfig
from adapter_critic.http_gateway import OpenAICompatibleHttpGateway
from adapter_critic.runtime import build_runtime_state

config = AppConfig.model_validate(
    {
        "served_models": {
            "served-direct": {
                "mode": "direct",
                "api": {"model": "gpt-4.1-mini", "base_url": "https://api.openai.com/v1"},
                "adapter_system_prompt": "Custom adapter prompt",
                "critic_system_prompt": "Custom critic prompt",
            }
        }
    }
)

gateway = OpenAICompatibleHttpGateway(api_key="...")
state = build_runtime_state(config=config, gateway=gateway)
app = create_app(config=config, gateway=gateway, state=state)
```

## Request Contract

Required fields:

- `model: str`
- `messages: [{role, content}]`

Workflow controls can be sent in either location:

- top-level `x_adapter_critic`
- `extra_body.x_adapter_critic`

Override precedence:

- top-level `x_adapter_critic` wins over `extra_body.x_adapter_critic`

Supported override fields:

- `mode`: `direct | adapter | critic`
- `api_model`, `api_base_url`
- `adapter_model`, `adapter_base_url`
- `critic_model`, `critic_base_url`

Mode target fallback:

- if `mode=adapter` and no adapter target is configured/provided, adapter stage falls back to API target
- if `mode=critic` and no critic target is configured/provided, critic stage falls back to API target
- partial secondary overrides are rejected (example: model without base URL and no default target)

Invalid routing/overrides return `400` with `invalid model routing or overrides`.

## Mode Behavior

| Mode | Upstream call order | `adapter_critic.intermediate` keys | `adapter_critic.tokens.stages` keys |
| --- | --- | --- | --- |
| `direct` | API | `api` | `api` |
| `adapter` | API -> Adapter | `api_draft`, `adapter`, `final` | `api`, `adapter` |
| `critic` | API -> Critic -> API | `api_draft`, `critic`, `final` | `api_draft`, `critic`, `api_final` |

Adapter edit semantics:

- adapter returns `lgtm` to accept draft unchanged, or
- adapter returns one or more SEARCH/REPLACE blocks applied sequentially.

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

Response keeps normal OpenAI fields and adds:

- `adapter_critic.mode`
- `adapter_critic.intermediate`
- `adapter_critic.tokens.stages`
- `adapter_critic.tokens.total`

Invariant:

- `usage == adapter_critic.tokens.total`

## Current Boundaries

- non-streaming only (`stream=true` not implemented)
- text-only message content (`content: str`)
- built-in gateway expects OpenAI-style `choices[0].message.content` and `usage`
