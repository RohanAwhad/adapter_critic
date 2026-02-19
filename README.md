## adapter_critic

FastAPI wrapper for OpenAI-compatible Chat Completions with three workflows:

- `direct`: API only
- `adapter`: API draft, then adapter search/replace refinement
- `critic`: API draft, critic feedback, second API pass

## Run

Build the app from code with startup routing config and your upstream gateway implementation:

```python
from adapter_critic.app import create_app
from adapter_critic.config import AppConfig

config = AppConfig.model_validate(
    {
        "served_models": {
            "served-direct": {
                "mode": "direct",
                "api": {"model": "gpt-4.1-mini", "base_url": "https://api.openai.com/v1"},
            },
            "served-adapter": {
                "mode": "adapter",
                "api": {"model": "gpt-4.1-mini", "base_url": "https://api.openai.com/v1"},
                "adapter": {"model": "gpt-4.1-mini", "base_url": "https://api.openai.com/v1"},
            },
            "served-critic": {
                "mode": "critic",
                "api": {"model": "gpt-4.1-mini", "base_url": "https://api.openai.com/v1"},
                "critic": {"model": "gpt-4.1-mini", "base_url": "https://api.openai.com/v1"},
            },
        }
    }
)
```

## Request examples

### direct (startup model routing)

```json
{
  "model": "served-direct",
  "messages": [{"role": "user", "content": "hello"}]
}
```

### adapter (request override)

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

### critic (request override)

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

Responses keep OpenAI fields and add:

- `adapter_critic.intermediate`
- `adapter_critic.tokens.stages`
- `adapter_critic.tokens.total`
