from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from .contracts import AdapterCriticOverrides, Mode
from .prompts import ADAPTER_SYSTEM_PROMPT, CRITIC_SYSTEM_PROMPT


class StageTarget(BaseModel):
    model: str
    base_url: str
    api_key_env: str | None = Field(
        default=None,
        validation_alias=AliasChoices("api_key_env", "api_key_var"),
        serialization_alias="api_key_env",
    )


class ServedModelConfig(BaseModel):
    mode: Mode
    api: StageTarget
    adapter: StageTarget | None = None
    critic: StageTarget | None = None
    adapter_system_prompt: str | None = None
    critic_system_prompt: str | None = None


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    served_models: dict[str, ServedModelConfig]


class RuntimeConfig(BaseModel):
    served_model: str
    mode: Mode
    api: StageTarget
    adapter: StageTarget | None = None
    critic: StageTarget | None = None
    adapter_system_prompt: str
    critic_system_prompt: str


def _resolve_stage(base: StageTarget | None, model: str | None, base_url: str | None) -> StageTarget | None:
    if base is None and model is None and base_url is None:
        return None
    resolved_model = model if model is not None else (base.model if base is not None else None)
    resolved_base_url = base_url if base_url is not None else (base.base_url if base is not None else None)
    resolved_api_key_env = base.api_key_env if base is not None else None
    if resolved_model is None or resolved_base_url is None:
        return None
    return StageTarget(model=resolved_model, base_url=resolved_base_url, api_key_env=resolved_api_key_env)


def resolve_runtime_config(
    config: AppConfig, served_model: str, overrides: AdapterCriticOverrides
) -> RuntimeConfig | None:
    served = config.served_models.get(served_model)
    if served is None:
        return None

    mode: Mode = overrides.mode if overrides.mode is not None else served.mode
    api_target = _resolve_stage(served.api, overrides.api_model, overrides.api_base_url)
    if api_target is None:
        return None

    adapter_override_provided = overrides.adapter_model is not None or overrides.adapter_base_url is not None
    critic_override_provided = overrides.critic_model is not None or overrides.critic_base_url is not None

    adapter_target = _resolve_stage(served.adapter, overrides.adapter_model, overrides.adapter_base_url)
    critic_target = _resolve_stage(served.critic, overrides.critic_model, overrides.critic_base_url)

    if mode == "adapter" and adapter_target is None:
        if adapter_override_provided:
            return None
        adapter_target = api_target

    if mode == "critic" and critic_target is None:
        if critic_override_provided:
            return None
        critic_target = api_target

    return RuntimeConfig(
        served_model=served_model,
        mode=mode,
        api=api_target,
        adapter=adapter_target,
        critic=critic_target,
        adapter_system_prompt=(
            served.adapter_system_prompt if served.adapter_system_prompt is not None else ADAPTER_SYSTEM_PROMPT
        ),
        critic_system_prompt=(
            served.critic_system_prompt if served.critic_system_prompt is not None else CRITIC_SYSTEM_PROMPT
        ),
    )
