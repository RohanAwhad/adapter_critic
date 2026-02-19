from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request

from .config import AppConfig, resolve_runtime_config
from .contracts import parse_request_payload
from .dispatcher import dispatch
from .response_builder import build_response
from .runtime import RuntimeState, build_runtime_state
from .upstream import UpstreamGateway
from .usage import aggregate_usage


def create_app(
    config: AppConfig,
    gateway: UpstreamGateway,
    *,
    state: RuntimeState | None = None,
) -> FastAPI:
    runtime_state = state if state is not None else build_runtime_state(config=config, gateway=gateway)
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> dict[str, Any]:
        payload = await request.json()
        parsed = parse_request_payload(payload)
        runtime = resolve_runtime_config(runtime_state.config, parsed.request.model, parsed.overrides)
        if runtime is None:
            raise HTTPException(status_code=400, detail="invalid model routing or overrides")

        workflow_output = await dispatch(
            runtime=runtime, messages=parsed.request.messages, gateway=runtime_state.gateway
        )
        tokens = aggregate_usage(workflow_output.stage_usage)
        return build_response(
            parsed.request,
            mode=runtime.mode,
            final_text=workflow_output.final_text,
            intermediate=workflow_output.intermediate,
            tokens=tokens,
            response_id=runtime_state.id_provider(),
            created=runtime_state.time_provider(),
        )

    return app
