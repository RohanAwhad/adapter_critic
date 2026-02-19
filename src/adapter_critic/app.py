from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request

from .config import AppConfig, resolve_runtime_config
from .contracts import parse_request_payload
from .dispatcher import dispatch
from .response_builder import build_response
from .upstream import UpstreamGateway
from .usage import aggregate_usage


def create_app(config: AppConfig, gateway: UpstreamGateway) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> dict[str, Any]:
        payload = await request.json()
        parsed = parse_request_payload(payload)
        runtime = resolve_runtime_config(config, parsed.request.model, parsed.overrides)
        if runtime is None:
            raise HTTPException(status_code=400, detail="invalid model routing or overrides")

        workflow_output = await dispatch(runtime=runtime, messages=parsed.request.messages, gateway=gateway)
        tokens = aggregate_usage(workflow_output.stage_usage)
        return build_response(
            parsed.request,
            mode=runtime.mode,
            final_text=workflow_output.final_text,
            intermediate=workflow_output.intermediate,
            tokens=tokens,
        )

    return app
