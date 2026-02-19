from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from .config import AppConfig, resolve_runtime_config
from .contracts import parse_request_payload
from .dispatcher import dispatch
from .http_gateway import UpstreamResponseFormatError
from .response_builder import build_response
from .runtime import RuntimeState, build_runtime_state
from .upstream import UpstreamGateway
from .usage import aggregate_usage

logger = logging.getLogger(__name__)


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

        try:
            workflow_output = await dispatch(
                runtime=runtime, messages=parsed.request.messages, gateway=runtime_state.gateway
            )
        except UpstreamResponseFormatError as exc:
            logger.error(
                "upstream response format error model=%s base_url=%s "
                "message_count=%s status_code=%s reason=%s payload=%s",
                exc.model,
                exc.base_url,
                exc.message_count,
                exc.status_code,
                exc.reason,
                exc.payload_preview,
            )
            raise HTTPException(status_code=502, detail="upstream returned non-OpenAI response shape") from exc

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
