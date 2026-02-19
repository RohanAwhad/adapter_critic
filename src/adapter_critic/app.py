from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from loguru import logger
from starlette.types import Message

from .config import AppConfig, resolve_runtime_config
from .contracts import parse_request_payload
from .dispatcher import dispatch
from .http_gateway import UpstreamResponseFormatError
from .logging_setup import is_debug_logging_enabled
from .response_builder import build_response
from .runtime import RuntimeState, build_runtime_state
from .upstream import UpstreamGateway
from .usage import aggregate_usage


def _body_preview(body: bytes, *, max_chars: int = 2000) -> str:
    text = body.decode("utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def create_app(
    config: AppConfig,
    gateway: UpstreamGateway,
    *,
    state: RuntimeState | None = None,
) -> FastAPI:
    runtime_state = state if state is not None else build_runtime_state(config=config, gateway=gateway)
    app = FastAPI()

    @app.middleware("http")
    async def debug_request_response_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not is_debug_logging_enabled():
            return await call_next(request)

        request_body = await request.body()
        logger.debug(
            "incoming request method={} path={} query={} body={}",
            request.method,
            request.url.path,
            request.url.query,
            _body_preview(request_body),
        )

        async def receive() -> Message:
            return {"type": "http.request", "body": request_body, "more_body": False}

        request_with_body = Request(request.scope, receive)
        response = await call_next(request_with_body)

        response_body = b""
        body_iterator = getattr(response, "body_iterator", None)
        if body_iterator is not None:
            async for chunk in body_iterator:
                response_body += chunk
        else:
            response_body = bytes(response.body)

        logger.debug(
            "outgoing response method={} path={} status_code={} body={}",
            request.method,
            request.url.path,
            response.status_code,
            _body_preview(response_body),
        )

        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
            background=response.background,
        )

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
                "upstream response format error model={} base_url={} "
                "message_count={} status_code={} reason={} payload={}",
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
