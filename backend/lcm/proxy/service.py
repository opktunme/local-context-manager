from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..adapters import BackendConnectionError, adapter_for
from ..config import Settings
from ..context import ContextEngine
from ..storage import Database


class ProxyService:
    def __init__(
        self, database: Database, context_engine: ContextEngine, settings: Settings
    ) -> None:
        self.database = database
        self.context_engine = context_engine
        self.settings = settings

    def _active_adapter(self):
        backend = self.database.get_backend()
        if not backend:
            raise HTTPException(
                status_code=503,
                detail="No inference backend is configured. Open the LCM dashboard to finish setup.",
            )
        api_key = self.database.backend_secret(backend["id"])
        return backend, adapter_for(backend, api_key, self.settings.allow_private_lan)

    async def models(self, request: Request) -> Response:
        _, adapter = self._active_adapter()
        try:
            status, payload, headers = await adapter.list_models(self._incoming_headers(request))
        except BackendConnectionError as exc:
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"Local backend unavailable: {exc}",
                        "type": "lcm_backend_error",
                    }
                },
            )
        return JSONResponse(status_code=status, content=payload, headers=headers)

    async def chat_completions(self, request: Request, payload: dict[str, Any]) -> Response:
        backend, adapter = self._active_adapter()
        explicit_session_id = request.headers.get("X-LCM-Session-ID")
        try:
            transformed = self.context_engine.transform(payload, explicit_session_id, backend)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        started = time.perf_counter()
        response_headers = {
            "X-LCM-Session-ID": transformed.session_id,
            "X-LCM-Request-ID": transformed.request_id,
            "X-LCM-Incoming-Tokens": str(transformed.incoming_tokens),
            "X-LCM-Forwarded-Tokens": str(transformed.forwarded_tokens),
        }
        incoming_headers = self._incoming_headers(request)

        if not payload.get("stream", False):
            try:
                status, data, backend_headers = await adapter.request_json(
                    "/v1/chat/completions", transformed.forwarded, incoming_headers
                )
            except BackendConnectionError as exc:
                elapsed = (time.perf_counter() - started) * 1000
                self.database.finish_request(transformed.request_id, 502, None, elapsed)
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": {
                            "message": f"Local backend unavailable: {exc}",
                            "type": "lcm_backend_error",
                        }
                    },
                    headers=response_headers,
                )
            elapsed = (time.perf_counter() - started) * 1000
            metrics = adapter.extract_metrics(data if isinstance(data, dict) else {})
            self.database.finish_request(
                transformed.request_id,
                status,
                elapsed,
                elapsed,
                metrics.get("prompt_tokens"),
                metrics.get("completion_tokens"),
            )
            return JSONResponse(
                status_code=status,
                content=data,
                headers={**backend_headers, **response_headers},
            )

        client = httpx.AsyncClient(timeout=None, trust_env=False)
        try:
            backend_request = client.build_request(
                "POST",
                adapter.url("/v1/chat/completions"),
                json=transformed.forwarded,
                headers=adapter.headers(incoming_headers),
            )
            backend_response = await client.send(backend_request, stream=True)
        except httpx.HTTPError as exc:
            await client.aclose()
            elapsed = (time.perf_counter() - started) * 1000
            self.database.finish_request(transformed.request_id, 502, None, elapsed)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"Local backend unavailable: {exc}",
                        "type": "lcm_backend_error",
                    }
                },
                headers=response_headers,
            )

        if backend_response.status_code >= 400:
            body = await backend_response.aread()
            await backend_response.aclose()
            await client.aclose()
            elapsed = (time.perf_counter() - started) * 1000
            self.database.finish_request(
                transformed.request_id, backend_response.status_code, None, elapsed
            )
            headers = {
                **adapter.response_headers(backend_response.headers),
                **response_headers,
            }
            return Response(
                body,
                status_code=backend_response.status_code,
                headers=headers,
                media_type=backend_response.headers.get("content-type"),
            )

        first_byte_at: float | None = None

        async def relay() -> AsyncIterator[bytes]:
            nonlocal first_byte_at
            try:
                async for chunk in backend_response.aiter_raw():
                    if chunk:
                        if first_byte_at is None:
                            first_byte_at = time.perf_counter()
                        yield chunk
            finally:
                await backend_response.aclose()
                await client.aclose()
                ended = time.perf_counter()
                self.database.finish_request(
                    transformed.request_id,
                    backend_response.status_code,
                    (first_byte_at - started) * 1000 if first_byte_at else None,
                    (ended - started) * 1000,
                )

        media_type = backend_response.headers.get("content-type", "text/event-stream").split(
            ";", 1
        )[0]
        headers = {
            **adapter.response_headers(backend_response.headers),
            **response_headers,
        }
        return StreamingResponse(
            relay(),
            status_code=backend_response.status_code,
            headers=headers,
            media_type=media_type,
        )

    @staticmethod
    def _incoming_headers(request: Request) -> dict[str, str]:
        ignored = {"host", "content-length", "connection", "x-lcm-session-id"}
        return {
            key.lower(): value
            for key, value in request.headers.items()
            if key.lower() not in ignored
        }


def sse_usage_from_chunk(chunk: bytes) -> dict[str, int] | None:
    """Best-effort helper kept separate for future streamed usage aggregation."""
    try:
        text = chunk.decode("utf-8")
        data_lines = [
            line[5:].strip()
            for line in text.splitlines()
            if line.startswith("data:") and "[DONE]" not in line
        ]
        for line in reversed(data_lines):
            payload = json.loads(line)
            if payload.get("usage"):
                return payload["usage"]
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return None
