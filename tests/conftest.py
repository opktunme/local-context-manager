from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from lcm.config import Settings
from lcm.context import ContextEngine
from lcm.storage import Database


@pytest.fixture
def database(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.sqlite3", request_retention=10)


@pytest.fixture
def engine(database: Database) -> ContextEngine:
    return ContextEngine(database)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def mock_backend() -> Iterator[dict[str, Any]]:
    app = FastAPI()
    captured: list[dict[str, Any]] = []

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "mock-local-model", "object": "model"}]}

    @app.get("/api/tags")
    async def tags() -> dict[str, Any]:
        return {"models": [{"name": "mock-local-model:latest"}]}

    @app.post("/v1/chat/completions")
    async def chat(payload: dict[str, Any]):
        captured.append(payload)
        if payload.get("stream"):

            async def events():
                yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                time.sleep(0.01)
                yield b'data: {"choices":[{"delta":{"content":" local"}}]}\n\n'
                yield b"data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello local"}}],
            "usage": {"prompt_tokens": 17, "completion_tokens": 2, "total_tokens": 19},
        }

    port = free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 5
    while not server.started and time.time() < deadline:
        time.sleep(0.01)
    if not server.started:
        raise RuntimeError("Mock backend failed to start")
    try:
        yield {"url": f"http://127.0.0.1:{port}", "captured": captured, "port": port}
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture
def local_settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, request_retention=10)
