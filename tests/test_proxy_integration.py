from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from lcm.adapters import adapter_for
from lcm.api import create_api_router
from lcm.context import ContextEngine
from lcm.proxy import ProxyService
from lcm.storage import Database


def make_app(database: Database, engine: ContextEngine, settings) -> TestClient:
    app = FastAPI()
    app.include_router(
        create_api_router(database, engine, ProxyService(database, engine, settings), settings)
    )
    return TestClient(app)


def configure(database: Database, url: str, kind: str = "openai"):
    return database.configure_backend(
        {"name": "Mock local", "kind": kind, "base_url": url, "context_limit": 32768}
    )


def test_overlay_state_is_compact_and_color_ready(database, engine, local_settings):
    result = engine.transform(
        {
            "model": "mock-local-model",
            "messages": [
                {"role": "system", "content": "stay local"},
                {"role": "assistant", "content": "```python\nprint('pixel')\n```"},
                {"role": "user", "content": "show the live window"},
            ],
        },
        "overlay-state",
        None,
    )
    client = make_app(database, engine, local_settings)
    response = client.get(
        "/api/overlay-state", params={"session_id": result.session_id}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["active"] is True
    assert body["metrics"]["active_tokens"] > 0
    assert {segment["category"] for segment in body["segments"]} >= {
        "System",
        "Conversation",
        "Code / files",
    }
    assert all("text" not in segment and "raw_message" not in segment for segment in body["segments"])


def test_models_and_non_stream_passthrough(database, engine, local_settings, mock_backend):
    configure(database, mock_backend["url"])
    client = make_app(database, engine, local_settings)
    models = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "mock-local-model"

    request = {
        "model": "mock-local-model",
        "stream": False,
        "temperature": 0.31,
        "custom_passthrough": {"preserved": True},
        "messages": [
            {"role": "system", "content": "stay local"},
            {"role": "user", "content": "hello"},
        ],
    }
    response = client.post(
        "/v1/chat/completions", json=request, headers={"X-LCM-Session-ID": "integration"}
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello local"
    assert response.headers["X-LCM-Session-ID"].startswith("explicit-")
    assert mock_backend["captured"][-1]["custom_passthrough"] == {"preserved": True}
    detail = database.session_detail(response.headers["X-LCM-Session-ID"])
    assert detail and detail["requests"][0]["status_code"] == 200
    assert detail["requests"][0]["prompt_tokens"] == 17


def test_sse_streaming_is_relayed(database, engine, local_settings, mock_backend):
    configure(database, mock_backend["url"])
    client = make_app(database, engine, local_settings)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "mock-local-model",
            "stream": True,
            "messages": [{"role": "user", "content": "stream"}],
        },
    ) as response:
        body = b"".join(response.iter_raw())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert b'"hello"' in body and b'" local"' in body and b"[DONE]" in body


def test_removal_changes_actual_backend_payload(database, engine, local_settings, mock_backend):
    configure(database, mock_backend["url"])
    client = make_app(database, engine, local_settings)
    messages = [
        {"role": "assistant", "content": "remove me"},
        {"role": "user", "content": "current"},
    ]
    first = client.post(
        "/v1/chat/completions",
        json={"model": "mock-local-model", "messages": messages},
        headers={"X-LCM-Session-ID": "overlay"},
    )
    session_id = first.headers["X-LCM-Session-ID"]
    detail = client.get(f"/api/sessions/{session_id}").json()
    old_id = detail["items"][0]["item_id"]
    applied = client.post(
        f"/api/sessions/{session_id}/actions", json={"action": "remove", "item_ids": [old_id]}
    )
    assert applied.status_code == 200
    client.post(
        "/v1/chat/completions",
        json={"model": "mock-local-model", "messages": messages},
        headers={"X-LCM-Session-ID": "overlay"},
    )
    assert {"role": "assistant", "content": "remove me"} not in mock_backend["captured"][-1][
        "messages"
    ]


def test_api_expands_tool_dependencies_atomically(database, engine, local_settings, mock_backend):
    configure(database, mock_backend["url"])
    client = make_app(database, engine, local_settings)
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-linked",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-linked", "content": "obsolete file"},
        {"role": "user", "content": "continue"},
    ]
    first = client.post(
        "/v1/chat/completions",
        json={"model": "mock-local-model", "messages": messages},
        headers={"X-LCM-Session-ID": "tool-overlay"},
    )
    session_id = first.headers["X-LCM-Session-ID"]
    detail = client.get(f"/api/sessions/{session_id}").json()
    result_id = next(item["item_id"] for item in detail["items"] if item["role"] == "tool")
    client.post(
        f"/api/sessions/{session_id}/actions",
        json={"action": "remove", "item_ids": [result_id]},
    )
    changed = client.get(f"/api/sessions/{session_id}").json()
    linked = [item for item in changed["items"] if item["role"] in {"assistant", "tool"}]
    assert {item["status"] for item in linked} == {"removed"}


def test_backend_failure_and_malformed_requests(database, engine, local_settings):
    configure(database, "http://127.0.0.1:9")
    client = make_app(database, engine, local_settings)
    failure = client.post(
        "/v1/chat/completions",
        json={"model": "mock", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert failure.status_code == 502
    malformed = client.post(
        "/v1/chat/completions", json={"model": "mock", "messages": "not-an-array"}
    )
    assert malformed.status_code == 422


@pytest.mark.asyncio
async def test_ollama_adapter_native_capability(database, engine, local_settings, mock_backend):
    backend = configure(database, mock_backend["url"], kind="ollama")
    adapter = adapter_for(backend)
    assert adapter.kind == "ollama"
    assert (await adapter.native_details())["models"][0]["name"] == "mock-local-model:latest"
    client = make_app(database, engine, local_settings)
    assert (
        client.post(
            "/api/backends/test",
            json={
                "name": "Ollama",
                "kind": "ollama",
                "base_url": mock_backend["url"],
                "context_limit": 32768,
            },
        ).json()["ok"]
        is True
    )


def test_public_backend_urls_are_rejected(database, engine, local_settings):
    client = make_app(database, engine, local_settings)
    result = client.post(
        "/api/backends",
        json={
            "name": "Cloud",
            "kind": "openai",
            "base_url": "https://api.example.com",
            "context_limit": 32768,
        },
    )
    assert result.status_code == 422
    assert "local" in result.json()["detail"].lower()
