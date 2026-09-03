from __future__ import annotations

import asyncio
import struct

from fastapi import FastAPI
from fastapi.testclient import TestClient
from lcm.api import create_api_router
from lcm.context import ContextEngine
from lcm.proxy import ProxyService
from lcm.workbench import PROFILES, LongCatRuntime, runtime_context_capacity, runtime_device_slots


def test_workbench_chat_history_persists(database) -> None:
    chat = database.create_workbench_chat({"profile": "full", "context_limit": 192})
    database.add_workbench_message(chat["id"], "user", "Hello", 2)
    database.add_workbench_message(
        chat["id"], "assistant", "Hi there", 3, {"throughput_tps": 17.5}
    )

    loaded = database.get_workbench_chat(chat["id"])

    assert loaded is not None
    assert loaded["profile"] == "full"
    assert loaded["context_limit"] == 192
    assert [message["role"] for message in loaded["messages"]] == ["user", "assistant"]
    assert loaded["messages"][1]["metrics"]["throughput_tps"] == 17.5


def test_longcat_output_parser_uses_native_metrics() -> None:
    output = """RAM cache top-off: 3584 records, 16.7344 GiB cache, 6.15 s
OK
model: LongCat-Flash-Lite-Sparse
precision: Q4G64T experts, Q8 attention
active MoE routes: 12 / 12
RAM budget/actual GiB: 24 / 16.7344 (3584 expert records)
peak device allocation estimate GiB: 9.65917
decode throughput: 14.6201 tok/s
token ids: 7793 2
"""

    content, metrics = LongCatRuntime.parse_output(output)

    assert content == "OK"
    assert metrics["active_routes"] == 12
    assert metrics["ram_actual_gib"] == 16.7344
    assert metrics["vram_actual_gib"] == 9.65917
    assert metrics["throughput_tps"] == 14.6201
    assert metrics["token_ids"] == [7793, 2]


def test_longcat_runtime_fits_recent_messages_without_mutating_history(
    local_settings, monkeypatch
) -> None:
    runtime = LongCatRuntime(local_settings)

    async def fake_tokenize(_profile_id: str, prompt: str) -> int:
        return len(prompt.split()) + 2

    monkeypatch.setattr(runtime, "tokenize", fake_tokenize)
    messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": " ".join(["long"] * 80)},
        {"role": "user", "content": "hi"},
    ]

    fitted, prompt_tokens, dropped, compacted = asyncio.run(
        runtime.fit_messages("exact", messages, 30)
    )

    assert fitted == [{"role": "user", "content": "hi"}]
    assert prompt_tokens <= 30
    assert dropped == 2
    assert compacted is False
    assert len(messages) == 3


def test_longcat_structured_transcript_preserves_roles_and_utf8(
    local_settings, tmp_path
) -> None:
    runtime = LongCatRuntime(local_settings)
    path = tmp_path / "chat.bin"
    messages = [
        {"role": "user", "content": "Remember PURPLE."},
        {"role": "assistant", "content": "Understood — مرحبا"},
        {"role": "user", "content": "What word?"},
    ]

    runtime.write_chat_transcript(path, messages)
    data = path.read_bytes()

    assert data[:8] == b"XTCHAT1\0"
    assert struct.unpack_from("<I", data, 8)[0] == 3
    assert "مرحبا".encode() in data


def test_longcat_normalizes_failed_adjacent_user_turns(local_settings) -> None:
    runtime = LongCatRuntime(local_settings)

    normalized = runtime.normalize_messages(
        [
            {"role": "user", "content": "stale failed request"},
            {"role": "user", "content": "new request"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "latest"},
        ]
    )

    assert normalized == [
        {"role": "user", "content": "new request"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "latest"},
    ]


def test_longcat_runtime_trades_device_slots_for_context() -> None:
    profile = PROFILES["full"]
    capacity = runtime_context_capacity(profile, 11.0)
    slots = runtime_device_slots(profile, 11.0, 499)

    assert capacity > 256
    assert capacity < profile.context_limit
    assert 12 <= slots < profile.max_device_slots


def test_workbench_api_creates_and_updates_chat(database, local_settings) -> None:
    engine = ContextEngine(database)
    proxy = ProxyService(database, engine, local_settings)
    app = FastAPI()
    app.include_router(create_api_router(database, engine, proxy, local_settings))
    client = TestClient(app)

    created = client.post(
        "/api/workbench/chats", json={"profile": "exact", "output_token_limit": 64}
    )
    assert created.status_code == 200
    chat = created.json()
    assert chat["profile"] == "exact"
    assert chat["output_token_limit"] == 64
    assert chat["context_ram_budget_gib"] == 0.1

    updated = client.patch(
        f"/api/workbench/chats/{chat['id']}",
        json={"profile": "full", "auto_mode": True, "cleanup_mode": "aggressive"},
    )
    assert updated.status_code == 200
    assert updated.json()["profile"] == "full"
    assert updated.json()["auto_mode"] is True
    assert updated.json()["vram_budget_gib"] == 10.3802

    listing = client.get("/api/workbench/chats").json()["chats"]
    assert listing[0]["id"] == chat["id"]


def test_workbench_exposes_and_enforces_shared_memory_envelope(
    database, local_settings
) -> None:
    engine = ContextEngine(database)
    proxy = ProxyService(database, engine, local_settings)
    app = FastAPI()
    app.include_router(create_api_router(database, engine, proxy, local_settings))
    client = TestClient(app)

    config = client.get("/api/workbench/config").json()
    assert config["limits"]["total_ram_gib"] == 70
    assert config["limits"]["max_vram_gib"] == 11.5
    assert config["limits"]["context_tokens"] == 983_040
    assert config["limits"]["executor_context_tokens"] == 7_445

    chat = client.post("/api/workbench/chats", json={}).json()
    expanded = client.patch(
        f"/api/workbench/chats/{chat['id']}",
        json={
            "ram_budget_gib": 40,
            "context_ram_budget_gib": 29.6,
            "context_limit": 983_040,
            "vram_budget_gib": 11.5,
        },
    )
    assert expanded.status_code == 200
    assert expanded.json()["context_limit"] == 983_040

    over_budget = client.patch(
        f"/api/workbench/chats/{chat['id']}",
        json={"ram_budget_gib": 41, "context_ram_budget_gib": 30},
    )
    assert over_budget.status_code == 422
    assert "70 GiB" in over_budget.json()["detail"]
