from __future__ import annotations

import asyncio
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from .adapters import ADAPTER_CATALOG, BackendConnectionError, adapter_for
from .config import PROJECT_ROOT, Settings
from .context import ContextEngine
from .metrics import estimate_kv_cache
from .proxy import ProxyService
from .storage import Database
from .workbench import (
    CONTEXT_BYTES_PER_TOKEN,
    DEVICE_CONTEXT_BYTES_PER_TOKEN,
    DEVICE_EXPERT_SLOT_BYTES,
    MAX_VRAM_BUDGET_GIB,
    MIN_DEVICE_SLOTS,
    MODEL_CONTEXT_LIMIT_TOKENS,
    PROFILES,
    TOTAL_RAM_BUDGET_GIB,
    VRAM_SAFETY_GIB,
    LongCatRuntime,
    context_ram_gib,
    host_ram_gib,
)


class BackendInput(BaseModel):
    id: str | None = None
    name: str = "Local backend"
    kind: str = "openai"
    base_url: str
    api_key: str | None = None
    active: bool = True
    context_limit: int = Field(default=32768, ge=512, le=10_000_000)
    architecture: dict[str, Any] | None = None


class ActionInput(BaseModel):
    action: str
    item_ids: list[str] = []
    summary: str | None = None


class SummaryInput(BaseModel):
    item_ids: list[str]
    mode: str = "zero_compute"


class RenameInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class PolicyInput(BaseModel):
    context_limit: int | None = Field(default=None, ge=512, le=10_000_000)
    context_budget: int | None = Field(default=None, ge=256, le=10_000_000)
    tokenizer_family: str | None = Field(default=None, max_length=40)
    auto_mode: bool | None = None
    auto_cleanup_mode: Literal["conservative", "balanced", "aggressive"] | None = None


class AutoCleanupInput(BaseModel):
    mode: Literal["conservative", "balanced", "aggressive"]


class OverlayLaunchInput(BaseModel):
    session_id: str | None = None


class WorkbenchChatInput(BaseModel):
    title: str | None = Field(default=None, max_length=100)
    profile: Literal["exact", "full"] = "exact"
    ram_budget_gib: float | None = Field(default=None, ge=4, le=TOTAL_RAM_BUDGET_GIB)
    context_ram_budget_gib: float = Field(default=0.1, ge=0.01, le=66)
    vram_budget_gib: float | None = Field(default=None, ge=4, le=MAX_VRAM_BUDGET_GIB)
    context_limit: int = Field(default=256, ge=64, le=MODEL_CONTEXT_LIMIT_TOKENS)
    output_token_limit: int = Field(default=96, ge=1, le=220)
    auto_mode: bool = False
    cleanup_mode: Literal["conservative", "balanced", "aggressive"] = "balanced"


class WorkbenchChatPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=100)
    profile: Literal["exact", "full"] | None = None
    ram_budget_gib: float | None = Field(default=None, ge=4, le=TOTAL_RAM_BUDGET_GIB)
    context_ram_budget_gib: float | None = Field(default=None, ge=0.01, le=66)
    vram_budget_gib: float | None = Field(default=None, ge=4, le=MAX_VRAM_BUDGET_GIB)
    context_limit: int | None = Field(default=None, ge=64, le=MODEL_CONTEXT_LIMIT_TOKENS)
    output_token_limit: int | None = Field(default=None, ge=1, le=220)
    auto_mode: bool | None = None
    cleanup_mode: Literal["conservative", "balanced", "aggressive"] | None = None
    archived: bool | None = None


class WorkbenchMessageInput(BaseModel):
    content: str = Field(min_length=1, max_length=16000)


def _validate_workbench_resources(data: dict[str, Any]) -> None:
    context_tokens = int(data["context_limit"])
    context_ram = float(data["context_ram_budget_gib"])
    output_tokens = int(data["output_token_limit"])
    minimum_context_ram = context_ram_gib(context_tokens)
    output_ram = context_ram_gib(output_tokens)
    if context_ram + 1e-9 < minimum_context_ram:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{context_tokens:,} context tokens need at least "
                f"{minimum_context_ram:.2f} GiB of context RAM"
            ),
        )
    total = float(data["ram_budget_gib"]) + context_ram
    if total > TOTAL_RAM_BUDGET_GIB + 1e-9:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Inference/cache and context (including {output_ram:.3f} GiB output reserve) "
                f"total {total:.2f} GiB; "
                f"the shared RAM envelope is {TOTAL_RAM_BUDGET_GIB:.0f} GiB"
            ),
        )


def create_api_router(
    database: Database, engine: ContextEngine, proxy: ProxyService, settings: Settings
) -> APIRouter:
    router = APIRouter()
    longcat = LongCatRuntime(settings)

    @router.get("/healthz")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service": "Local Context Manager", "local_only": True}

    @router.get("/api/status")
    async def status() -> dict[str, Any]:
        backend = database.get_backend()
        sessions = [
            _session_summary(database, entry["id"]) for entry in database.recent_sessions(30)
        ]
        return {
            "configured": backend is not None,
            "backend": backend,
            "backends": database.list_backends(),
            "sessions": [entry for entry in sessions if entry],
            "adapter_catalog": ADAPTER_CATALOG,
            "proxy_url": f"http://{settings.host}:{settings.port}/v1",
            "dashboard_url": f"http://{settings.host}:{settings.port}",
            "privacy": {
                "cloud_requests": False,
                "telemetry": False,
                "accounts": False,
                "outbound_policy": "localhost/private network backends only",
            },
        }

    @router.post("/api/backends")
    async def configure_backend(value: BackendInput) -> dict[str, Any]:
        data = value.model_dump()
        if value.kind not in {entry["kind"] for entry in ADAPTER_CATALOG}:
            raise HTTPException(status_code=422, detail="Unknown backend adapter kind")
        try:
            adapter_for(
                {**data, "id": value.id or "validation"},
                value.api_key,
                settings.allow_private_lan,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        saved = database.configure_backend(data)
        return {
            "backend": saved,
            "proxy_url": f"http://{settings.host}:{settings.port}/v1",
        }

    @router.post("/api/backends/test")
    async def test_backend(value: BackendInput) -> dict[str, Any]:
        data = value.model_dump()
        try:
            adapter = adapter_for(
                {**data, "id": value.id or "test"},
                value.api_key,
                settings.allow_private_lan,
            )
            status_code, payload, _ = await adapter.list_models()
        except (ValueError, BackendConnectionError) as exc:
            return {"ok": False, "message": str(exc), "models": []}
        models = payload.get("data", []) if isinstance(payload, dict) else []
        return {
            "ok": 200 <= status_code < 300,
            "status_code": status_code,
            "models": models,
            "message": "Connection successful"
            if status_code < 300
            else "Backend rejected the connection",
        }

    @router.get("/api/sessions")
    async def list_sessions() -> dict[str, Any]:
        sessions = [
            _session_summary(database, entry["id"]) for entry in database.recent_sessions(100)
        ]
        return {"sessions": [entry for entry in sessions if entry]}

    @router.get("/api/sessions/{session_id}")
    async def session_detail(session_id: str) -> dict[str, Any]:
        detail = _enriched_session(database, session_id)
        if not detail:
            raise HTTPException(status_code=404, detail="Session not found")
        return detail

    @router.patch("/api/sessions/{session_id}")
    async def rename_session(session_id: str, value: RenameInput) -> dict[str, Any]:
        if not database.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        database.rename_session(session_id, value.name)
        return _enriched_session(database, session_id) or {}

    @router.patch("/api/sessions/{session_id}/policy")
    async def update_policy(session_id: str, value: PolicyInput) -> dict[str, Any]:
        if not database.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        data = {key: val for key, val in value.model_dump().items() if val is not None}
        if (
            data.get("context_budget")
            and data.get("context_limit")
            and data["context_budget"] > data["context_limit"]
        ):
            raise HTTPException(
                status_code=422, detail="Context budget cannot exceed the context limit"
            )
        database.update_session_policy(session_id, data)
        return _enriched_session(database, session_id) or {}

    @router.post("/api/sessions/{session_id}/actions")
    async def apply_action(session_id: str, value: ActionInput) -> dict[str, Any]:
        detail = database.session_detail(session_id)
        if not detail:
            raise HTTPException(status_code=404, detail="Session not found")
        item_ids = (
            _expand_dependencies(detail["items"], value.item_ids)
            if value.action in {"remove", "summarize"}
            else value.item_ids
        )
        summary_tokens = (
            engine.tokenizer.count_text(value.summary or "", detail.get("model")).value
            if value.action == "summarize"
            else 0
        )
        try:
            result = database.apply_action(
                session_id, value.action, item_ids, value.summary, summary_tokens
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"result": result, "session": _enriched_session(database, session_id)}

    @router.post("/api/sessions/{session_id}/auto-cleanup")
    async def run_auto_cleanup(
        session_id: str, value: AutoCleanupInput
    ) -> dict[str, Any]:
        if not database.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        result = engine.cleanup_session(session_id, value.mode, actor="manual")
        return {"result": result, "session": _enriched_session(database, session_id)}

    @router.post("/api/sessions/{session_id}/undo")
    async def undo_action(session_id: str) -> dict[str, Any]:
        result = database.undo_last_action(session_id)
        if not result:
            raise HTTPException(status_code=409, detail="There is no reversible action to undo")
        return {"result": result, "session": _enriched_session(database, session_id)}

    @router.post("/api/sessions/{session_id}/summarize-preview")
    async def summarize_preview(session_id: str, value: SummaryInput) -> dict[str, Any]:
        detail = database.session_detail(session_id)
        if not detail:
            raise HTTPException(status_code=404, detail="Session not found")
        item_ids = _expand_dependencies(detail["items"], value.item_ids)
        selected = [item for item in detail["items"] if item["item_id"] in item_ids]
        if not selected:
            raise HTTPException(status_code=422, detail="Select at least one context item")
        draft = engine.draft_summary(selected)
        used_mode = "zero_compute"
        if value.mode == "llm":
            backend = database.get_backend(detail.get("backend_id"))
            if not backend:
                raise HTTPException(
                    status_code=409,
                    detail="Configure a local backend before using LLM-assisted summaries",
                )
            adapter = adapter_for(
                backend,
                database.backend_secret(backend["id"]),
                settings.allow_private_lan,
            )
            prompt = "\n\n".join(item["text"] for item in selected)
            summary_request = {
                "model": detail.get("model"),
                "stream": False,
                "temperature": 0.1,
                "messages": [
                    {
                        "role": "system",
                        "content": "Summarize this context compactly. Preserve requirements, decisions, identifiers, current state, constraints, and unresolved work. Return only the summary.",
                    },
                    {"role": "user", "content": prompt},
                ],
            }
            try:
                status_code, result, _ = await adapter.request_json(
                    "/v1/chat/completions", summary_request
                )
                if status_code >= 400:
                    raise HTTPException(
                        status_code=502,
                        detail="The local model could not generate a summary",
                    )
                draft = result["choices"][0]["message"]["content"]
                used_mode = "local_llm"
            except (BackendConnectionError, KeyError, IndexError, TypeError) as exc:
                raise HTTPException(
                    status_code=502, detail=f"Local summarization failed: {exc}"
                ) from exc
        original_tokens = sum(item["token_count"] for item in selected)
        summary_tokens = engine.tokenizer.count_text(draft, detail.get("model")).value
        return {
            "summary": draft,
            "mode": used_mode,
            "item_ids": item_ids,
            "original_tokens": original_tokens,
            "summary_tokens": summary_tokens,
            "estimated_saving": max(0, original_tokens - summary_tokens),
        }

    @router.delete("/api/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str) -> Response:
        database.delete_session(session_id)
        return Response(status_code=204)

    @router.delete("/api/data", status_code=204)
    async def delete_all_data() -> Response:
        database.delete_all_local_data()
        return Response(status_code=204)

    @router.post("/api/demo")
    async def create_demo() -> dict[str, Any]:
        payload = _demo_payload()
        result = engine.transform(payload, f"demo-{uuid.uuid4().hex[:10]}", None)
        database.rename_session(result.session_id, "Demo · Debugging a local build")
        items = database.get_items(result.session_id)
        pin_ids = [
            item["item_id"]
            for item in items
            if item["kind"] in {"system_instruction", "developer_instruction"}
        ]
        if pin_ids:
            database.apply_action(result.session_id, "pin", pin_ids)
        return _enriched_session(database, result.session_id) or {}

    @router.get("/api/system")
    async def system_information() -> dict[str, Any]:
        return {
            "os": platform.platform(),
            "processor": platform.processor() or "Unavailable",
            "cpu_count": os.cpu_count(),
            "gpu": "Unavailable without an optional platform-specific provider",
            "measured": True,
        }

    @router.get("/api/overlay-state")
    async def overlay_state(session_id: str | None = None) -> dict[str, Any]:
        target_id = session_id
        if not target_id:
            recent = database.recent_sessions(1)
            target_id = recent[0]["id"] if recent else None
        if not target_id:
            return {"active": False, "message": "Waiting for the first context request"}
        detail = _enriched_session(database, target_id)
        if not detail:
            raise HTTPException(status_code=404, detail="Session not found")
        segments = [
            {
                "id": item["item_id"],
                "category": _category_for_kind(item["kind"]),
                "tokens": item["token_count"],
                "sequence": item["sequence"],
            }
            for item in detail["items"]
            if item["status"] not in {"removed", "summarized"}
        ]
        segments.extend(
            {
                "id": summary["id"],
                "category": "Summaries",
                "tokens": summary["token_count"],
                "sequence": summary["sequence"],
            }
            for summary in detail["summaries"]
        )
        segments.sort(key=lambda entry: (int(entry["sequence"]), str(entry["id"])))
        return {
            "active": True,
            "session_id": detail["id"],
            "name": detail["name"],
            "model": detail["model"],
            "updated_at": detail["updated_at"],
            "request_count": detail["request_count"],
            "context_limit": detail["context_limit"],
            "context_budget": detail["context_budget"],
            "auto_mode": detail["auto_mode"],
            "auto_cleanup_mode": detail["auto_cleanup_mode"],
            "metrics": detail["metrics"],
            "segments": segments,
        }

    @router.post("/api/overlay/launch")
    async def launch_overlay(value: OverlayLaunchInput) -> dict[str, Any]:
        if value.session_id and not database.get_session(value.session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        query = f"?session={value.session_id}" if value.session_id else ""
        url = f"http://{settings.host}:{settings.port}/overlay{query}"
        if os.name != "nt":
            return {"launched": False, "url": url, "reason": "Always-on-top launcher is Windows-only"}
        try:
            subprocess.Popen(
                [sys.executable, str(PROJECT_ROOT / "overlay.py"), "--url", url],
                cwd=PROJECT_ROOT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Overlay could not be launched: {exc}") from exc
        return {"launched": True, "url": url}

    @router.get("/api/workbench/config")
    async def workbench_config() -> dict[str, Any]:
        return {
            "available": longcat.available,
                "xtllm_root": str(longcat.root),
                "xtllm_binary_root": str(longcat.binary_root),
            "profiles": longcat.profile_payloads(),
            "host": {
                "ram_gib": host_ram_gib(),
                "gpu": "AMD Radeon RX 6700 XT",
                "runtime": "XTLLM · native Vulkan",
            },
            "limits": {
                "total_ram_gib": TOTAL_RAM_BUDGET_GIB,
                "max_vram_gib": MAX_VRAM_BUDGET_GIB,
                "context_tokens": MODEL_CONTEXT_LIMIT_TOKENS,
                "executor_context_tokens": max(
                    profile.executor_context_limit for profile in PROFILES.values()
                ),
                "context_bytes_per_token": CONTEXT_BYTES_PER_TOKEN,
                "device_context_bytes_per_token": DEVICE_CONTEXT_BYTES_PER_TOKEN,
                "device_expert_slot_bytes": DEVICE_EXPERT_SLOT_BYTES,
                "minimum_device_slots": MIN_DEVICE_SLOTS,
                "vram_safety_gib": VRAM_SAFETY_GIB,
                "output_tokens": 220,
                "note": (
                    "LongCat Flash Lite Sparse supports up to 983,040 tokens. "
                    "XTLLM now sizes context at runtime; practical capacity is limited by "
                    "the selected VRAM budget and its device-local KV cache."
                ),
            },
        }

    @router.get("/api/workbench/chats")
    async def list_workbench_chats() -> dict[str, Any]:
        return {"chats": database.list_workbench_chats()}

    @router.post("/api/workbench/chats")
    async def create_workbench_chat(value: WorkbenchChatInput) -> dict[str, Any]:
        profile = PROFILES[value.profile]
        data = value.model_dump()
        data["ram_budget_gib"] = value.ram_budget_gib or profile.recommended_ram_gib
        data["vram_budget_gib"] = value.vram_budget_gib or profile.recommended_vram_gib
        _validate_workbench_resources(data)
        return database.create_workbench_chat(data)

    @router.get("/api/workbench/chats/{chat_id}")
    async def get_workbench_chat(chat_id: str) -> dict[str, Any]:
        chat = database.get_workbench_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        return chat

    @router.patch("/api/workbench/chats/{chat_id}")
    async def update_workbench_chat(
        chat_id: str, value: WorkbenchChatPatch
    ) -> dict[str, Any]:
        chat = database.get_workbench_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        data = {key: val for key, val in value.model_dump().items() if val is not None}
        profile_id = str(data.get("profile") or chat["profile"])
        profile = PROFILES[profile_id]
        if "profile" in data and "vram_budget_gib" not in data:
            data["vram_budget_gib"] = profile.recommended_vram_gib
        if float(data.get("vram_budget_gib", chat["vram_budget_gib"])) > profile.max_vram_gib:
            raise HTTPException(
                status_code=422,
                detail=f"{profile.name} uses at most {profile.max_vram_gib:.2f} GiB VRAM",
            )
        candidate = {
            "ram_budget_gib": data.get("ram_budget_gib", chat["ram_budget_gib"]),
            "context_ram_budget_gib": data.get(
                "context_ram_budget_gib", chat["context_ram_budget_gib"]
            ),
            "vram_budget_gib": data.get("vram_budget_gib", chat["vram_budget_gib"]),
            "context_limit": data.get("context_limit", chat["context_limit"]),
            "output_token_limit": data.get(
                "output_token_limit", chat["output_token_limit"]
            ),
        }
        _validate_workbench_resources(candidate)
        updated = database.update_workbench_chat(chat_id, data)
        if chat.get("lcm_session_id"):
            policy = {
                "context_limit": int(data.get("context_limit", chat["context_limit"])),
                "context_budget": max(
                    32,
                    int(data.get("context_limit", chat["context_limit"]))
                    - int(data.get("output_token_limit", chat["output_token_limit"])),
                ),
                "auto_mode": bool(data.get("auto_mode", chat["auto_mode"])),
                "auto_cleanup_mode": str(
                    data.get("cleanup_mode", chat["cleanup_mode"])
                ),
            }
            database.update_session_policy(str(chat["lcm_session_id"]), policy)
        return updated or {}

    @router.post("/api/workbench/chats/{chat_id}/messages")
    async def send_workbench_message(
        chat_id: str, value: WorkbenchMessageInput, request: Request
    ) -> StreamingResponse:
        chat = database.get_workbench_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        if chat.get("archived"):
            raise HTTPException(status_code=409, detail="Archived chats are read-only")

        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def emit(event: str, **payload: Any) -> None:
            await queue.put({"event": event, **payload})

        async def produce() -> None:
            request_id: str | None = None
            started = time.perf_counter()
            try:
                await emit(
                    "status",
                    stage="received",
                    message="Reading the conversation and preparing local context",
                )
                user_message = database.add_workbench_message(
                    chat_id,
                    "user",
                    value.content.strip(),
                    engine.tokenizer.count_text(value.content, "longcat").value,
                )
                current = database.get_workbench_chat(chat_id) or chat
                if int(current.get("message_count", 0)) == 0 or current["title"] == "New local chat":
                    title = " ".join(value.content.strip().split())[:64]
                    current = database.update_workbench_chat(chat_id, {"title": title}) or current
                messages = [
                    {"role": item["role"], "content": item["content"]}
                    for item in database.workbench_messages(chat_id)
                    if item["role"] in {"user", "assistant"}
                ]
                payload = {
                    "model": f"longcat-{current['profile']}",
                    "stream": True,
                    "messages": messages,
                }
                transformed = engine.transform(
                    payload,
                    f"workbench-{chat_id}",
                    {"id": None, "context_limit": int(current["context_limit"])},
                )
                request_id = transformed.request_id
                database.update_workbench_chat(
                    chat_id, {"lcm_session_id": transformed.session_id}
                )
                database.rename_session(transformed.session_id, str(current["title"]))
                database.update_session_policy(
                    transformed.session_id,
                    {
                        "context_limit": int(current["context_limit"]),
                        "context_budget": max(
                            32,
                            int(current["context_limit"])
                            - int(current["output_token_limit"]),
                        ),
                        "auto_mode": bool(current["auto_mode"]),
                        "auto_cleanup_mode": str(current["cleanup_mode"]),
                    },
                )
                await emit(
                    "context",
                    message="Context inspected locally",
                    session_id=transformed.session_id,
                    incoming_tokens=transformed.incoming_tokens,
                    forwarded_tokens=transformed.forwarded_tokens,
                    removed_tokens=max(
                        0, transformed.incoming_tokens - transformed.forwarded_tokens
                    ),
                )

                async def progress(event: dict[str, Any]) -> None:
                    await emit("status", **event)

                result = await longcat.generate(
                    chat_id,
                    str(current["profile"]),
                    transformed.forwarded["messages"],
                    float(current["ram_budget_gib"]),
                    float(current["context_ram_budget_gib"]),
                    float(current["vram_budget_gib"]),
                    int(current["context_limit"]),
                    int(current["output_token_limit"]),
                    progress,
                )
                assistant_message = database.add_workbench_message(
                    chat_id,
                    "assistant",
                    result.content,
                    result.completion_tokens,
                    result.metrics,
                )
                database.finish_request(
                    request_id,
                    200,
                    result.elapsed_ms,
                    (time.perf_counter() - started) * 1000,
                    result.prompt_tokens,
                    result.completion_tokens,
                )
                await emit("chunk", content=result.content)
                await emit(
                    "done",
                    message="Response complete",
                    assistant=assistant_message,
                    user=user_message,
                    metrics=result.metrics,
                    chat=database.get_workbench_chat(chat_id),
                )
            except asyncio.CancelledError:
                await longcat.cancel(chat_id)
                raise
            except Exception as exc:
                if request_id:
                    database.finish_request(
                        request_id,
                        500,
                        None,
                        (time.perf_counter() - started) * 1000,
                    )
                await emit("error", message=str(exc))
            finally:
                await queue.put(None)

        async def events():
            task = asyncio.create_task(produce())
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=1)
                    except TimeoutError:
                        if await request.is_disconnected():
                            await longcat.cancel(chat_id)
                            task.cancel()
                            break
                        continue
                    if event is None:
                        break
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                await task
            finally:
                if not task.done():
                    task.cancel()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/api/workbench/chats/{chat_id}/stop")
    async def stop_workbench_generation(chat_id: str) -> dict[str, Any]:
        if not database.get_workbench_chat(chat_id):
            raise HTTPException(status_code=404, detail="Chat not found")
        stopped = await longcat.cancel(chat_id)
        return {"stopped": stopped}

    @router.get("/v1/models")
    async def proxy_models(request: Request) -> Response:
        return await proxy.models(request)

    @router.post("/v1/chat/completions")
    async def proxy_chat(request: Request, payload: dict[str, Any]) -> Response:
        return await proxy.chat_completions(request, payload)

    return router


def _expand_dependencies(items: list[dict[str, Any]], selected: list[str]) -> list[str]:
    expanded = set(selected)
    by_id = {item["item_id"]: item for item in items}
    changed = True
    while changed:
        changed = False
        for item_id in list(expanded):
            for relation in by_id.get(item_id, {}).get("relationships", []):
                related = relation.get("item_id")
                if related and related not in expanded:
                    expanded.add(related)
                    changed = True
    return sorted(expanded)


def _session_summary(database: Database, session_id: str) -> dict[str, Any] | None:
    detail = database.session_detail(session_id)
    if not detail:
        return None
    active = [item for item in detail["items"] if item["status"] not in {"removed", "summarized"}]
    total = sum(item["token_count"] for item in active) + sum(
        summary["token_count"] for summary in detail["summaries"]
    )
    return {
        "id": detail["id"],
        "name": detail["name"],
        "model": detail["model"],
        "updated_at": detail["updated_at"],
        "request_count": detail["request_count"],
        "token_count": total,
        "context_limit": detail["context_limit"],
        "context_budget": detail["context_budget"],
        "item_count": len(detail["items"]),
    }


def _enriched_session(database: Database, session_id: str) -> dict[str, Any] | None:
    detail = database.session_detail(session_id)
    if not detail:
        return None
    backend = database.get_backend(detail.get("backend_id")) if detail.get("backend_id") else None
    active_items = [
        item for item in detail["items"] if item["status"] not in {"removed", "summarized"}
    ]
    summary_tokens = sum(summary["token_count"] for summary in detail["summaries"])
    total = sum(item["token_count"] for item in active_items) + summary_tokens
    incoming_total = sum(item["token_count"] for item in detail["items"])
    categories = {
        "System": 0,
        "Conversation": 0,
        "Code / files": 0,
        "Tool outputs": 0,
        "Memory / RAG": 0,
        "Images": 0,
        "Summaries": summary_tokens,
    }
    for item in active_items:
        kind = item["kind"]
        categories[_category_for_kind(kind)] += item["token_count"]
    free = max(0, int(detail["context_limit"]) - total)
    categories["Free"] = free
    detail["backend"] = backend
    detail["metrics"] = {
        "active_tokens": total,
        "incoming_tokens": incoming_total,
        "removed_tokens": max(0, incoming_total - total),
        "context_percent": round(total / max(1, int(detail["context_limit"])) * 100, 1),
        "budget_percent": round(total / max(1, int(detail["context_budget"])) * 100, 1),
        "categories": categories,
        "kv_cache": estimate_kv_cache(total, (backend or {}).get("architecture")),
        "kv_cache_incoming": estimate_kv_cache(incoming_total, (backend or {}).get("architecture")),
        "counts_are_estimates": any(not item["token_exact"] for item in detail["items"]),
    }
    return detail


def _category_for_kind(kind: str) -> str:
    if kind in {"system_instruction", "developer_instruction"}:
        return "System"
    if kind == "code_or_file":
        return "Code / files"
    if kind in {"tool_output", "tool_result", "tool_call"}:
        return "Tool outputs"
    if kind in {"rag_chunk", "memory"}:
        return "Memory / RAG"
    if kind == "image":
        return "Images"
    return "Conversation"


def _demo_payload() -> dict[str, Any]:
    build_lines = "\n".join(
        f"[{index:03}] warning TS6133: 'legacyValue{index}' is declared but its value is never read."
        for index in range(1, 72)
    )
    old_code = """```python\n# main.py (old)\nfrom flask import Flask\napp = Flask(__name__)\n@app.route('/chat')\ndef chat():\n    return {'status': 'prototype'}\n```"""
    new_code = """```python\n# main.py (current)\nfrom fastapi import FastAPI\nfrom fastapi.responses import StreamingResponse\napp = FastAPI()\n@app.post('/v1/chat/completions')\nasync def chat():\n    return StreamingResponse(relay(), media_type='text/event-stream')\n```"""
    return {
        "model": "Qwen2.5-Coder-7B-Instruct-Q4_K_M",
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful local coding assistant. Preserve user requirements and never send project data to cloud services.",
            },
            {
                "role": "user",
                "content": "Build a local OpenAI-compatible proxy. Streaming, reversible pruning, and exact payload visibility are hard requirements.",
            },
            {
                "role": "assistant",
                "content": "I will start with a small Flask prototype and replace it once the streaming contract is clear.",
            },
            {"role": "user", "content": old_code},
            {
                "role": "assistant",
                "content": "The Flask prototype worked, but it buffers the response. We should use FastAPI and httpx byte streaming.",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-build-1",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command":"npm run build"}',
                        },
                    }
                ],
                "content": "",
            },
            {
                "role": "tool",
                "tool_call_id": "call-build-1",
                "name": "terminal",
                "content": "npm run build\n" + build_lines,
            },
            {"role": "user", "content": new_code},
            {
                "role": "assistant",
                "content": "Streaming now works. The remaining issue is persistent removal when a client resends the original conversation.",
            },
            {
                "role": "user",
                "content": "Current requirements: keep the system instruction pinned, remove the obsolete build log, and retain the current FastAPI version.",
            },
        ],
    }
