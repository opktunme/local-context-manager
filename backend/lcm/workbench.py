from __future__ import annotations

import asyncio
import os
import re
import struct
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import Settings

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]

TOTAL_RAM_BUDGET_GIB = 70.0
MAX_VRAM_BUDGET_GIB = 11.5
MODEL_CONTEXT_LIMIT_TOKENS = 983_040
# LongCat MLA keeps one BF16 compressed KV vector (512 latent + 64 RoPE values)
# for each of the model's 28 attention blocks.
CONTEXT_BYTES_PER_TOKEN = 28 * (512 + 64) * 2
# The current Vulkan executor expands K/V per head in FP32 on the GPU.
DEVICE_CONTEXT_BYTES_PER_TOKEN = 28 * 32 * 320 * 4
LEGACY_MEASURED_CONTEXT_TOKENS = 256
DEVICE_EXPERT_SLOT_BYTES = 14 * 5_013_504
MIN_DEVICE_SLOTS = 12
VRAM_SAFETY_GIB = 0.5
CHAT_MAGIC = b"XTCHAT1\0"
CHAT_REFERENCE_PREFIX = "@XTLLM_CHAT_FILE:"
MAX_CHAT_MESSAGES = 63
MAX_CHAT_CONTENT_BYTES = 512 * 1024


def context_ram_gib(tokens: int) -> float:
    return max(0, int(tokens)) * CONTEXT_BYTES_PER_TOKEN / 1024**3


@dataclass(frozen=True, slots=True)
class LongCatProfile:
    id: str
    name: str
    description: str
    executable: str
    runtime: str
    precision: str
    max_ram_gib: float
    recommended_ram_gib: float
    measured_ram_gib: float
    max_vram_gib: float
    recommended_vram_gib: float
    measured_vram_gib: float
    max_device_slots: int
    active_routes: int = 12
    context_limit: int = MODEL_CONTEXT_LIMIT_TOKENS
    executor_context_limit: int = 256
    max_output_tokens: int = 220
    q4_one_lane: bool = False


PROFILES = {
    "exact": LongCatProfile(
        id="exact",
        name="LongCat Exact",
        description="Highest-fidelity shared path with Q8 attention and dense projections.",
        executable="build/xtllm-longcat-flash-lite.exe",
        runtime="runtime/longcat-flash-lite-q8shared",
        precision="Q4 experts · Q8 shared path",
        max_ram_gib=TOTAL_RAM_BUDGET_GIB,
        recommended_ram_gib=24.0,
        measured_ram_gib=16.7344,
        max_vram_gib=MAX_VRAM_BUDGET_GIB,
        recommended_vram_gib=9.6592,
        measured_vram_gib=9.6592,
        max_device_slots=96,
        executor_context_limit=6_652,
    ),
    "full": LongCatProfile(
        id="full",
        name="LongCat Full",
        description="Full 12-route hybrid with quantized dense and attention-output paths.",
        executable="build/xtllm-longcat-flash-lite-hybrid.exe",
        runtime="runtime/longcat-flash-lite-dense-outq4",
        precision="Q4 experts/dense/output · Q8 QKV",
        max_ram_gib=TOTAL_RAM_BUDGET_GIB,
        recommended_ram_gib=24.0,
        measured_ram_gib=16.7344,
        max_vram_gib=MAX_VRAM_BUDGET_GIB,
        recommended_vram_gib=10.3802,
        measured_vram_gib=10.3802,
        max_device_slots=120,
        executor_context_limit=7_445,
        q4_one_lane=True,
    ),
}


def runtime_context_capacity(profile: LongCatProfile, vram_budget_gib: float) -> int:
    fixed_vram_gib = _fixed_vram_gib(profile)
    minimum_slot_gib = MIN_DEVICE_SLOTS * DEVICE_EXPERT_SLOT_BYTES / 1024**3
    context_gib = max(
        0.0,
        float(vram_budget_gib)
        - VRAM_SAFETY_GIB
        - fixed_vram_gib
        - minimum_slot_gib,
    )
    tokens = int(context_gib * 1024**3 / DEVICE_CONTEXT_BYTES_PER_TOKEN)
    return max(64, min(tokens, profile.context_limit))


def _fixed_vram_gib(profile: LongCatProfile) -> float:
    measured_context_gib = (
        LEGACY_MEASURED_CONTEXT_TOKENS * DEVICE_CONTEXT_BYTES_PER_TOKEN / 1024**3
    )
    measured_slots_gib = (
        profile.max_device_slots * DEVICE_EXPERT_SLOT_BYTES / 1024**3
    )
    return profile.measured_vram_gib - measured_context_gib - measured_slots_gib


def runtime_device_slots(
    profile: LongCatProfile,
    vram_budget_gib: float,
    context_tokens: int,
) -> int:
    context_gib = context_tokens * DEVICE_CONTEXT_BYTES_PER_TOKEN / 1024**3
    available_gib = (
        float(vram_budget_gib)
        - VRAM_SAFETY_GIB
        - _fixed_vram_gib(profile)
        - context_gib
    )
    slots = int(available_gib * 1024**3 / DEVICE_EXPERT_SLOT_BYTES)
    return max(MIN_DEVICE_SLOTS, min(slots, profile.max_device_slots))

ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


@dataclass(slots=True)
class GenerationResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_ms: float
    metrics: dict[str, Any]
    raw_output: str


class LongCatRuntime:
    """Async bridge for the two native XTLLM LongCat executables on A:."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.xtllm_root
        self.binary_root = settings.xtllm_binary_root
        self._lock = asyncio.Lock()
        self._active: dict[str, asyncio.subprocess.Process] = {}

    def profile(self, profile_id: str) -> LongCatProfile:
        try:
            return PROFILES[profile_id]
        except KeyError as exc:
            raise ValueError(f"Unknown LongCat profile: {profile_id}") from exc

    def profile_payloads(self) -> list[dict[str, Any]]:
        payloads = []
        for profile in PROFILES.values():
            data = asdict(profile)
            data["installed"] = (self.binary_root / profile.executable).is_file() and (
                self.root / profile.runtime
            ).is_dir()
            payloads.append(data)
        return payloads

    @property
    def available(self) -> bool:
        return all(item["installed"] for item in self.profile_payloads())

    async def tokenize(self, profile_id: str, prompt: str) -> int:
        profile = self.profile(profile_id)
        executable = self.binary_root / profile.executable
        runtime = self.root / profile.runtime
        process = await asyncio.create_subprocess_exec(
            str(executable),
            str(runtime),
            "--tokenize",
            prompt,
            cwd=str(self.binary_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=getattr(__import__("subprocess"), "CREATE_NO_WINDOW", 0),
        )
        output, _ = await asyncio.wait_for(process.communicate(), timeout=20)
        text = output.decode("utf-8", errors="replace")
        if process.returncode:
            raise RuntimeError(text.strip() or "LongCat tokenizer failed")
        match = re.search(r"tokens:\s*([0-9\s]+)", text)
        if not match:
            raise RuntimeError("LongCat tokenizer returned an unexpected response")
        return len(match.group(1).split())

    @staticmethod
    def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Return an alternating LongCat transcript ending with the newest user turn."""
        clean: list[dict[str, str]] = []
        for item in messages:
            role = str(item.get("role") or "")
            content = item.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            content = content.strip()
            if not content:
                continue
            if not clean and role != "user":
                continue
            message = {"role": role, "content": content}
            if clean and clean[-1]["role"] == role:
                # A stopped or failed run can leave adjacent user turns in storage.
                # The newest turn is the one the user expects LongCat to answer.
                clean[-1] = message
            else:
                clean.append(message)
        if not clean or clean[-1]["role"] != "user":
            raise ValueError("A final user message is required")
        if len(clean) > MAX_CHAT_MESSAGES:
            clean = clean[-MAX_CHAT_MESSAGES:]
        return clean

    @staticmethod
    def write_chat_transcript(path: Path, messages: list[dict[str, str]]) -> None:
        if not messages or len(messages) > 64:
            raise ValueError("XTLLM structured chat requires 1 to 64 messages")
        encoded: list[tuple[int, bytes]] = []
        total_bytes = 0
        for index, message in enumerate(messages):
            expected = "user" if index % 2 == 0 else "assistant"
            if message["role"] != expected:
                raise ValueError("XTLLM chat roles must alternate from user")
            content = message["content"].encode("utf-8")
            total_bytes += len(content)
            if len(content) > 256 * 1024 or total_bytes > MAX_CHAT_CONTENT_BYTES:
                raise ValueError("XTLLM chat transcript is too large")
            encoded.append((1 if expected == "user" else 2, content))
        if messages[-1]["role"] != "user":
            raise ValueError("XTLLM chat transcript must end with a user turn")
        with path.open("wb") as output:
            output.write(CHAT_MAGIC)
            output.write(struct.pack("<I", len(encoded)))
            for role, content in encoded:
                output.write(struct.pack("<BI", role, len(content)))
                output.write(content)

    async def tokenize_messages(
        self, profile_id: str, messages: list[dict[str, str]]
    ) -> int:
        # `--tokenize` measures one LongCat user turn: one user marker, the
        # content, and one assistant marker. Remove those two markers to get
        # each content length, then add render_longcat's structured markers.
        content_tokens = 0
        assistants = 0
        for message in messages:
            content_tokens += max(
                0, await self.tokenize(profile_id, message["content"]) - 2
            )
            assistants += int(message["role"] == "assistant")
        return content_tokens + len(messages) + assistants + 1

    async def fit_messages(
        self,
        profile_id: str,
        messages: list[dict[str, Any]],
        prompt_token_limit: int,
    ) -> tuple[list[dict[str, str]], int, int, bool]:
        """Fit recent conversation turns to the native executor without changing history."""
        source_count = sum(
            1
            for item in messages
            if item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str)
            and str(item["content"]).strip()
        )
        clean = self.normalize_messages(messages)
        structurally_dropped = source_count - len(clean)

        async def token_count(candidate: list[dict[str, str]]) -> int:
            return await self.tokenize_messages(profile_id, candidate)

        full_tokens = await token_count(clean)
        if full_tokens <= prompt_token_limit:
            return clean, full_tokens, structurally_dropped, False

        dropped_for_size = 0
        best = clean
        best_tokens = full_tokens
        while best_tokens > prompt_token_limit and len(best) > 1:
            best = best[2:]
            dropped_for_size += 2
            best_tokens = await token_count(best)

        compacted = False
        if best_tokens > prompt_token_limit:
            original = clean[-1]["content"]
            low_chars, high_chars = 1, len(original)
            while low_chars <= high_chars:
                keep_chars = (low_chars + high_chars) // 2
                if keep_chars >= len(original):
                    content = original
                elif keep_chars < 32:
                    content = original[:keep_chars]
                else:
                    head = max(1, int(keep_chars * 0.55))
                    tail = max(1, keep_chars - head)
                    content = f"{original[:head]}\n[… older prompt text omitted …]\n{original[-tail:]}"
                candidate = [{"role": "user", "content": content}]
                candidate_tokens = await token_count(candidate)
                if candidate_tokens <= prompt_token_limit:
                    best = candidate
                    best_tokens = candidate_tokens
                    compacted = keep_chars < len(original)
                    low_chars = keep_chars + 1
                else:
                    high_chars = keep_chars - 1

        if best_tokens > prompt_token_limit:
            raise RuntimeError(
                "The newest message cannot fit inside the installed LongCat executor window. "
                "Shorten the message or lower the output-token limit."
            )
        return (
            best,
            best_tokens,
            structurally_dropped + dropped_for_size,
            compacted,
        )

    async def generate(
        self,
        run_id: str,
        profile_id: str,
        messages: list[dict[str, Any]],
        ram_budget_gib: float,
        context_ram_budget_gib: float,
        vram_budget_gib: float,
        context_limit: int,
        output_token_limit: int,
        progress: ProgressCallback,
    ) -> GenerationResult:
        profile = self.profile(profile_id)
        executable = self.binary_root / profile.executable
        runtime = self.root / profile.runtime
        if not executable.is_file() or not runtime.is_dir():
            raise RuntimeError(
                f"{profile.name} binary/assets are not installed under "
                f"{self.binary_root} and {self.root}"
            )

        configured_context = max(64, min(int(context_limit), profile.context_limit))
        effective_context = min(
            configured_context,
            runtime_context_capacity(profile, vram_budget_gib),
        )
        requested_output = max(1, min(int(output_token_limit), profile.max_output_tokens))
        output_reserve = min(requested_output, max(32, effective_context // 4))
        fitted_messages, prompt_tokens, dropped_messages, compacted_prompt = (
            await self.fit_messages(
                profile_id,
                messages,
                max(1, effective_context - output_reserve),
            )
        )
        if dropped_messages or compacted_prompt:
            details = []
            if dropped_messages:
                details.append(f"{dropped_messages} older message{'s' if dropped_messages != 1 else ''}")
            if compacted_prompt:
                details.append("part of the newest prompt")
            await progress(
                {
                    "stage": "context-fit",
                    "message": (
                        f"Fit context to the {effective_context}-token XTLLM window · "
                        f"trimmed {' and '.join(details)}"
                    ),
                    "dropped_messages": dropped_messages,
                    "prompt_compacted": compacted_prompt,
                }
            )
        available_output = effective_context - prompt_tokens
        if available_output < 1:
            raise RuntimeError(
                f"The prepared prompt uses {prompt_tokens} tokens, exceeding the "
                f"{effective_context}-token LongCat window. Clean or shorten the context."
            )
        output_tokens = max(
            1,
            min(int(output_token_limit), profile.max_output_tokens, available_output),
        )
        ram_budget = max(4.0, min(float(ram_budget_gib), profile.max_ram_gib))
        context_ram_budget = max(0.01, float(context_ram_budget_gib))
        vram_budget = max(4.0, min(float(vram_budget_gib), profile.max_vram_gib))
        runtime_context_tokens = prompt_tokens + output_tokens
        slots = runtime_device_slots(
            profile,
            vram_budget,
            runtime_context_tokens,
        )
        if configured_context > effective_context:
            await progress(
                {
                    "stage": "runtime-limit",
                    "message": (
                        f"Context plan is {configured_context:,} tokens · "
                        f"the {vram_budget:.1f} GiB VRAM budget can grow to "
                        f"{effective_context:,} with the current device-local KV cache"
                    ),
                    "configured_context": configured_context,
                    "executor_context": effective_context,
                }
            )
        await progress(
            {
                "stage": "prepared",
                "message": f"Prepared {prompt_tokens} input tokens · {output_tokens} available for output",
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
            }
        )

        started = time.perf_counter()
        async with self._lock:
            await progress(
                {
                    "stage": "loading",
                    "message": (
                        f"Loading {profile.name} into {ram_budget:g} GiB inference RAM / "
                        f"{context_ram_budget:g} GiB context RAM / {vram_budget:.2f} GiB VRAM"
                    ),
                    "device_slots": slots,
                }
            )
            temp_dir = self.settings.data_dir / "tmp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(
                {
                    "LONGCAT_RAM_GIB": f"{ram_budget:g}",
                    "LONGCAT_DEVICE_SLOTS_PER_LAYER": str(slots),
                    "LONGCAT_FILL_RAM_CACHE": "1",
                    "LONGCAT_ACTIVE_TOPK": str(profile.active_routes),
                    "LONGCAT_CONTEXT_TOKENS": str(runtime_context_tokens),
                    "XTLLM_STRUCTURED_CHAT": "1",
                    "TEMP": str(temp_dir),
                    "TMP": str(temp_dir),
                }
            )
            env.pop("LONGCAT_PROGRESSIVE_EXPERTS", None)
            if profile.q4_one_lane:
                env["LONGCAT_Q4_ONE_LANE"] = "1"
            else:
                env.pop("LONGCAT_Q4_ONE_LANE", None)
            handle, transcript_name = tempfile.mkstemp(
                prefix="xtllm-chat-", suffix=".bin", dir=temp_dir
            )
            os.close(handle)
            transcript_path = Path(transcript_name)
            try:
                self.write_chat_transcript(transcript_path, fitted_messages)
                process = await asyncio.create_subprocess_exec(
                    str(executable),
                    str(runtime),
                    CHAT_REFERENCE_PREFIX + str(transcript_path),
                    str(output_tokens),
                    cwd=str(self.binary_root),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    creationflags=getattr(
                        __import__("subprocess"), "CREATE_NO_WINDOW", 0
                    ),
                )
                self._active[run_id] = process
                try:
                    output, _ = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.settings.longcat_timeout_seconds,
                    )
                except TimeoutError as exc:
                    process.kill()
                    await process.wait()
                    raise RuntimeError("LongCat inference timed out") from exc
            finally:
                self._active.pop(run_id, None)
                transcript_path.unlink(missing_ok=True)

        elapsed_ms = (time.perf_counter() - started) * 1000
        raw_output = ANSI_ESCAPE.sub("", output.decode("utf-8", errors="replace"))
        if process.returncode:
            tail = raw_output[-3000:].strip()
            if process.returncode == 1 and "terminated" in tail.lower():
                raise RuntimeError("Generation stopped")
            raise RuntimeError(tail or f"LongCat exited with code {process.returncode}")
        await progress(
            {
                "stage": "generating",
                "message": "RAM cache is ready · decoding the response on Vulkan",
            }
        )
        content, metrics = self.parse_output(raw_output)
        completion_tokens = len(metrics.get("token_ids", []))
        metrics.update(
            {
                "profile": profile.id,
                "profile_name": profile.name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed_ms": round(elapsed_ms, 1),
                "device_slots": slots,
                "configured_context_tokens": configured_context,
                "executor_context_tokens": effective_context,
                "allocated_context_tokens": runtime_context_tokens,
                "context_ram_budget_gib": context_ram_budget,
                "runtime_dropped_messages": dropped_messages,
                "runtime_prompt_compacted": compacted_prompt,
            }
        )
        return GenerationResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_ms=elapsed_ms,
            metrics=metrics,
            raw_output=raw_output,
        )

    async def cancel(self, run_id: str) -> bool:
        process = self._active.get(run_id)
        if not process or process.returncode is not None:
            return False
        process.terminate()
        return True

    @staticmethod
    def parse_output(output: str) -> tuple[str, dict[str, Any]]:
        model_marker = output.find("\nmodel: LongCat-Flash-Lite-Sparse")
        if model_marker < 0:
            raise RuntimeError("Could not identify the generated LongCat response")
        prefix = output[:model_marker]
        cache_match = re.search(r"RAM cache top-off:[^\r\n]*\r?\n", prefix)
        content = prefix[cache_match.end() if cache_match else 0 :].strip()
        metrics: dict[str, Any] = {}
        patterns: dict[str, tuple[str, Callable[[str], Any]]] = {
            "ram_budget_gib": (r"RAM budget/actual GiB:\s*([0-9.]+)", float),
            "ram_actual_gib": (r"RAM budget/actual GiB:\s*[0-9.]+\s*/\s*([0-9.]+)", float),
            "vram_actual_gib": (r"peak device allocation estimate GiB:\s*([0-9.]+)", float),
            "throughput_tps": (r"decode throughput:\s*([0-9.]+)", float),
            "active_routes": (r"active MoE routes:\s*([0-9]+)", int),
            "allocated_context_tokens": (
                r"allocated context capacity:\s*([0-9]+)",
                int,
            ),
        }
        for key, (pattern, cast) in patterns.items():
            match = re.search(pattern, output)
            if match:
                metrics[key] = cast(match.group(1))
        precision = re.search(r"precision:\s*(.+?)\r?\nactive MoE routes", output, re.DOTALL)
        if precision:
            metrics["precision"] = " ".join(precision.group(1).split())
        token_ids = re.search(r"token ids:\s*([0-9\s]+)", output)
        metrics["token_ids"] = (
            [int(value) for value in token_ids.group(1).split()] if token_ids else []
        )
        if not content:
            content = "The model completed without returning visible text."
        return content, metrics


def host_ram_gib() -> float | None:
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / (1 << 30), 2)
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return round(page_size * pages / (1 << 30), 2)
    except (AttributeError, ValueError):
        return None
