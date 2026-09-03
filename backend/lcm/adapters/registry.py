from __future__ import annotations

from typing import Any

from .base import BackendAdapter
from .ollama import OllamaAdapter
from .openai import (
    LlamaCppAdapter,
    LMStudioAdapter,
    MLXAdapter,
    OpenAICompatibleAdapter,
    VLLMAdapter,
    XTLLMAdapter,
)

ADAPTERS: dict[str, type[BackendAdapter]] = {
    "openai": OpenAICompatibleAdapter,
    "ollama": OllamaAdapter,
    "lmstudio": LMStudioAdapter,
    "llamacpp": LlamaCppAdapter,
    "vllm": VLLMAdapter,
    "mlx": MLXAdapter,
    "xtllm": XTLLMAdapter,
}

ADAPTER_CATALOG = [
    {
        "kind": "lmstudio",
        "name": "LM Studio",
        "default_url": "http://127.0.0.1:1234",
        "status": "OpenAI-compatible + model metadata",
    },
    {
        "kind": "ollama",
        "name": "Ollama",
        "default_url": "http://127.0.0.1:11434",
        "status": "OpenAI-compatible + native discovery",
    },
    {
        "kind": "llamacpp",
        "name": "llama.cpp",
        "default_url": "http://127.0.0.1:8080",
        "status": "OpenAI-compatible",
    },
    {
        "kind": "vllm",
        "name": "vLLM",
        "default_url": "http://127.0.0.1:8000",
        "status": "OpenAI-compatible",
    },
    {
        "kind": "mlx",
        "name": "MLX / MLX-LM",
        "default_url": "http://127.0.0.1:8080",
        "status": "OpenAI-style chat API",
    },
    {
        "kind": "xtllm",
        "name": "XTLLM / xLLM",
        "default_url": "http://127.0.0.1:9977",
        "status": "OpenAI-compatible",
    },
    {
        "kind": "openai",
        "name": "Other OpenAI-compatible",
        "default_url": "http://127.0.0.1:1234",
        "status": "Generic compatibility",
    },
]


def adapter_for(
    config: dict[str, Any], api_key: str | None = None, allow_private_lan: bool = True
) -> BackendAdapter:
    adapter_class = ADAPTERS.get(str(config.get("kind", "openai")), OpenAICompatibleAdapter)
    return adapter_class(config, api_key=api_key, allow_private_lan=allow_private_lan)
