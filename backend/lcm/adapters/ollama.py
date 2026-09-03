from __future__ import annotations

from typing import Any

import httpx

from .openai import OpenAICompatibleAdapter


class OllamaAdapter(OpenAICompatibleAdapter):
    kind = "ollama"
    label = "Ollama"

    async def native_details(self) -> dict[str, Any] | None:
        try:
            async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
                response = await client.get(self.url("/api/tags"))
            return response.json() if response.is_success else None
        except (httpx.HTTPError, ValueError):
            return None
