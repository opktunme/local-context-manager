from __future__ import annotations

import ipaddress
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

import httpx


class BackendConnectionError(RuntimeError):
    pass


def validate_local_backend_url(url: str, allow_private_lan: bool = True) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Backend URL must be a valid http:// or https:// URL")
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith((".localhost", ".local")):
        return
    try:
        ip = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError(
            "LCM only connects to localhost, private IP addresses, or .local hosts"
        ) from exc
    if ip.is_loopback or (allow_private_lan and ip.is_private):
        return
    raise ValueError("Public/cloud backend addresses are disabled by the local-only safety policy")


class BackendAdapter:
    kind = "openai"
    label = "OpenAI-compatible"

    def __init__(
        self,
        config: dict[str, Any],
        api_key: str | None = None,
        allow_private_lan: bool = True,
    ) -> None:
        self.config = config
        self.base_url = str(config["base_url"]).rstrip("/")
        self.api_key = api_key
        validate_local_backend_url(self.base_url, allow_private_lan)

    def url(self, path: str) -> str:
        if self.base_url.endswith("/v1") and path.startswith("/v1/"):
            return self.base_url + path[3:]
        return self.base_url + path

    def headers(self, incoming: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if incoming and incoming.get("authorization"):
            headers["Authorization"] = incoming["authorization"]
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def list_models(
        self, incoming_headers: dict[str, str] | None = None
    ) -> tuple[int, Any, dict[str, str]]:
        try:
            async with httpx.AsyncClient(timeout=8.0, trust_env=False) as client:
                response = await client.get(
                    self.url("/v1/models"), headers=self.headers(incoming_headers)
                )
            try:
                payload: Any = response.json()
            except ValueError:
                payload = {
                    "error": {"message": response.text or "Backend returned a non-JSON response"}
                }
            return (
                response.status_code,
                payload,
                self.response_headers(response.headers),
            )
        except httpx.HTTPError as exc:
            raise BackendConnectionError(str(exc)) from exc

    async def request_json(
        self,
        path: str,
        payload: dict[str, Any],
        incoming_headers: dict[str, str] | None = None,
        timeout: float | None = 300.0,
    ) -> tuple[int, Any, dict[str, str]]:
        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                response = await client.post(
                    self.url(path), json=payload, headers=self.headers(incoming_headers)
                )
            try:
                data: Any = response.json()
            except ValueError:
                data = {
                    "error": {"message": response.text or "Backend returned a non-JSON response"}
                }
            return response.status_code, data, self.response_headers(response.headers)
        except httpx.HTTPError as exc:
            raise BackendConnectionError(str(exc)) from exc

    async def stream(
        self,
        path: str,
        payload: dict[str, Any],
        incoming_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[tuple[bytes, int, dict[str, str]]]:
        client = httpx.AsyncClient(timeout=None, trust_env=False)
        try:
            async with client.stream(
                "POST",
                self.url(path),
                json=payload,
                headers=self.headers(incoming_headers),
            ) as response:
                response_headers = self.response_headers(response.headers)
                if response.status_code >= 400:
                    body = await response.aread()
                    yield body, response.status_code, response_headers
                    return
                async for chunk in response.aiter_raw():
                    if chunk:
                        yield chunk, response.status_code, response_headers
        except httpx.HTTPError as exc:
            raise BackendConnectionError(str(exc)) from exc
        finally:
            await client.aclose()

    def extract_metrics(self, payload: dict[str, Any]) -> dict[str, int | float | None]:
        usage = payload.get("usage") or {}
        return {
            "prompt_tokens": usage.get("prompt_tokens") or payload.get("prompt_eval_count"),
            "completion_tokens": usage.get("completion_tokens") or payload.get("eval_count"),
            "prompt_eval_ms": _ns_to_ms(payload.get("prompt_eval_duration")),
        }

    @staticmethod
    def response_headers(headers: httpx.Headers) -> dict[str, str]:
        allowed = {"content-type", "cache-control", "content-encoding", "x-request-id"}
        return {key: value for key, value in headers.items() if key.lower() in allowed}


def _ns_to_ms(value: Any) -> float | None:
    try:
        return round(float(value) / 1_000_000, 3)
    except (TypeError, ValueError):
        return None
