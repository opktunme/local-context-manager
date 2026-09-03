from __future__ import annotations

import hashlib
import json
from typing import Any

FINGERPRINT_FIELDS = (
    "role",
    "name",
    "content",
    "tool_calls",
    "tool_call_id",
    "function_call",
)


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, str):
        return value.replace("\r\n", "\n").rstrip()
    return value


def fingerprint_message(message: dict[str, Any]) -> str:
    identity = {key: _normalize(message[key]) for key in FINGERPRINT_FIELDS if key in message}
    canonical = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def prefix_signature(fingerprints: list[str]) -> str:
    canonical = "|".join(fingerprints[:8])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
