from __future__ import annotations

from typing import Any

DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "q8_0": 1, "q4_0": 0.5}


def estimate_kv_cache(tokens: int, architecture: dict[str, Any] | None) -> dict[str, Any] | None:
    """Estimate bytes = tokens * layers * 2(K,V) * kv_heads * head_dim * bytes."""
    if not architecture:
        return None
    try:
        layers = int(architecture["layers"])
        kv_heads = int(architecture["kv_heads"])
        head_dim = int(architecture["head_dim"])
        batch = int(architecture.get("batch_size", 1))
        precision = str(architecture.get("kv_precision", "fp16")).lower()
        bytes_per_element = DTYPE_BYTES[precision]
    except (KeyError, TypeError, ValueError):
        return None
    byte_count = tokens * layers * 2 * kv_heads * head_dim * bytes_per_element * batch
    return {
        "bytes": int(byte_count),
        "gib": round(byte_count / 1024**3, 3),
        "estimated": True,
        "formula": "tokens x layers x 2 (K+V) x KV heads x head dimension x precision bytes x batch",
        "assumptions": {
            "layers": layers,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "kv_precision": precision,
            "batch_size": batch,
        },
    }
