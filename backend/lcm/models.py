from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TokenCount:
    value: int
    exact: bool
    source: str


@dataclass(slots=True)
class ContextItem:
    id: str
    fingerprint: str
    kind: str
    role: str
    source: str
    sequence: int
    token_count: int
    token_exact: bool
    token_source: str
    preview: str
    text: str
    raw_message: dict[str, Any]
    status: str = "active"
    reason: str = "Sent by frontend"
    relationships: list[dict[str, str]] = field(default_factory=list)
    recommendation: str = "probably_keep"
    recommendation_reason: str = "Part of the active conversation."

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "fingerprint": self.fingerprint,
            "kind": self.kind,
            "role": self.role,
            "source": self.source,
            "sequence": self.sequence,
            "token_count": self.token_count,
            "token_exact": self.token_exact,
            "token_source": self.token_source,
            "preview": self.preview,
            "text": self.text,
            "raw_message": self.raw_message,
            "status": self.status,
            "reason": self.reason,
            "relationships": self.relationships,
            "recommendation": self.recommendation,
            "recommendation_reason": self.recommendation_reason,
        }
