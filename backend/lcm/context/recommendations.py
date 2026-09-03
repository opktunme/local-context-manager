from __future__ import annotations

from collections import Counter

from ..models import ContextItem


def apply_recommendations(items: list[ContextItem]) -> None:
    fingerprints = Counter(item.fingerprint for item in items)
    latest_user = max((item.sequence for item in items if item.role == "user"), default=-1)
    latest_file: dict[str, int] = {}
    for item in items:
        if item.kind == "code_or_file":
            latest_file[item.source] = item.sequence

    for item in items:
        if item.status == "pinned":
            item.recommendation = "pinned_critical"
            item.recommendation_reason = "Pinned by the user; automatic policies will not alter it."
        elif item.role in {"system", "developer"}:
            item.recommendation = "pinned_critical"
            item.recommendation_reason = (
                "Instruction messages define model behavior and are protected by default."
            )
        elif item.sequence == latest_user:
            item.recommendation = "probably_keep"
            item.recommendation_reason = "This is the current user request and is protected."
        elif fingerprints[item.fingerprint] > 1 and not item.id.endswith(".0"):
            item.recommendation = "safe_to_remove"
            item.recommendation_reason = "An exact duplicate is already present in this request."
        elif item.kind == "tool_output" and item.token_count >= 700:
            item.recommendation = "safe_to_remove"
            item.recommendation_reason = (
                "Large command or compiler output is old enough to review for removal."
            )
        elif (
            item.kind in {"assistant_response", "user_message"}
            and item.sequence < latest_user - 5
            and item.token_count >= 500
        ):
            item.recommendation = "summarize"
            item.recommendation_reason = "An older, substantial conversation turn can often be compacted into decisions and state."
        elif item.kind == "rag_chunk" and item.sequence < latest_user - 3:
            item.recommendation = "review_manually"
            item.recommendation_reason = "This retrieved context is no longer recent; verify that later turns still depend on it."
        else:
            item.recommendation = "probably_keep"
            item.recommendation_reason = (
                "No strong zero-compute signal says this item is safe to remove."
            )
