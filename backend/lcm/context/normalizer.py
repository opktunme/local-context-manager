from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from ..models import ContextItem
from ..tokenizers import TokenizerService
from .fingerprints import fingerprint_message

FILE_HINT = re.compile(
    r"(?:^|[\\/\s])[^\s]+\.(?:py|ts|tsx|js|jsx|rs|go|cpp|c|h|json|yaml|yml|toml|md)\b",
    re.IGNORECASE,
)
LOG_HINT = re.compile(
    r"(?:traceback|error:|warning:|build log|npm (?:install|run)|compiler|exit code)",
    re.IGNORECASE,
)
RAG_HINT = re.compile(r"(?:retrieved|source document|rag chunk|similarity score)", re.IGNORECASE)


def classify(message: dict[str, Any], text: str) -> str:
    role = str(message.get("role", "unknown"))
    content = message.get("content")
    if role == "system":
        return "system_instruction"
    if role == "developer":
        return "developer_instruction"
    if role == "tool":
        return "tool_result"
    if message.get("tool_calls") or message.get("function_call"):
        return "tool_call"
    if isinstance(content, list) and any(
        isinstance(part, dict) and part.get("type") in {"image", "image_url", "input_image"}
        for part in content
    ):
        return "image"
    if RAG_HINT.search(text):
        return "rag_chunk"
    if LOG_HINT.search(text) and len(text) > 300:
        return "tool_output"
    if "```" in text or FILE_HINT.search(text):
        return "code_or_file"
    if role == "user":
        return "user_message"
    if role == "assistant":
        return "assistant_response"
    return "unknown"


def normalize_messages(
    messages: list[dict[str, Any]],
    model: str | None,
    tokenizer: TokenizerService,
) -> list[ContextItem]:
    occurrences: defaultdict[str, int] = defaultdict(int)
    items: list[ContextItem] = []
    call_owners: dict[str, str] = {}

    for sequence, message in enumerate(messages):
        fingerprint = fingerprint_message(message)
        occurrence = occurrences[fingerprint]
        occurrences[fingerprint] += 1
        item_id = f"{fingerprint}.{occurrence}"
        text = tokenizer.content_to_text(message.get("content"))
        if message.get("tool_calls"):
            text = (text + "\n" + tokenizer.content_to_text(message["tool_calls"])).strip()
        count = tokenizer.count_message(message, model)
        role = str(message.get("role", "unknown"))
        kind = classify(message, text)
        source = str(message.get("name") or "frontend")
        preview = " ".join(text.split())[:180] or f"{role.title()} message"
        item = ContextItem(
            id=item_id,
            fingerprint=fingerprint,
            kind=kind,
            role=role,
            source=source,
            sequence=sequence,
            token_count=count.value,
            token_exact=count.exact,
            token_source=count.source,
            preview=preview,
            text=text,
            raw_message=message,
        )
        items.append(item)
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("id"):
                call_owners[str(call["id"])] = item_id

    by_id = {item.id: item for item in items}
    for item in items:
        call_id = item.raw_message.get("tool_call_id")
        if call_id and str(call_id) in call_owners:
            owner_id = call_owners[str(call_id)]
            item.relationships.append({"type": "result_for", "item_id": owner_id})
            by_id[owner_id].relationships.append({"type": "has_result", "item_id": item.id})

    return items
