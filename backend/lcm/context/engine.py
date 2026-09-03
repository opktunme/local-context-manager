from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..models import ContextItem
from ..storage import Database
from ..tokenizers import TokenizerService
from .normalizer import normalize_messages
from .recommendations import apply_recommendations


@dataclass(slots=True)
class TransformResult:
    session_id: str
    request_id: str
    incoming: dict[str, Any]
    forwarded: dict[str, Any]
    items: list[ContextItem]
    incoming_tokens: int
    forwarded_tokens: int


class ContextEngine:
    def __init__(self, database: Database, tokenizer: TokenizerService | None = None) -> None:
        self.database = database
        self.tokenizer = tokenizer or TokenizerService()

    def transform(
        self,
        payload: dict[str, Any],
        explicit_session_id: str | None,
        backend: dict[str, Any] | None,
    ) -> TransformResult:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not all(
            isinstance(message, dict) for message in messages
        ):
            raise ValueError("'messages' must be an array of message objects")
        model = str(payload.get("model") or "unknown-model")
        items = normalize_messages(messages, model, self.tokenizer)
        fingerprints = [item.fingerprint for item in items]
        session_id = self._identify_session(explicit_session_id, fingerprints, model, backend)
        context_limit = int((backend or {}).get("context_limit", 32768))
        session = self.database.ensure_session(
            session_id,
            explicit_session_id,
            (backend or {}).get("id"),
            model,
            fingerprints,
            context_limit,
        )
        tokenizer_family = str(session.get("tokenizer_family") or "auto")
        if tokenizer_family != "auto":
            for item in items:
                count = self.tokenizer.count_message(item.raw_message, tokenizer_family)
                item.token_count = count.value
                item.token_exact = count.exact
                item.token_source = f"fallback_estimate:{tokenizer_family}"
        apply_recommendations(items)
        self.database.upsert_items(session_id, items)

        statuses = self.database.item_statuses(session_id, [item.id for item in items])
        for item in items:
            item.status = statuses.get(item.id, "active")
        apply_recommendations(items)
        self.database.upsert_items(session_id, items)

        if session.get("auto_mode"):
            self.cleanup_session(
                session_id,
                str(session.get("auto_cleanup_mode") or "conservative"),
                actor="automatic",
            )
            statuses = self.database.item_statuses(session_id, [item.id for item in items])
            for item in items:
                item.status = statuses.get(item.id, "active")

        forwarded_messages = self._forwarded_messages(session_id, items)
        forwarded = {**payload, "messages": forwarded_messages}
        incoming_tokens = sum(item.token_count for item in items)
        forwarded_tokens = sum(
            self.tokenizer.count_message(message, model).value for message in forwarded_messages
        )
        request_id = f"req-{uuid.uuid4().hex[:20]}"
        self.database.record_request(
            request_id,
            session_id,
            payload,
            forwarded,
            incoming_tokens,
            forwarded_tokens,
        )
        return TransformResult(
            session_id=session_id,
            request_id=request_id,
            incoming=payload,
            forwarded=forwarded,
            items=items,
            incoming_tokens=incoming_tokens,
            forwarded_tokens=forwarded_tokens,
        )

    def cleanup_session(
        self, session_id: str, mode: str, actor: str = "automatic"
    ) -> dict[str, Any]:
        """Apply a local cleanup policy to the current overlay before forwarding."""
        if mode not in {"conservative", "balanced", "aggressive"}:
            raise ValueError(f"Unsupported cleanup mode: {mode}")
        detail = self.database.session_detail(session_id)
        if not detail:
            raise ValueError("Session not found")
        policy_actor = f"{actor}:{mode}"

        items = sorted(detail["items"], key=lambda entry: int(entry["sequence"]))
        live = [
            item for item in items if item["status"] not in {"removed", "summarized"}
        ]
        latest_user = max(
            (int(item["sequence"]) for item in live if item["role"] == "user"),
            default=-1,
        )
        protected = {
            item["item_id"]
            for item in live
            if item["status"] == "pinned"
            or item["role"] in {"system", "developer"}
            or (item["role"] == "user" and int(item["sequence"]) == latest_user)
        }

        duplicate_ids: list[str] = []
        seen_fingerprints: set[str] = set()
        for item in live:
            fingerprint = str(item["fingerprint"])
            if fingerprint in seen_fingerprints and item["status"] == "active":
                duplicate_ids.append(item["item_id"])
            else:
                seen_fingerprints.add(fingerprint)

        remove_ids = set(duplicate_ids)
        if mode in {"balanced", "aggressive"}:
            remove_ids.update(
                item["item_id"]
                for item in live
                if item["status"] == "active"
                and item["recommendation"] == "safe_to_remove"
            )
        if mode == "aggressive":
            remove_ids.update(
                item["item_id"]
                for item in live
                if item["status"] == "active"
                and item["recommendation"] == "review_manually"
                and item["kind"] in {"rag_chunk", "memory"}
            )

        remove_ids = self._expand_dependencies(items, remove_ids) - protected
        removable = {
            item["item_id"]
            for item in live
            if item["status"] == "active" and item["item_id"] not in protected
        }
        remove_ids &= removable

        actions: list[dict[str, Any]] = []
        if remove_ids:
            try:
                actions.append(
                    self.database.apply_action(
                        session_id,
                        "remove",
                        sorted(remove_ids),
                        actor=policy_actor,
                    )
                )
            except ValueError:
                remove_ids = set()

        summarized_ids: set[str] = set()
        if mode == "aggressive":
            summarized_ids = {
                item["item_id"]
                for item in live
                if item["status"] == "active"
                and item["item_id"] not in protected
                and item["item_id"] not in remove_ids
                and item["recommendation"] == "summarize"
            }
            if summarized_ids:
                selected = [item for item in live if item["item_id"] in summarized_ids]
                summary = self.draft_summary(selected)
                summary_tokens = self.tokenizer.count_text(
                    summary, str(detail.get("model") or "")
                ).value
                try:
                    actions.append(
                        self.database.apply_action(
                            session_id,
                            "summarize",
                            sorted(summarized_ids),
                            summary,
                            summary_tokens,
                            actor=policy_actor,
                        )
                    )
                except ValueError:
                    summarized_ids = set()

        return {
            "mode": mode,
            "removed": sorted(remove_ids),
            "summarized": sorted(summarized_ids),
            "estimated_token_saving": sum(
                int(action.get("estimated_token_saving", 0)) for action in actions
            ),
            "actions": len(actions),
        }

    @staticmethod
    def _expand_dependencies(
        items: list[dict[str, Any]], selected: set[str]
    ) -> set[str]:
        expanded = set(selected)
        by_id = {str(item["item_id"]): item for item in items}
        changed = True
        while changed:
            changed = False
            for item_id in list(expanded):
                for relation in by_id.get(item_id, {}).get("relationships", []):
                    related = relation.get("item_id")
                    if related and related not in expanded:
                        expanded.add(related)
                        changed = True
        return expanded

    def _identify_session(
        self,
        explicit_id: str | None,
        fingerprints: list[str],
        model: str,
        backend: dict[str, Any] | None,
    ) -> str:
        if explicit_id:
            safe_digest = hashlib.sha256(explicit_id.encode("utf-8")).hexdigest()[:24]
            return f"explicit-{safe_digest}"
        now = datetime.now(UTC)
        best_id: str | None = None
        best_score = 0
        for session in self.database.recent_sessions():
            if session.get("model") != model or session.get("backend_id") != (backend or {}).get(
                "id"
            ):
                continue
            previous = session.get("fingerprints", [])
            prefix = 0
            for left, right in zip(previous, fingerprints, strict=False):
                if left != right:
                    break
                prefix += 1
            updated = datetime.fromisoformat(session["updated_at"])
            recent_seconds = (now - updated).total_seconds()
            qualifies = prefix >= 2 or (prefix == 1 and recent_seconds < 180)
            if qualifies and prefix > best_score:
                best_score = prefix
                best_id = session["id"]
        return best_id or f"inferred-{uuid.uuid4().hex[:24]}"

    def _forwarded_messages(
        self, session_id: str, items: list[ContextItem]
    ) -> list[dict[str, Any]]:
        summaries = self.database.get_active_summaries(session_id)
        summaries_at: dict[int, list[dict[str, Any]]] = {}
        for summary in summaries:
            summaries_at.setdefault(int(summary["sequence"]), []).append(summary)
        outgoing: list[dict[str, Any]] = []
        for item in items:
            for summary in summaries_at.get(item.sequence, []):
                outgoing.append(
                    {
                        "role": "system",
                        "content": "[LCM generated context summary]\n" + summary["text"],
                    }
                )
            if item.status not in {"removed", "summarized"}:
                outgoing.append(item.raw_message)
        return self._repair_tool_history(outgoing)

    @staticmethod
    def _repair_tool_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result_ids = {
            str(message.get("tool_call_id"))
            for message in messages
            if message.get("role") == "tool" and message.get("tool_call_id")
        }
        call_ids = {
            str(call.get("id"))
            for message in messages
            for call in (message.get("tool_calls") or [])
            if isinstance(call, dict) and call.get("id")
        }
        repaired: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") == "tool" and str(message.get("tool_call_id")) not in call_ids:
                continue
            calls = message.get("tool_calls")
            if calls:
                valid_calls = [
                    call
                    for call in calls
                    if not isinstance(call, dict)
                    or not call.get("id")
                    or str(call["id"]) in result_ids
                ]
                if not valid_calls and not message.get("content"):
                    continue
                if len(valid_calls) != len(calls):
                    message = {**message, "tool_calls": valid_calls}
            repaired.append(message)
        return repaired

    def draft_summary(self, items: list[dict[str, Any]]) -> str:
        lines = ["Context compacted by LCM. Preserve these facts and constraints:"]
        for item in sorted(items, key=lambda entry: entry["sequence"]):
            text = " ".join(str(item.get("text", "")).split())
            if not text:
                continue
            label = str(item.get("role", "context")).title()
            lines.append(f"- {label}: {text[:360]}{'…' if len(text) > 360 else ''}")
        return "\n".join(lines)
