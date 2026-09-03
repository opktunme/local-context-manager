from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..models import ContextItem


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Database:
    def __init__(self, path: Path, request_retention: int = 30) -> None:
        self.path = path
        self.request_retention = request_retention
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()

    def init_schema(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS backends (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                base_url TEXT NOT NULL,
                api_key TEXT,
                active INTEGER NOT NULL DEFAULT 0,
                context_limit INTEGER NOT NULL DEFAULT 32768,
                architecture_json TEXT,
                capabilities_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                external_id TEXT,
                name TEXT NOT NULL,
                backend_id TEXT,
                model TEXT,
                context_limit INTEGER NOT NULL DEFAULT 32768,
                context_budget INTEGER NOT NULL DEFAULT 20000,
                tokenizer_family TEXT NOT NULL DEFAULT 'auto',
                auto_mode INTEGER NOT NULL DEFAULT 0,
                auto_cleanup_mode TEXT NOT NULL DEFAULT 'conservative',
                fingerprint_sequence_json TEXT NOT NULL DEFAULT '[]',
                request_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (backend_id) REFERENCES backends(id) ON DELETE SET NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS session_items (
                session_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                kind TEXT NOT NULL,
                role TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                token_count INTEGER NOT NULL,
                token_exact INTEGER NOT NULL DEFAULT 0,
                token_source TEXT NOT NULL,
                preview TEXT NOT NULL,
                text TEXT NOT NULL,
                raw_json TEXT NOT NULL,
                relationships_json TEXT NOT NULL DEFAULT '[]',
                recommendation TEXT NOT NULL,
                recommendation_reason TEXT NOT NULL,
                reason TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                PRIMARY KEY (session_id, item_id),
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS summaries (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                text TEXT NOT NULL,
                source_ids_json TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                undone INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                incoming_json TEXT NOT NULL,
                forwarded_json TEXT NOT NULL,
                incoming_tokens INTEGER NOT NULL,
                forwarded_tokens INTEGER NOT NULL,
                status_code INTEGER,
                ttft_ms REAL,
                total_ms REAL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                action TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS workbench_chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                profile TEXT NOT NULL DEFAULT 'exact',
                ram_budget_gib REAL NOT NULL DEFAULT 24,
                context_ram_budget_gib REAL NOT NULL DEFAULT 0.1,
                vram_budget_gib REAL NOT NULL DEFAULT 9.66,
                context_limit INTEGER NOT NULL DEFAULT 256,
                output_token_limit INTEGER NOT NULL DEFAULT 96,
                auto_mode INTEGER NOT NULL DEFAULT 0,
                cleanup_mode TEXT NOT NULL DEFAULT 'balanced',
                lcm_session_id TEXT,
                archived INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (lcm_session_id) REFERENCES sessions(id) ON DELETE SET NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS workbench_messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'complete',
                token_count INTEGER NOT NULL DEFAULT 0,
                metrics_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES workbench_chats(id) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_items_session_sequence ON session_items(session_id, sequence)",
            "CREATE INDEX IF NOT EXISTS idx_items_session_status ON session_items(session_id, status)",
            "CREATE INDEX IF NOT EXISTS idx_requests_session_created ON requests(session_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_audit_session_created ON audit_log(session_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_workbench_chats_updated ON workbench_chats(archived, updated_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_workbench_messages_chat_created ON workbench_messages(chat_id, created_at)",
        ]
        with self._lock, self.connect() as connection:
            for statement in statements:
                connection.execute(statement)
            session_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "tokenizer_family" not in session_columns:
                connection.execute(
                    "ALTER TABLE sessions ADD COLUMN tokenizer_family TEXT NOT NULL DEFAULT 'auto'"
                )
            if "auto_cleanup_mode" not in session_columns:
                connection.execute(
                    "ALTER TABLE sessions ADD COLUMN auto_cleanup_mode TEXT NOT NULL DEFAULT 'conservative'"
                )
            workbench_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(workbench_chats)").fetchall()
            }
            if "context_ram_budget_gib" not in workbench_columns:
                connection.execute(
                    "ALTER TABLE workbench_chats ADD COLUMN context_ram_budget_gib "
                    "REAL NOT NULL DEFAULT 0.1"
                )
            connection.execute("PRAGMA optimize")

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value_json FROM settings WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row["value_json"]) if row else default

    def set_setting(self, key: str, value: Any) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO settings(key, value_json, updated_at) VALUES(?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
                """,
                (key, json.dumps(value), utc_now()),
            )

    def configure_backend(self, data: dict[str, Any]) -> dict[str, Any]:
        backend_id = str(data.get("id") or uuid.uuid4())
        now = utc_now()
        with self._lock, self.connect() as connection:
            if data.get("active", True):
                connection.execute("UPDATE backends SET active = 0")
            connection.execute(
                """
                INSERT INTO backends(
                    id, name, kind, base_url, api_key, active, context_limit,
                    architecture_json, capabilities_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name, kind=excluded.kind, base_url=excluded.base_url,
                    api_key=excluded.api_key, active=excluded.active,
                    context_limit=excluded.context_limit,
                    architecture_json=excluded.architecture_json,
                    capabilities_json=excluded.capabilities_json, updated_at=excluded.updated_at
                """,
                (
                    backend_id,
                    data.get("name", data.get("kind", "Local backend")),
                    data.get("kind", "openai"),
                    str(data.get("base_url", "http://127.0.0.1:1234")).rstrip("/"),
                    data.get("api_key"),
                    int(bool(data.get("active", True))),
                    int(data.get("context_limit", 32768)),
                    json.dumps(data.get("architecture")) if data.get("architecture") else None,
                    json.dumps(data.get("capabilities", {})),
                    now,
                    now,
                ),
            )
        return self.get_backend(backend_id) or {}

    def get_backend(self, backend_id: str | None = None) -> dict[str, Any] | None:
        query = (
            "SELECT * FROM backends WHERE id = ?"
            if backend_id
            else "SELECT * FROM backends WHERE active = 1 ORDER BY updated_at DESC LIMIT 1"
        )
        params = (backend_id,) if backend_id else ()
        with self.connect() as connection:
            row = connection.execute(query, params).fetchone()
        return self._backend_dict(row) if row else None

    def list_backends(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM backends ORDER BY active DESC, updated_at DESC"
            ).fetchall()
        return [self._backend_dict(row) for row in rows]

    @staticmethod
    def _backend_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["active"] = bool(data["active"])
        data["architecture"] = json.loads(data.pop("architecture_json") or "null")
        data["capabilities"] = json.loads(data.pop("capabilities_json") or "{}")
        data["has_api_key"] = bool(data.pop("api_key", None))
        return data

    def backend_secret(self, backend_id: str) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT api_key FROM backends WHERE id = ?", (backend_id,)
            ).fetchone()
        return row["api_key"] if row else None

    def recent_sessions(self, limit: int = 12) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._session_dict(row) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return self._session_dict(row) if row else None

    @staticmethod
    def _session_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["auto_mode"] = bool(data["auto_mode"])
        data["auto_cleanup_mode"] = data.get("auto_cleanup_mode") or "conservative"
        data["fingerprints"] = json.loads(data.pop("fingerprint_sequence_json") or "[]")
        return data

    def ensure_session(
        self,
        session_id: str,
        external_id: str | None,
        backend_id: str | None,
        model: str,
        fingerprints: list[str],
        context_limit: int,
    ) -> dict[str, Any]:
        now = utc_now()
        default_name = f"{model or 'Local model'} · {now[11:16]}"
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions(
                    id, external_id, name, backend_id, model, context_limit,
                    context_budget, fingerprint_sequence_json, request_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    backend_id=excluded.backend_id, model=excluded.model,
                    fingerprint_sequence_json=excluded.fingerprint_sequence_json,
                    request_count=sessions.request_count + 1, updated_at=excluded.updated_at
                """,
                (
                    session_id,
                    external_id,
                    default_name,
                    backend_id,
                    model,
                    context_limit,
                    min(20000, context_limit),
                    json.dumps(fingerprints),
                    now,
                    now,
                ),
            )
        return self.get_session(session_id) or {}

    def rename_session(self, session_id: str, name: str) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                "UPDATE sessions SET name = ?, updated_at = ? WHERE id = ?",
                (name.strip()[:100], utc_now(), session_id),
            )

    def update_session_policy(self, session_id: str, data: dict[str, Any]) -> None:
        allowed = {
            "context_limit",
            "context_budget",
            "tokenizer_family",
            "auto_mode",
            "auto_cleanup_mode",
        }
        fields = [(key, data[key]) for key in allowed if key in data]
        if not fields:
            return
        assignments = ", ".join(f"{key} = ?" for key, _ in fields)
        values = [
            str(value)
            if key in {"tokenizer_family", "auto_cleanup_mode"}
            else int(bool(value))
            if key == "auto_mode"
            else int(value)
            for key, value in fields
        ]
        with self._lock, self.connect() as connection:
            connection.execute(
                f"UPDATE sessions SET {assignments}, updated_at = ? WHERE id = ?",
                (*values, utc_now(), session_id),
            )

    def upsert_items(self, session_id: str, items: list[ContextItem]) -> None:
        now = utc_now()
        with self._lock, self.connect() as connection:
            for item in items:
                connection.execute(
                    """
                    INSERT INTO session_items(
                        session_id, item_id, fingerprint, sequence, kind, role, source, status,
                        token_count, token_exact, token_source, preview, text, raw_json,
                        relationships_json, recommendation, recommendation_reason, reason,
                        first_seen, last_seen
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, item_id) DO UPDATE SET
                        sequence=excluded.sequence, kind=excluded.kind, role=excluded.role,
                        source=excluded.source, token_count=excluded.token_count,
                        token_exact=excluded.token_exact, token_source=excluded.token_source,
                        preview=excluded.preview, text=excluded.text, raw_json=excluded.raw_json,
                        relationships_json=excluded.relationships_json,
                        recommendation=excluded.recommendation,
                        recommendation_reason=excluded.recommendation_reason,
                        reason=CASE
                            WHEN session_items.status IN ('removed', 'summarized', 'pinned') THEN session_items.reason
                            ELSE excluded.reason
                        END,
                        last_seen=excluded.last_seen
                    """,
                    (
                        session_id,
                        item.id,
                        item.fingerprint,
                        item.sequence,
                        item.kind,
                        item.role,
                        item.source,
                        item.token_count,
                        int(item.token_exact),
                        item.token_source,
                        item.preview,
                        item.text,
                        json.dumps(item.raw_message, ensure_ascii=False),
                        json.dumps(item.relationships),
                        item.recommendation,
                        item.recommendation_reason,
                        item.reason,
                        now,
                        now,
                    ),
                )

    def item_statuses(self, session_id: str, item_ids: list[str]) -> dict[str, str]:
        if not item_ids:
            return {}
        placeholders = ",".join("?" for _ in item_ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT item_id, status FROM session_items WHERE session_id = ? AND item_id IN ({placeholders})",
                (session_id, *item_ids),
            ).fetchall()
        return {row["item_id"]: row["status"] for row in rows}

    def get_items(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM session_items WHERE session_id = ? ORDER BY sequence, first_seen",
                (session_id,),
            ).fetchall()
        result = []
        for row in rows:
            data = dict(row)
            data["token_exact"] = bool(data["token_exact"])
            data["raw_message"] = json.loads(data.pop("raw_json"))
            data["relationships"] = json.loads(data.pop("relationships_json") or "[]")
            result.append(data)
        return result

    def get_active_summaries(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM summaries WHERE session_id = ? AND active = 1 ORDER BY sequence",
                (session_id,),
            ).fetchall()
        result = []
        for row in rows:
            data = dict(row)
            data["source_ids"] = json.loads(data.pop("source_ids_json"))
            data["active"] = bool(data["active"])
            result.append(data)
        return result

    def apply_action(
        self,
        session_id: str,
        kind: str,
        item_ids: list[str],
        summary_text: str | None = None,
        summary_tokens: int = 0,
        actor: str = "manual",
    ) -> dict[str, Any]:
        allowed = {"remove", "restore", "pin", "unpin", "keep", "summarize", "reset"}
        if kind not in allowed:
            raise ValueError(f"Unsupported action: {kind}")
        with self._lock, self.connect() as connection:
            if kind == "reset":
                rows = connection.execute(
                    "SELECT item_id, status FROM session_items WHERE session_id = ? AND status != 'active'",
                    (session_id,),
                ).fetchall()
                before = {row["item_id"]: row["status"] for row in rows}
                connection.execute(
                    "UPDATE session_items SET status='active', reason='Sent by frontend' WHERE session_id = ?",
                    (session_id,),
                )
                connection.execute(
                    "UPDATE summaries SET active=0 WHERE session_id = ?", (session_id,)
                )
                summary_id = None
            else:
                placeholders = ",".join("?" for _ in item_ids)
                rows = connection.execute(
                    f"SELECT item_id, status, role, sequence, token_count FROM session_items WHERE session_id = ? AND item_id IN ({placeholders})",
                    (session_id, *item_ids),
                ).fetchall()
                before = {row["item_id"]: row["status"] for row in rows}
                if len(before) != len(set(item_ids)):
                    raise ValueError("One or more context items were not found")
                protected = [
                    row["item_id"]
                    for row in rows
                    if row["role"] in {"system", "developer"} and kind in {"remove", "summarize"}
                ]
                latest_user = connection.execute(
                    "SELECT item_id FROM session_items WHERE session_id = ? AND role='user' ORDER BY sequence DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
                if (
                    latest_user
                    and latest_user["item_id"] in item_ids
                    and kind in {"remove", "summarize"}
                ):
                    protected.append(latest_user["item_id"])
                pinned = [
                    row["item_id"]
                    for row in rows
                    if row["status"] == "pinned" and kind in {"remove", "summarize"}
                ]
                if protected or pinned:
                    raise ValueError(
                        f"Protected items cannot be changed: {', '.join(sorted(set(protected + pinned)))}"
                    )
                status_map = {
                    "remove": ("removed", "Removed by user policy"),
                    "restore": ("active", "Restored by user"),
                    "pin": ("pinned", "Pinned by user"),
                    "unpin": ("active", "Unpinned by user"),
                    "keep": ("active", "Explicitly kept by user"),
                    "summarize": ("summarized", "Replaced by generated summary"),
                }
                new_status, reason = status_map[kind]
                connection.execute(
                    f"UPDATE session_items SET status = ?, reason = ? WHERE session_id = ? AND item_id IN ({placeholders})",
                    (new_status, reason, session_id, *item_ids),
                )
                summary_id = None
                if kind == "summarize":
                    if not summary_text or not summary_text.strip():
                        raise ValueError("A non-empty editable summary is required")
                    summary_id = f"summary-{uuid.uuid4().hex[:16]}"
                    sequence = min(row["sequence"] for row in rows)
                    connection.execute(
                        "INSERT INTO summaries(id, session_id, text, source_ids_json, sequence, token_count, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            summary_id,
                            session_id,
                            summary_text.strip(),
                            json.dumps(item_ids),
                            sequence,
                            summary_tokens,
                            utc_now(),
                        ),
                    )
            payload = {"before": before, "item_ids": item_ids, "summary_id": summary_id}
            cursor = connection.execute(
                "INSERT INTO actions(session_id, kind, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (session_id, kind, json.dumps(payload), utc_now()),
            )
            saved = sum(row["token_count"] for row in (rows if kind != "reset" else [])) - (
                summary_tokens if kind == "summarize" else 0
            )
            connection.execute(
                "INSERT INTO audit_log(session_id, action, details_json, created_at) VALUES (?, ?, ?, ?)",
                (
                    session_id,
                    kind,
                    json.dumps(
                        {
                            "item_ids": item_ids,
                            "token_delta": -max(saved, 0),
                            "summary_id": summary_id,
                            "actor": actor,
                        }
                    ),
                    utc_now(),
                ),
            )
        return {
            "action_id": cursor.lastrowid,
            "kind": kind,
            "affected": item_ids,
            "summary_id": summary_id,
            "estimated_token_saving": max(saved, 0),
        }

    def undo_last_action(self, session_id: str) -> dict[str, Any] | None:
        with self._lock, self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM actions WHERE session_id = ? AND undone = 0 ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            payload = json.loads(row["payload_json"])
            for item_id, status in payload.get("before", {}).items():
                connection.execute(
                    "UPDATE session_items SET status = ?, reason = 'Restored by undo' WHERE session_id = ? AND item_id = ?",
                    (status, session_id, item_id),
                )
            if payload.get("summary_id"):
                connection.execute(
                    "UPDATE summaries SET active = 0 WHERE id = ?",
                    (payload["summary_id"],),
                )
            connection.execute("UPDATE actions SET undone = 1 WHERE id = ?", (row["id"],))
            connection.execute(
                "INSERT INTO audit_log(session_id, action, details_json, created_at) VALUES (?, 'undo', ?, ?)",
                (
                    session_id,
                    json.dumps({"action_id": row["id"], "original_action": row["kind"]}),
                    utc_now(),
                ),
            )
        return {"undone_action_id": row["id"], "kind": row["kind"]}

    def record_request(
        self,
        request_id: str,
        session_id: str,
        incoming: dict[str, Any],
        forwarded: dict[str, Any],
        incoming_tokens: int,
        forwarded_tokens: int,
    ) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                "INSERT INTO requests(id, session_id, incoming_json, forwarded_json, incoming_tokens, forwarded_tokens, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    request_id,
                    session_id,
                    json.dumps(incoming, ensure_ascii=False),
                    json.dumps(forwarded, ensure_ascii=False),
                    incoming_tokens,
                    forwarded_tokens,
                    utc_now(),
                ),
            )
            old_rows = connection.execute(
                "SELECT id FROM requests WHERE session_id = ? ORDER BY created_at DESC LIMIT -1 OFFSET ?",
                (session_id, self.request_retention),
            ).fetchall()
            if old_rows:
                connection.executemany(
                    "DELETE FROM requests WHERE id = ?",
                    [(entry["id"],) for entry in old_rows],
                )

    def finish_request(
        self,
        request_id: str,
        status_code: int,
        ttft_ms: float | None,
        total_ms: float,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                "UPDATE requests SET status_code=?, ttft_ms=?, total_ms=?, prompt_tokens=?, completion_tokens=? WHERE id=?",
                (
                    status_code,
                    ttft_ms,
                    total_ms,
                    prompt_tokens,
                    completion_tokens,
                    request_id,
                ),
            )

    def request_history(self, session_id: str, limit: int = 10) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM requests WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        result = []
        for row in rows:
            data = dict(row)
            data["incoming"] = json.loads(data.pop("incoming_json"))
            data["forwarded"] = json.loads(data.pop("forwarded_json"))
            result.append(data)
        return result

    def audit_history(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_log WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{**dict(row), "details": json.loads(row["details_json"])} for row in rows]

    def session_detail(self, session_id: str) -> dict[str, Any] | None:
        session = self.get_session(session_id)
        if not session:
            return None
        items = self.get_items(session_id)
        summaries = self.get_active_summaries(session_id)
        requests = self.request_history(session_id)
        session.update(
            {
                "items": items,
                "summaries": summaries,
                "requests": requests,
                "audit": self.audit_history(session_id),
            }
        )
        return session

    def create_workbench_chat(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        value = data or {}
        chat_id = f"chat-{uuid.uuid4().hex[:20]}"
        now = utc_now()
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO workbench_chats(
                    id, title, profile, ram_budget_gib, context_ram_budget_gib, vram_budget_gib,
                    context_limit, output_token_limit, auto_mode, cleanup_mode,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    str(value.get("title") or "New local chat")[:100],
                    str(value.get("profile") or "exact"),
                    float(value.get("ram_budget_gib", 24)),
                    float(value.get("context_ram_budget_gib", 0.1)),
                    float(value.get("vram_budget_gib", 9.66)),
                    int(value.get("context_limit", 256)),
                    int(value.get("output_token_limit", 96)),
                    int(bool(value.get("auto_mode", False))),
                    str(value.get("cleanup_mode") or "balanced"),
                    now,
                    now,
                ),
            )
        return self.get_workbench_chat(chat_id) or {}

    def list_workbench_chats(self, include_archived: bool = False) -> list[dict[str, Any]]:
        where = "" if include_archived else "WHERE archived = 0"
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT c.*, COUNT(m.id) AS message_count
                FROM workbench_chats c
                LEFT JOIN workbench_messages m ON m.chat_id = c.id
                {where}
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                """
            ).fetchall()
        return [self._workbench_chat_dict(row) for row in rows]

    def get_workbench_chat(self, chat_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM workbench_chats WHERE id = ?", (chat_id,)
            ).fetchone()
        if not row:
            return None
        data = self._workbench_chat_dict(row)
        data["messages"] = self.workbench_messages(chat_id)
        return data

    @staticmethod
    def _workbench_chat_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["auto_mode"] = bool(data["auto_mode"])
        data["archived"] = bool(data["archived"])
        return data

    def update_workbench_chat(self, chat_id: str, data: dict[str, Any]) -> dict[str, Any] | None:
        allowed = {
            "title",
            "profile",
            "ram_budget_gib",
            "context_ram_budget_gib",
            "vram_budget_gib",
            "context_limit",
            "output_token_limit",
            "auto_mode",
            "cleanup_mode",
            "lcm_session_id",
            "archived",
        }
        fields = [(key, data[key]) for key in allowed if key in data]
        if not fields:
            return self.get_workbench_chat(chat_id)
        assignments = ", ".join(f"{key} = ?" for key, _ in fields)
        values = [
            int(bool(value)) if key in {"auto_mode", "archived"} else value
            for key, value in fields
        ]
        with self._lock, self.connect() as connection:
            connection.execute(
                f"UPDATE workbench_chats SET {assignments}, updated_at = ? WHERE id = ?",
                (*values, utc_now(), chat_id),
            )
        return self.get_workbench_chat(chat_id)

    def add_workbench_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        token_count: int = 0,
        metrics: dict[str, Any] | None = None,
        status: str = "complete",
    ) -> dict[str, Any]:
        message_id = f"msg-{uuid.uuid4().hex[:20]}"
        now = utc_now()
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO workbench_messages(
                    id, chat_id, role, content, status, token_count, metrics_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    chat_id,
                    role,
                    content,
                    status,
                    int(token_count),
                    json.dumps(metrics, ensure_ascii=False) if metrics else None,
                    now,
                ),
            )
            connection.execute(
                "UPDATE workbench_chats SET updated_at = ? WHERE id = ?", (now, chat_id)
            )
        return {
            "id": message_id,
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "status": status,
            "token_count": int(token_count),
            "metrics": metrics,
            "created_at": now,
        }

    def workbench_messages(self, chat_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM workbench_messages WHERE chat_id = ? ORDER BY created_at, rowid",
                (chat_id,),
            ).fetchall()
        result = []
        for row in rows:
            data = dict(row)
            data["metrics"] = json.loads(data.pop("metrics_json") or "null")
            result.append(data)
        return result

    def delete_session(self, session_id: str) -> None:
        with self._lock, self.connect() as connection:
            connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def delete_all_local_data(self) -> None:
        with self._lock, self.connect() as connection:
            for table in (
                "audit_log",
                "requests",
                "actions",
                "summaries",
                "session_items",
                "sessions",
                "settings",
                "backends",
            ):
                connection.execute(f"DELETE FROM {table}")
