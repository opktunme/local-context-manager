from __future__ import annotations

import pytest
from lcm.context import fingerprint_message
from lcm.context.engine import ContextEngine
from lcm.storage import Database
from lcm.tokenizers import TokenizerService


def payload(messages, **extra):
    return {"model": "mock-local-model", "messages": messages, "temperature": 0.2, **extra}


def test_content_fingerprint_is_stable_and_content_sensitive():
    left = {"role": "user", "content": "hello\r\nworld", "ignored": "one"}
    right = {"ignored": "two", "content": "hello\nworld", "role": "user"}
    assert fingerprint_message(left) == fingerprint_message(right)
    assert fingerprint_message(left) != fingerprint_message(
        {"role": "assistant", "content": "hello\nworld"}
    )


def test_fallback_token_count_is_always_labelled_estimated():
    count = TokenizerService().count_text("A small code block: print('hello')")
    assert count.value > 0
    assert count.exact is False
    assert count.source == "fallback_estimate"


def test_pass_through_preserves_unknown_fields(engine: ContextEngine):
    request = payload(
        [
            {"role": "system", "content": "stay local"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                ],
            },
        ],
        seed=42,
        custom_runtime_field={"keep": True},
    )
    result = engine.transform(request, "passthrough", None)
    assert result.forwarded["seed"] == 42
    assert result.forwarded["custom_runtime_field"] == {"keep": True}
    assert result.forwarded["messages"][1]["content"][1]["type"] == "image_url"


def test_automatic_session_inference_uses_shared_prefix(engine: ContextEngine):
    initial = payload(
        [{"role": "system", "content": "rules"}, {"role": "user", "content": "first"}]
    )
    continued = payload(
        initial["messages"]
        + [{"role": "assistant", "content": "reply"}, {"role": "user", "content": "second"}]
    )
    first = engine.transform(initial, None, None)
    second = engine.transform(continued, None, None)
    assert second.session_id == first.session_id


def test_manual_removal_persists_when_client_resends_history(
    engine: ContextEngine, database: Database
):
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old verbose answer" * 30},
        {"role": "user", "content": "current question"},
    ]
    first = engine.transform(payload(messages), "persistent", None)
    old_answer = first.items[2]
    database.apply_action(first.session_id, "remove", [old_answer.id])
    resent = engine.transform(payload(messages), "persistent", None)
    assert old_answer.raw_message not in resent.forwarded["messages"]
    assert messages[-1] in resent.forwarded["messages"]


def test_restore_and_pin(engine: ContextEngine, database: Database):
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "now"},
    ]
    result = engine.transform(payload(messages), "restore-pin", None)
    item = result.items[1]
    database.apply_action(result.session_id, "remove", [item.id])
    database.apply_action(result.session_id, "restore", [item.id])
    database.apply_action(result.session_id, "pin", [item.id])
    with pytest.raises(ValueError, match="Protected"):
        database.apply_action(result.session_id, "remove", [item.id])
    resent = engine.transform(payload(messages), "restore-pin", None)
    assert item.raw_message in resent.forwarded["messages"]


def test_system_and_current_user_are_protected(engine: ContextEngine, database: Database):
    result = engine.transform(
        payload([{"role": "system", "content": "rules"}, {"role": "user", "content": "current"}]),
        "protected",
        None,
    )
    for item in result.items:
        with pytest.raises(ValueError, match="Protected"):
            database.apply_action(result.session_id, "remove", [item.id])


def test_summary_replaces_source_and_undo_restores(engine: ContextEngine, database: Database):
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "old problem"},
        {"role": "assistant", "content": "old solution details" * 20},
        {"role": "user", "content": "current"},
    ]
    result = engine.transform(payload(messages), "summary", None)
    source = result.items[2]
    database.apply_action(
        result.session_id, "summarize", [source.id], "Decision: use the local proxy.", 9
    )
    compacted = engine.transform(payload(messages), "summary", None)
    assert source.raw_message not in compacted.forwarded["messages"]
    assert any(
        "LCM generated context summary" in str(message.get("content"))
        for message in compacted.forwarded["messages"]
    )
    database.undo_last_action(result.session_id)
    restored = engine.transform(payload(messages), "summary", None)
    assert source.raw_message in restored.forwarded["messages"]


def test_invalid_summary_rolls_back_status(engine: ContextEngine, database: Database):
    result = engine.transform(
        payload([{"role": "assistant", "content": "old"}, {"role": "user", "content": "current"}]),
        "rollback",
        None,
    )
    source = result.items[0]
    with pytest.raises(ValueError, match="non-empty"):
        database.apply_action(result.session_id, "summarize", [source.id], "")
    assert database.item_statuses(result.session_id, [source.id])[source.id] == "active"


def test_tool_call_and_result_can_be_removed_atomically(engine: ContextEngine, database: Database):
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "old output"},
        {"role": "user", "content": "continue"},
    ]
    first = engine.transform(payload(messages), "tools", None)
    linked = {first.items[0].id, first.items[1].id}
    database.apply_action(first.session_id, "remove", list(linked))
    resent = engine.transform(payload(messages), "tools", None)
    assert all(message.get("tool_call_id") != "call-1" for message in resent.forwarded["messages"])
    assert all(not message.get("tool_calls") for message in resent.forwarded["messages"])


def test_duplicate_detection_and_large_context(engine: ContextEngine):
    repeated = {"role": "assistant", "content": "same block" * 50}
    messages = (
        [{"role": "system", "content": "rules"}]
        + [repeated.copy() for _ in range(100)]
        + [{"role": "user", "content": "current"}]
    )
    result = engine.transform(payload(messages), "large", None)
    assert len(result.items) == 102
    assert any(item.recommendation == "safe_to_remove" for item in result.items)
    assert result.incoming_tokens > 1000


def test_context_budget_is_persistent(engine: ContextEngine, database: Database):
    result = engine.transform(payload([{"role": "user", "content": "current"}]), "budget", None)
    database.update_session_policy(
        result.session_id,
        {
            "context_budget": 16000,
            "context_limit": 32768,
            "tokenizer_family": "qwen",
            "auto_mode": True,
            "auto_cleanup_mode": "balanced",
        },
    )
    session = database.get_session(result.session_id)
    assert session and session["context_budget"] == 16000
    assert session["tokenizer_family"] == "qwen"
    assert session["auto_mode"] is True
    assert session["auto_cleanup_mode"] == "balanced"


def test_safe_automatic_mode_removes_only_later_exact_duplicates(
    engine: ContextEngine, database: Database
):
    messages = [
        {"role": "assistant", "content": "duplicate output"},
        {"role": "assistant", "content": "duplicate output"},
        {"role": "user", "content": "current"},
    ]
    first = engine.transform(payload(messages), "auto-duplicates", None)
    database.update_session_policy(first.session_id, {"auto_mode": True})
    second = engine.transform(payload(messages), "auto-duplicates", None)
    forwarded_duplicates = [
        message
        for message in second.forwarded["messages"]
        if message.get("content") == "duplicate output"
    ]
    assert len(forwarded_duplicates) == 1
    assert second.forwarded["messages"][-1]["content"] == "current"
    assert (
        database.audit_history(first.session_id)[0]["details"]["actor"]
        == "automatic:conservative"
    )


def test_balanced_auto_mode_removes_large_stale_tool_output(
    engine: ContextEngine, database: Database
):
    noisy_log = "warning: stale compiler output\n" * 500
    messages = [
        {"role": "assistant", "content": noisy_log},
        {"role": "assistant", "content": "Keep this unique conclusion."},
        {"role": "user", "content": "current"},
    ]
    first = engine.transform(payload(messages), "auto-balanced", None)
    database.update_session_policy(
        first.session_id,
        {"auto_mode": True, "auto_cleanup_mode": "balanced"},
    )
    second = engine.transform(payload(messages), "auto-balanced", None)
    assert messages[0] not in second.forwarded["messages"]
    assert messages[1] in second.forwarded["messages"]
    assert messages[2] in second.forwarded["messages"]
    assert (
        database.audit_history(first.session_id)[0]["details"]["actor"]
        == "automatic:balanced"
    )


def test_aggressive_auto_mode_compacts_old_conversation(
    engine: ContextEngine, database: Database
):
    old_answer = "An old implementation discussion with decisions and detail. " * 180
    messages = [
        {"role": "system", "content": "Keep local-only constraints."},
        {"role": "assistant", "content": old_answer},
        {"role": "user", "content": "follow-up one"},
        {"role": "assistant", "content": "short response one"},
        {"role": "user", "content": "follow-up two"},
        {"role": "assistant", "content": "short response two"},
        {"role": "user", "content": "follow-up three"},
        {"role": "assistant", "content": "short response three"},
        {"role": "user", "content": "current request"},
    ]
    first = engine.transform(payload(messages), "auto-aggressive", None)
    database.update_session_policy(
        first.session_id,
        {"auto_mode": True, "auto_cleanup_mode": "aggressive"},
    )
    second = engine.transform(payload(messages), "auto-aggressive", None)
    assert messages[1] not in second.forwarded["messages"]
    assert messages[-1] in second.forwarded["messages"]
    assert any(
        "LCM generated context summary" in str(message.get("content"))
        for message in second.forwarded["messages"]
    )
