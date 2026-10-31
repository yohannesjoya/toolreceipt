"""Unit tests for ``toolreceipt.messages``: run scope, skipped roles, async twin, exports."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.helpers import FIXTURES
from toolreceipt.ledger import NO_RESULT, Ledger, LedgerEntry
from toolreceipt.messages import (
    _is_turn,
    _pair,
    _run_messages,
    _settled,
    acheck_messages,
    check_messages,
    final_reply,
    ledger_from_messages,
)

TOOLS = [
    {"type": "function", "function": {"name": "issue_refund", "description": "Issue a refund."}}
]


def _turn(text: str, cid: str) -> list[dict]:
    return [
        {"role": "user", "content": text},
        {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": cid, "type": "function", "function": {
                "name": "issue_refund", "arguments": "{\"order_id\": \"4821\", \"amount\": 64}",
            }}],
        },
        {"role": "tool", "tool_call_id": cid, "content": "Refund issued"},
        {"role": "assistant", "content": "I've refunded $64 for order 4821."},
    ]


def test_system_and_developer_messages_are_skipped():
    messages = [{"role": "system", "content": "Be brief."}, {"role": "developer", "content": "x"}]
    messages += _turn("Refund order 4821.", "call_1")
    assert _run_messages(messages) == messages[3:]  # after the user message
    assert check_messages(messages, known_tools=TOOLS).verdict == "clean"


def test_run_scope_last_user_turn():
    first, second = _turn("Refund order 4821.", "call_1"), _turn("And again.", "call_2")
    assert _run_messages(first + second) == second[1:]
    assert _run_messages([]) == []
    no_user = first[1:]
    assert _run_messages(no_user) == no_user


def test_acheck_messages_equals_check_messages():
    messages = _turn("Refund order 4821.", "call_1")
    sync = check_messages(messages, known_tools=TOOLS).to_dict()
    assert asyncio.run(acheck_messages(messages, known_tools=TOOLS)).to_dict() == sync
    off = check_messages(messages, known_tools=TOOLS, check_args=False)
    assert off.verdict == "clean"


def test_top_level_exports():
    import toolreceipt

    assert toolreceipt.check_messages is check_messages
    assert toolreceipt.acheck_messages is acheck_messages
    assert "check_messages" in toolreceipt.__all__ and "acheck_messages" in toolreceipt.__all__


def _objects(value):
    """Dicts become ``SimpleNamespace`` objects, recursively; lists stay lists."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _objects(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_objects(v) for v in value]
    return value


def test_sdk_objects_read_like_dicts():
    case = json.loads((FIXTURES / "messages" / "openai_refund_and_email_ok.json").read_text())
    as_dicts = check_messages(case["messages"], known_tools=case["known_tools"]).to_dict()
    as_objects = check_messages(_objects(case["messages"]), known_tools=case["known_tools"])
    assert as_objects.to_dict() == as_dicts
    assert as_dicts["verdict"] == "clean" and len(as_dicts["ledger"]) == 2


def test_reasoning_item_is_skipped():
    messages = [
        {"role": "user", "content": "Refund order 4821."},
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "issue_refund",
         "arguments": "{\"order_id\": \"4821\", \"amount\": 64}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "Refund issued"},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "I've refunded $64 for order 4821."}]},
    ]
    assert len(_run_messages(messages)) == 4
    assert [e.status for e in ledger_from_messages(messages).entries] == ["ok"]
    assert check_messages(messages, known_tools=TOOLS).verdict == "clean"


def test_known_tools_skips_builtin_tools_without_a_name():
    from toolreceipt.claims import normalize_tool_specs

    specs = normalize_tool_specs([
        {"type": "web_search"},
        {"type": "custom", "custom": {"name": "send_email", "description": "Send an email."}},
        {"type": "function", "name": "issue_refund", "description": "Issue a refund."},
    ])
    assert [(s.name, s.description) for s in specs] == [
        ("send_email", "Send an email."), ("issue_refund", "Issue a refund."),
    ]


def test_settled_entry_defaults_and_errors():
    assert _settled({"tool": "lookup_order"}).status == "ok"
    e = _settled({"tool": "x", "status": "error"})
    assert e.status == "error" and e.error == NO_RESULT
    assert _settled({"tool": "x", "status": "error", "error": "boom"}).error == "boom"
    assert _settled({"tool": None}) is None and _settled({"role": "user", "tool": "x"}) is None
    assert _settled({"tool": "x", "type": "function_call"}) is None
    assert _settled({"tool": "x", "role": None, "type": None}) is not None
    bad_items = (
        {"tool": ""}, {"tool": 5}, {"tool": "x", "status": "pending"},
        {"tool": "x", "status": ["ok"]},
    )
    for bad in bad_items:
        with pytest.raises(ValueError):
            _settled(bad)


def test_settled_entry_errors_do_not_leak_the_entry():
    secret = "dana.reyes@example.com owes $64"
    leaky = (
        {"tool": "", "result": secret},
        {"tool": "x", "status": "pending", "args": {"note": secret}},
    )
    for bad in leaky:
        with pytest.raises(ValueError) as info:
            _settled(bad)
        assert secret not in str(info.value) and "dana" not in str(info.value)


def test_settled_entry_round_trips_ledger_to_dict():
    # Hand-written in reverse time order: _pair keeps call order, Ledger.entries sorts by time.
    dumped = [
        LedgerEntry("b", started_at=2.0, ended_at=2.5, parent_run_id="r").to_dict(),
        LedgerEntry("a", started_at=1.0, call_id="c1", args={"k": 1}, result="ok").to_dict(),
    ]
    entries = _pair([{"role": "user", "content": "hi"}, *dumped])
    assert [e.to_dict() for e in entries] == dumped
    assert [e.tool for e in Ledger(entries).entries] == ["a", "b"]
    assert _pair([LedgerEntry("a", status="error", error="boom")])[0].error == "boom"


def test_final_reply_empty_after_settled_entry():
    msgs = [
        {"role": "user", "content": "x"}, {"role": "assistant", "content": "Done."},
        {"tool": "issue_refund"},
    ]
    assert final_reply(msgs) == ""
    assert final_reply(msgs[:2]) == "Done."


def test_is_turn():
    assert _is_turn({"role": "assistant", "content": "Hi"})
    assert _is_turn({
        "type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": "Hi"}],
    })
    assert not _is_turn({"role": "assistant", "content": None, "tool_calls": []})
    assert not _is_turn({"role": "assistant", "content": "  \n"})
    assert not _is_turn({"role": "user", "content": "Hi"})
    assert not _is_turn({"tool": "lookup_order"})
