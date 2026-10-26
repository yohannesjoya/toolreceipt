"""Unit tests for ``toolreceipt.messages``: run scope, skipped roles, async twin, exports."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from tests.helpers import FIXTURES
from toolreceipt.messages import (
    _run_messages,
    acheck_messages,
    check_messages,
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
