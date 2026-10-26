"""Unit tests for ``toolreceipt.messages``: run scope, skipped roles, async twin, exports."""

from __future__ import annotations

import asyncio

from toolreceipt.messages import acheck_messages, check_messages, run_messages

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
    assert run_messages(messages) == messages[3:]  # after the user message
    assert check_messages(messages, known_tools=TOOLS).verdict == "clean"


def test_run_scope_last_user_turn():
    first, second = _turn("Refund order 4821.", "call_1"), _turn("And again.", "call_2")
    assert run_messages(first + second) == second[1:]
    assert run_messages([]) == []
    no_user = first[1:]
    assert run_messages(no_user) == no_user


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
