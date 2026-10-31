"""Unit tests for ``toolreceipt.turns``: windows, edge shapes, async twin, exports."""

from __future__ import annotations

import asyncio
import json

import pytest

from tests.helpers import load_cases
from toolreceipt.claims import Claim
from toolreceipt.messages import check_messages
from toolreceipt.receipt import Receipt
from toolreceipt.turns import TurnReceipt, _windows, acheck_turns, check_turns


def U(text: str) -> dict:
    return {"role": "user", "content": text}


def A(text: str) -> dict:
    return {"role": "assistant", "content": text}


def CALL(cid: str) -> dict:
    return {"id": cid, "type": "function", "function": {
        "name": "issue_refund", "arguments": json.dumps({"order_id": "4821", "amount": 64}),
    }}


def TOOL(cid: str) -> dict:
    return {"role": "tool", "tool_call_id": cid, "content": "Refund issued"}


CALL_ONLY = {"role": "assistant", "content": None, "tool_calls": [CALL("c1")]}
TOOL_RESULT = TOOL("c1")


def test_windows_partition_the_list():
    msgs = [
        U("a"), CALL_ONLY, TOOL_RESULT, A("t1"), CALL_ONLY, TOOL_RESULT, A("t2"), U("b"), A("t3"),
        U("c"),
    ]
    assert _windows(msgs) == [(3, 6), (6, 8), (8, 10)]  # window 1 covers [0, 6)
    assert _windows([]) == []
    assert _windows([U("a"), CALL_ONLY]) == []


def test_tool_only_and_whitespace_messages_are_not_turns():
    msgs = [
        U("a"), {"role": "assistant", "content": "  \n", "tool_calls": [CALL("c1")]}, TOOL("c1"),
        A("Done, refunded."),
    ]
    [r] = check_turns(msgs, known_tools=["issue_refund"])
    assert (r.turn, r.message_index, r.verdict) == (1, 3, "clean")


def test_leading_entries_before_any_turn_are_in_window_one():
    msgs = [{"tool": "lookup_order"}, CALL_ONLY, TOOL_RESULT, A("Anything else?")]
    [r] = check_turns(msgs, known_tools=["lookup_order", "issue_refund"])
    assert [e.tool for e in r.ran_but_unreported] == ["lookup_order", "issue_refund"]


def test_consecutive_turns_second_has_empty_window():
    msgs = [U("a"), A("Let me check."), CALL_ONLY, TOOL_RESULT, A("Still checking.")]
    r1, r2 = check_turns(msgs, known_tools=["issue_refund"])
    assert [e.tool for e in r1.ran_but_unreported] == ["issue_refund"]
    assert r2.ran_but_unreported == [] and r2.verdict == "clean"
    assert [e.tool for e in r2.ledger] == ["issue_refund"]


def test_empty_and_no_text_lists():
    assert check_turns([]) == []
    assert check_turns([U("a"), CALL_ONLY, TOOL_RESULT]) == []


def test_langchain_object_in_earlier_run_raises_and_trailing_user_is_harmless():
    with pytest.raises(ValueError):
        check_turns([{"type": "ai", "content": "x"}, U("b"), A("Done.")])
    assert len(check_turns([U("a"), A("Hi."), U("b")])) == 1


def test_check_messages_clean_where_check_turns_flags():
    case = dict(load_cases("turns"))["openai_claim_then_pending_call"]
    assert check_messages(case["messages"], known_tools=case["known_tools"]).verdict == "clean"
    assert check_turns(case["messages"], known_tools=case["known_tools"])[0].verdict == "fabricated"


def test_ignore_tools_and_tool_dicts():
    tools = [{"type": "function", "function": {
        "name": "lookup_order", "description": "Look up an order.",
    }}]
    msgs = [U("a"), {"tool": "lookup_order"}, A("Anything else?")]
    [r] = check_turns(msgs, known_tools=tools, ignore_tools=["lookup_order"])
    assert r.verdict == "clean" and r.known_tools == ["lookup_order"]


def test_turn_receipt_to_dict_key_order_and_exports():
    [r] = check_turns([U("a"), A("Hi.")])
    assert isinstance(r, TurnReceipt)
    assert list(r.to_dict())[:3] == ["turn", "message_index", "text"]
    assert list(r.to_dict())[3:] == list(Receipt.to_dict(r))
    from toolreceipt import TurnReceipt as T
    from toolreceipt import acheck_turns as a
    from toolreceipt import check_turns as c
    assert (T, a, c) == (TurnReceipt, acheck_turns, check_turns)


def test_acheck_turns_matches_and_awaits_per_turn():
    msgs = [U("a"), A("I've issued your refund."), CALL_ONLY, TOOL_RESULT, A("Done.")]
    sync = [r.to_dict() for r in check_turns(msgs, known_tools=["issue_refund"])]
    got = asyncio.run(acheck_turns(msgs, known_tools=["issue_refund"]))
    assert [r.to_dict() for r in got] == sync

    class Stub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def extract(self, text, known):
            return []

        async def aextract(self, text, known):
            self.calls.append(text)
            return []

    stub = Stub()
    asyncio.run(acheck_turns(msgs, extractor=stub))
    assert stub.calls == ["I've issued your refund.", "Done."]


def test_llm_claim_lands_in_its_turn_only():
    class Llm:
        def extract(self, text, known):
            if text == "Done.":
                return [Claim(text=text, verb="other", candidates=["send_email"], source="llm")]
            return []

    msgs = [U("a"), A("Hi."), A("Done.")]
    r1, r2 = check_turns(msgs, known_tools=["send_email"], extractor=Llm())
    assert r1.suspected == []
    assert [c.tool for c in r2.suspected] == ["send_email"] and r2.ok
