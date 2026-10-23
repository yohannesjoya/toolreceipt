"""Async API (PRD 03 §6): ``acheck`` and ``Receipts.acheck`` on real LangGraph runs."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from tests.helpers import load_cases
from tests.helpers_graph import build_fake_graph
from toolreceipt import Ledger, Receipts, acheck, check
from toolreceipt.claims import Claim, ToolSpec
from toolreceipt.langgraph import ReceiptCallbackHandler, final_reply


@tool
def send_email(to: str) -> str:
    """Send an email to the customer."""
    return f"Email sent to {to}"


@tool
def slow_tool(x: str) -> str:
    """Take a while."""
    time.sleep(0.3)
    return "done"


def _calls(name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "c1"}])


def _graph(final: str = "I have emailed you.", tool_name: str = "send_email") -> Any:
    tools = [send_email, slow_tool]
    return build_fake_graph(
        tools,
        [
            _calls(tool_name, {"to": "a@b.c"} if tool_name == "send_email" else {"x": "1"}),
            AIMessage(content=final),
        ],
    )


def test_ainvoke_records_ledger_and_known_tools():
    async def main() -> None:
        rc = Receipts()
        result = await _graph().ainvoke(
            {"messages": [HumanMessage("email me")]}, config=rc.config()
        )
        reply = final_reply(result)
        assert [(e.tool, e.status) for e in rc.ledger.entries] == [("send_email", "ok")]
        assert set(rc._handler.known_tools) == {"send_email", "slow_tool"}
        a, s = await rc.acheck(reply), rc.check(reply)
        assert a.to_dict() == s.to_dict() and a.verdict == "clean"

    asyncio.run(main())


def test_astream_records_the_same():
    async def main() -> None:
        rc = Receipts()
        last: Any = None
        async for chunk in _graph().astream(
            {"messages": [HumanMessage("email me")]}, config=rc.config(), stream_mode="values"
        ):
            last = chunk
        assert [(e.tool, e.status) for e in rc.ledger.entries] == [("send_email", "ok")]
        assert (await rc.acheck(final_reply(last))).verdict == "clean"

    asyncio.run(main())


def test_cancelled_run_closes_open_entry():
    async def main() -> None:
        rc = Receipts()
        task = asyncio.create_task(
            _graph(tool_name="slow_tool").ainvoke(
                {"messages": [HumanMessage("go")]}, config=rc.config()
            )
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        r = await rc.acheck("I have emailed you.")
        assert [(e.tool, e.status, e.error) for e in rc.ledger.entries][0][:2] == (
            "slow_tool",
            "error",
        )
        assert r.verdict == "fabricated"

    asyncio.run(main())


def test_two_concurrent_runs_separate_receipts():
    async def main() -> None:
        rcs = [Receipts(), Receipts()]
        results = await asyncio.gather(
            *(
                _graph().ainvoke({"messages": [HumanMessage("email me")]}, config=rc.config())
                for rc in rcs
            )
        )
        for rc, result in zip(rcs, results, strict=True):
            assert len(rc.ledger.entries) == 1
            assert (await rc.acheck(final_reply(result))).verdict == "clean"

    asyncio.run(main())


def test_handler_runs_inline():
    assert ReceiptCallbackHandler.run_inline is True


@pytest.mark.parametrize(
    "stem,case", load_cases("check"), ids=lambda c: c if isinstance(c, str) else ""
)
def test_acheck_equals_check_on_every_check_fixture(stem, case):
    ledger = Ledger.from_dict(case["ledger"])
    sync = check(case["text"], ledger, known_tools=case["known_tools"])
    a = asyncio.run(acheck(case["text"], ledger, known_tools=case["known_tools"]))
    assert a.to_dict() == sync.to_dict()


def test_acheck_propagates_extractor_errors():
    class Boom:
        def extract(self, text: str, known_tools: Any) -> list[Claim]:
            return []

        async def aextract(self, text: str, known_tools: Any) -> list[Claim]:
            raise ValueError("boom")

    with pytest.raises(ValueError):
        asyncio.run(acheck("I sent it.", Ledger(), known_tools=["send_email"], extractor=Boom()))


def test_sync_aextract_is_ignored():
    class Odd:
        def extract(self, text: str, known_tools: Any) -> list[Claim]:
            return [Claim(text="I sent it", verb="send", candidates=["send_email"])]

        def aextract(self, text: str, known_tools: Any) -> list[Claim]:  # plain def
            return []

    r = asyncio.run(
        acheck("I sent it.", Ledger(), known_tools=[ToolSpec("send_email", "")], extractor=Odd())
    )
    assert [c.tool for c in r.claimed_but_never_ran] == ["send_email"]
