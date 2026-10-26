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
        for _ in range(200):  # wait until the tool call is in flight, not a fixed sleep
            if rc._handler is not None and rc._handler._in_flight:
                break
            await asyncio.sleep(0.01)
        assert rc._handler is not None and rc._handler._in_flight
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        r = await rc.acheck("I have emailed you.")
        assert [(e.tool, e.status, e.error) for e in rc.ledger.entries] == [
            ("slow_tool", "error", "no result")
        ]
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


def test_receipts_acheck_with_llm_extractor_equals_check():
    from toolreceipt.llm import LLMExtractor

    output = '{"claims": [{"text": "check your inbox", "tool": "send_email"}]}'
    rc = Receipts(extractor=LLMExtractor(complete=lambda p: output), known_tools=["send_email"])
    text = "Refund done, check your inbox."
    a = asyncio.run(rc.acheck(text, Ledger()))
    assert a.to_dict() == rc.check(text, Ledger()).to_dict()
    assert [c.tool for c in a.suspected] == ["send_email"]


# ---------------------------------------------------------------------------
# PRD 04 §9.1: bind a check to a run's config
# ---------------------------------------------------------------------------


def test_check_with_config_reads_that_run():
    rc = Receipts()
    c1, c2 = rc.config(), rc.config()
    r1 = _graph("I have emailed you.").invoke({"messages": [HumanMessage("email me")]}, config=c1)
    r2 = _graph("Done.", "slow_tool").invoke({"messages": [HumanMessage("wait")]}, config=c2)
    first = rc.check(final_reply(r1), config=c1)
    assert first.verdict == "clean" and [e.tool for e in first.ledger] == ["send_email"]
    second = rc.check(final_reply(r2), config=c2)
    assert [e.tool for e in second.ledger] == ["slow_tool"]
    # the stored handler is still the newest one
    assert [e.tool for e in rc.check(final_reply(r2)).ledger] == ["slow_tool"]


def test_two_concurrent_runs_one_receipts_with_config():
    async def main() -> None:
        rc = Receipts()
        configs = [rc.config(), rc.config()]
        results = await asyncio.gather(
            *(
                _graph().ainvoke({"messages": [HumanMessage("email me")]}, config=cfg)
                for cfg in configs
            )
        )
        for cfg, result in zip(configs, results, strict=True):
            receipt = await rc.acheck(final_reply(result), config=cfg)
            assert receipt.verdict == "clean" and len(receipt.ledger) == 1

    asyncio.run(main())


def test_check_config_errors():
    rc = Receipts()
    with pytest.raises(ValueError, match="ReceiptCallbackHandler"):
        rc.check("x", config={"callbacks": []})
    other = Receipts().config()
    with pytest.raises(ValueError, match="another Receipts"):
        rc.check("x", config=other)
    nested = rc.config(rc.config())
    with pytest.raises(ValueError, match="more than one"):
        rc.check("x", config=nested)
    cfg = rc.config()
    with pytest.raises(ValueError, match="not both"):
        rc.check("x", ledger=Ledger(), config=cfg)
    with pytest.raises(ValueError, match="not both"):
        asyncio.run(rc.acheck("x", ledger=Ledger(), config=cfg))
    assert rc.check("x", config=cfg).verdict == "clean"
