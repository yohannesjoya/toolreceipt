import pytest

from toolreceipt import Claim, Ledger, LedgerEntry, Receipts, check

REPLY = "Refund issued and I have emailed you the confirmation."


def test_check_framework_free_fabricated():
    r = check(
        REPLY,
        Ledger([LedgerEntry(tool="issue_refund"), LedgerEntry(tool="send_email", status="error")]),
    )
    assert r.verdict == "fabricated"
    assert r.claimed_but_never_ran[0].tool == "send_email"
    assert r.claimed_but_never_ran[0].reason == "tool_failed"
    assert r.confirmed[0].tool == "issue_refund"


def test_check_not_called_needs_known_tools():
    ledger = Ledger([LedgerEntry(tool="issue_refund")])
    assert check(REPLY, ledger).verdict == "clean"  # send_email unknown -> unmatched
    r = check(REPLY, ledger, known_tools=["send_email"])
    assert r.claimed_but_never_ran[0].reason == "not_called"


def test_receipts_check_with_explicit_ledger():
    rc = Receipts(known_tools=["send_email"], ignore_tools=["lookup_order"])
    r = rc.check(
        REPLY, ledger=Ledger([LedgerEntry(tool="issue_refund"), LedgerEntry(tool="lookup_order")])
    )
    assert r.verdict == "fabricated" and r.ran_but_unreported == []


def test_receipts_without_ledger_raises():
    with pytest.raises(ValueError, match="no ledger"):
        Receipts().check(REPLY)


def test_empty_reply_is_clean():
    assert check("", Ledger()).verdict == "clean"


def test_receipt_known_tools_lists_explicit_and_ledger_names():
    r = check(REPLY, Ledger([LedgerEntry(tool="issue_refund")]), known_tools=["send_email"])
    assert r.known_tools == ["send_email", "issue_refund"]


def test_extractor_receives_descriptions_and_ledger_names():
    seen = []

    class Spy:
        def extract(self, text, known_tools):
            seen.append(list(known_tools))
            return []

    ledger = Ledger([LedgerEntry(tool="send_email"), LedgerEntry(tool="lookup_order")])
    check("hi", ledger, known_tools=[("send_email", "Send an email")], extractor=Spy())
    assert seen == [[("send_email", "Send an email"), ("lookup_order", "")]]


def test_ledger_property_none_before_config():
    assert Receipts().ledger is None


def test_config_without_langgraph_raises_helpful_import_error(monkeypatch):
    import sys

    # Simulate the extra not being installed: langchain_core cannot be imported.
    monkeypatch.delitem(sys.modules, "toolreceipt.langgraph", raising=False)
    blocked = {"langchain_core", "langgraph"}
    for name in [m for m in sys.modules if m.split(".")[0] in blocked] + sorted(blocked):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=r"toolreceipt\[langgraph\] is required"):
        Receipts().config()


def test_config_unrelated_import_error_propagates(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "toolreceipt.langgraph", None)
    with pytest.raises(ImportError) as info:
        Receipts().config()
    assert "is required" not in str(info.value)


def test_check_uses_stored_handler(monkeypatch):
    class FakeHandler:
        def __init__(self):
            self.ledger = Ledger([LedgerEntry(tool="issue_refund")])
            self.known_tools = {"send_email": "Send an email"}
            self.flushed = False

        def flush(self):
            self.flushed = True

    rc = Receipts()
    handler = FakeHandler()
    rc._handler = handler
    r = rc.check(REPLY)
    assert handler.flushed
    assert r.claimed_but_never_ran[0].reason == "not_called"
    assert rc.ledger is handler.ledger


def test_capture_results_false_redacts_caller_ledger():
    entry = LedgerEntry(tool="issue_refund", args={"order_id": "1"}, result="refunded")
    ledger = Ledger([entry])
    r = Receipts(capture_results=False).check("I issued the refund.", ledger=ledger)
    assert [(e.tool, e.args, e.result) for e in r.ledger] == [("issue_refund", {}, None)]
    assert entry.args == {"order_id": "1"} and entry.result == "refunded"


def test_known_tools_accepts_objects_with_name_and_description():
    class Tool:  # duck-typed like LangChain's BaseTool
        name = "send_email"
        description = "Send an email to the customer"

    r = check(REPLY, Ledger(), known_tools=[Tool()])
    assert r.known_tools == ["send_email"]
    assert [c.tool for c in r.claimed_but_never_ran] == ["send_email"]
    rc = Receipts(known_tools=[Tool()])
    assert rc.check(REPLY, ledger=Ledger()).known_tools == ["send_email"]


def test_known_tools_accepts_langchain_tools():
    tools = pytest.importorskip("langchain_core.tools")

    def send_email(to: str) -> str:
        """Send an email to the customer."""
        return "sent"

    tool = tools.StructuredTool.from_function(send_email)
    r = check(REPLY, Ledger(), known_tools=[tool])
    assert r.known_tools == ["send_email"]
    assert [c.tool for c in r.claimed_but_never_ran] == ["send_email"]


def test_known_tools_rejects_unknown_shapes():
    with pytest.raises(TypeError, match="known_tools"):
        check(REPLY, Ledger(), known_tools=[42])  # type: ignore[list-item]


def test_llm_blocks_passes_through_check_and_receipts():
    class Llm:
        def extract(self, text, known_tools):
            return [Claim(text=text, verb="other", candidates=["send_email"], source="llm")]

    r = check("check your inbox", Ledger(), known_tools=["send_email"], extractor=Llm())
    assert r.ok and [c.tool for c in r.suspected] == ["send_email"]
    r = check(
        "check your inbox", Ledger(), known_tools=["send_email"], extractor=Llm(), llm_blocks=True
    )
    assert not r.ok and r.claimed_but_never_ran[0].reason == "not_called"
    rc = Receipts(extractor=Llm(), known_tools=["send_email"], llm_blocks=True)
    assert not rc.check("check your inbox", Ledger()).ok


def test_check_args_off_restores_confirmation():
    import asyncio

    from toolreceipt import acheck

    ledger = Ledger([LedgerEntry(tool="issue_refund", args={"amount": 32})])
    assert check("I've refunded $64.", ledger, known_tools=["issue_refund"]).verdict == "fabricated"
    assert check(
        "I've refunded $64.", ledger, known_tools=["issue_refund"], check_args=False
    ).verdict == "clean"
    assert asyncio.run(
        acheck("I've refunded $64.", ledger, known_tools=["issue_refund"], check_args=False)
    ).verdict == "clean"
    rc = Receipts(known_tools=["issue_refund"], check_args=False)
    assert rc.check("I've refunded $64.", ledger).verdict == "clean"


def test_capture_results_off_disables_argument_check():
    ledger = Ledger([LedgerEntry(tool="issue_refund", args={"amount": 32})])
    rc = Receipts(known_tools=["issue_refund"], capture_results=False)
    assert rc.check("I've refunded $64.", ledger).verdict == "clean"
