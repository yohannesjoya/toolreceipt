import pytest

from toolreceipt import Ledger, LedgerEntry, Receipts, check

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

    monkeypatch.setitem(sys.modules, "toolreceipt.langgraph", None)
    with pytest.raises(ImportError, match=r"toolreceipt\[langgraph\] is required"):
        Receipts().config()


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
