import json

from toolreceipt.claims import Claim
from toolreceipt.diff import diff
from toolreceipt.ledger import Ledger, LedgerEntry
from toolreceipt.receipt import render_summary


def test_empty_reply_empty_ledger_is_clean():
    r = diff([], Ledger())
    assert r.verdict == "clean" and r.ok and r.summary == "All claims confirmed."
    assert json.dumps(r.to_dict())
    assert set(r.to_dict()) == {
        "confirmed", "claimed_but_never_ran", "unmatched", "ran_but_unreported",
        "suspected", "verdict", "ok", "summary", "ledger", "known_tools",
    }


def test_set_semantics_never_consume():
    claims = [
        Claim(text="I sent it", verb="send", candidates=["send_email"]),
        Claim(text="the email I sent", verb="send", candidates=["send_email"]),
    ]
    r = diff(claims, Ledger([LedgerEntry(tool="send_email")]))
    assert r.verdict == "clean" and len(r.confirmed) == 2


def test_tie_resolved_by_ledger():
    r = diff(
        [Claim(text="sent it", verb="send", candidates=["send_email", "send_sms"])],
        Ledger([LedgerEntry(tool="send_sms")]),
    )
    assert r.verdict == "clean" and r.confirmed[0].tool == "send_sms"


def test_not_called_vs_tool_failed():
    c = [Claim(text="created a ticket", verb="create", candidates=["create_ticket"])]
    never = diff(c, Ledger(), known_tools=["create_ticket"])
    assert never.claimed_but_never_ran[0].reason == "not_called"
    failed = diff(c, Ledger([LedgerEntry(tool="create_ticket", status="error")]))
    assert failed.claimed_but_never_ran[0].reason == "tool_failed"


def test_unmatched_never_affects_verdict_and_ignore_tools():
    r = diff(
        [Claim(text="summarised", verb="save", candidates=[])],
        Ledger([LedgerEntry(tool="lookup_order")]),
        ignore_tools=["lookup_order"],
    )
    assert r.verdict == "clean" and len(r.unmatched) == 1 and r.ran_but_unreported == []


def test_diff_does_not_mutate_input_claims():
    c = Claim(text="I sent it", verb="send", candidates=["send_email"])
    r = diff([c], Ledger([LedgerEntry(tool="send_email")]))
    assert c.tool is None and c.reason is None
    assert r.confirmed[0] is not c


def test_to_dict_ok_and_ledger_are_json_safe_for_fabricated():
    c = Claim(text="I emailed you", verb="send", candidates=["send_email"])
    d = diff([c], Ledger([LedgerEntry(tool="send_email", status="error", error="boom")])).to_dict()
    assert d["ok"] is False and d["ledger"][0]["status"] == "error"
    assert d["claimed_but_never_ran"][0]["reason"] == "tool_failed"
    json.dumps(d)


def test_summary_forms_with_suspected():
    s = Claim(
        text="check your inbox", verb="other", candidates=["send_email"], tool="send_email",
        reason="not_called", source="llm",
    )
    assert render_summary("clean", [], [], [s], confirmed_count=2) == (
        '2 claims confirmed; 1 suspected (LLM only): send_email ("check your inbox").'
    )
    assert render_summary("clean", [], [], [s], confirmed_count=0) == (
        'No claims; 1 suspected (LLM only): send_email ("check your inbox").'
    )
    assert render_summary("clean", [], [], [], confirmed_count=1) == "All claims confirmed."
    lookup = LedgerEntry(tool="lookup_order")
    assert render_summary("unreported", [], [lookup], [s], confirmed_count=2) == (
        "2 claims confirmed; 1 tool ran unreported: lookup_order; "
        '1 suspected (LLM only): send_email ("check your inbox").'
    )
    assert render_summary("unreported", [], [lookup], [], confirmed_count=2) == (
        "All claims confirmed; 1 tool ran unreported: lookup_order."
    )
    never = Claim(
        text="I have emailed you", verb="send", candidates=["send_email"], tool="send_email",
        reason="not_called",
    )
    assert render_summary("fabricated", [never], [], [s]) == (
        '1 claimed action never ran: send_email ("I have emailed you"); 1 more suspected.'
    )
    assert render_summary("fabricated", [never], [], [s, s]) == (
        '1 claimed action never ran: send_email ("I have emailed you"); 2 more suspected.'
    )


def test_claim_source_defaults_and_roundtrips():
    c = Claim(text="x", verb="send")
    assert c.source == "rules" and c.to_dict()["source"] == "rules"
    assert Claim.from_dict({"text": "x", "verb": "send"}).source == "rules"
    assert Claim.from_dict({"text": "x", "verb": "other", "source": "llm"}).source == "llm"


def test_receipt_to_dict_has_suspected_after_unmatched():
    keys = list(diff([], Ledger()).to_dict())
    assert keys[keys.index("unmatched") + 1] == "suspected"
    assert diff([], Ledger()).suspected == []
