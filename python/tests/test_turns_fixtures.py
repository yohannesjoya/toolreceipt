"""Per-turn receipt fixtures (PRD 05 §10): a message list -> one receipt per assistant turn."""

from __future__ import annotations

import pytest

from tests.helpers import load_cases
from toolreceipt.messages import check_messages
from toolreceipt.turns import check_turns

CASES = load_cases("turns")
HONEST = [c for s, c in CASES if s.startswith("honest_multi_turn_")]


def test_honest_multi_turn_fixtures_exist_and_are_honest():
    assert len(HONEST) >= 6
    for case in HONEST:
        assert len(case["expected"]) == 3
        assert all(r["verdict"] != "fabricated" for r in case["expected"])
        assert case["source"].startswith("derived from demo/replays/honest-")


def test_fixture_schema():
    for _stem, case in CASES:
        assert "messages" in case and "known_tools" in case
        assert ("expected" in case) != ("raises" in case)
        for r in case.get("expected", []):
            assert {"turn", "message_index", "text", "verdict"} <= set(r)


REQUIRED = {
    "openai_abandoned_claim_no_call", "openai_claim_with_call_same_message",
    "openai_claim_then_pending_call", "openai_claim_confirmed_from_earlier_window",
    "openai_claim_before_call_in_next_window_flags", "openai_two_runs_ledger_carries",
    "openai_two_runs_new_claim_not_called", "openai_single_turn_equals_check_messages",
    "openai_args_mismatch_on_turn", "responses_abandoned_claim", "responses_two_turns",
    "anthropic_text_and_tool_use_one_turn", "anthropic_claim_then_silence",
    "transcript_silence_mid_auth", "transcript_tool_error_no_recovery",
    "transcript_claim_then_silence", "transcript_settled_entry_defaults",
    "transcript_bad_status_raises", "langchain_objects_rejected",
}


def _entry_matches(entry, want) -> bool:
    if isinstance(want, str):
        return entry.tool == want
    return all(getattr(entry, key) == value for key, value in want.items())


@pytest.mark.parametrize("stem,case", CASES, ids=lambda c: c if isinstance(c, str) else "")
def test_turns_fixture(stem, case):
    if "raises" in case:
        with pytest.raises(ValueError):
            check_turns(case["messages"], known_tools=case["known_tools"])
        return
    receipts = check_turns(case["messages"], known_tools=case["known_tools"])
    assert len(receipts) == len(case["expected"]), [r.to_dict() for r in receipts]
    for r, want in zip(receipts, case["expected"], strict=True):
        got = (r.turn, r.message_index, r.text, r.verdict)
        assert got == (want["turn"], want["message_index"], want["text"], want["verdict"]), r.to_dict()
        assert r.ok == (want["verdict"] != "fabricated")
        never = want.get("claimed_but_never_ran", [])
        assert [(c.tool, c.reason) for c in r.claimed_but_never_ran] == [
            (c["tool"], c["reason"]) for c in never
        ], r.to_dict()
        for claim, w in zip(r.claimed_but_never_ran, never, strict=True):
            if "observed" in w:
                assert claim.observed == w["observed"]
        assert [e.tool for e in r.ran_but_unreported] == want.get("ran_but_unreported", []), r.to_dict()
        if "ledger" in want:
            entries = list(r.ledger)  # Ledger order == call order here (untimed or chronological)
            assert len(entries) == len(want["ledger"]), r.to_dict()
            assert all(_entry_matches(e, w) for e, w in zip(entries, want["ledger"], strict=True)), r.to_dict()
        if "summary" in want:
            assert r.summary == want["summary"]


def test_single_turn_equals_check_messages():
    case = dict(CASES)["openai_single_turn_equals_check_messages"]
    [r] = check_turns(case["messages"], known_tools=case["known_tools"])
    got = {k: v for k, v in r.to_dict().items() if k not in {"turn", "message_index", "text"}}
    assert got == check_messages(case["messages"], known_tools=case["known_tools"]).to_dict()


def test_fixture_coverage():
    assert REQUIRED <= {stem for stem, _ in CASES}
