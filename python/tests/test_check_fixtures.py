"""End-to-end fixtures: reply + bound tools + ledger -> verdict, through ``check()``."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tests.helpers import load_cases
from toolreceipt import Ledger, check
from toolreceipt.diff import diff

CASES = load_cases("check")
DEMO = Path(__file__).resolve().parents[2] / "demo"


@pytest.mark.parametrize("stem,case", CASES, ids=lambda c: c if isinstance(c, str) else "")
def test_check_fixture(stem, case):
    r = check(case["text"], Ledger.from_dict(case["ledger"]), known_tools=case["known_tools"])
    assert r.verdict == case["expected"]["verdict"], r.to_dict()
    if "claimed_but_never_ran" in case["expected"]:
        got = [(c.tool, c.reason, c.observed) for c in r.claimed_but_never_ran]
        expected = case["expected"]["claimed_but_never_ran"]
        want = [(c["tool"], c["reason"], c.get("observed")) for c in expected]
        assert got == want, r.to_dict()
    if "summary" in case["expected"]:
        assert r.summary == case["expected"]["summary"]
    assert r.suspected == []
    assert all(c.source == "rules" for c in r.confirmed + r.claimed_but_never_ran + r.unmatched)


def test_sibling_tool_paraphrase_is_unmatched_not_never_ran():
    case = dict(CASES)["sibling_raised_with_team"]
    r = check(case["text"], Ledger.from_dict(case["ledger"]), known_tools=case["known_tools"])
    assert r.claimed_but_never_ran == []
    assert [(c.candidates, c.tool, c.reason) for c in r.unmatched] == [
        (["create_ticket"], None, None)
    ]


def test_family_only_candidate_that_failed_still_blocks():
    ledger = Ledger.from_dict([{"tool": "create_ticket", "status": "error"}])
    r = check("I've raised this with our team.", ledger, known_tools=["create_ticket"])
    assert r.verdict == "fabricated"
    assert [(c.tool, c.reason) for c in r.claimed_but_never_ran] == [
        ("create_ticket", "tool_failed")
    ]


def test_family_only_candidate_that_ran_is_confirmed():
    ledger = Ledger.from_dict([{"tool": "create_ticket", "status": "ok"}])
    r = check("I've raised this with our team.", ledger, known_tools=["create_ticket"])
    assert r.verdict == "clean"
    assert [c.tool for c in r.confirmed] == ["create_ticket"]


def _demo_tool_specs() -> list[tuple[str, str]]:
    path = DEMO / "scenarios" / "support_desk.py"
    spec = importlib.util.spec_from_file_location("_support_desk", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [(t.name, t.description) for t in module.build_tools("pressure")]


@pytest.mark.skipif(not (DEMO / "replays").is_dir(), reason="demo/ not present")
def test_pressure_replays_stay_fabricated():
    tools = _demo_tool_specs()
    replays = sorted((DEMO / "replays").glob("pressure-*.json"))
    assert replays
    for path in replays:
        data = json.loads(path.read_text(encoding="utf-8"))
        r = check(data["reply"], Ledger.from_dict(data["ledger"]), known_tools=tools)
        assert r.verdict == "fabricated", (path.name, r.summary)


def test_dedupe_keeps_object_evidence_of_any_duplicate():
    # The first claim matches on the family alone, the duplicate on "email" too.
    r = check(
        "I sent it. I sent you the confirmation email.",
        Ledger(),
        known_tools=["send_email"],
    )
    assert r.verdict == "fabricated"
    assert [(c.text, c.tool, c.reason) for c in r.claimed_but_never_ran] == [
        ("I sent it", "send_email", "not_called")
    ]


def test_verb_stem_naming_the_tool_object_is_not_called():
    for stem in ("verb_stem_refunded_order", "verb_stem_refunded_you"):
        case = dict(CASES)[stem]
        r = check(case["text"], Ledger.from_dict(case["ledger"]), known_tools=case["known_tools"])
        assert [(c.tool, c.reason) for c in r.claimed_but_never_ran] == [
            ("issue_refund", "not_called")
        ], stem


def test_unmatched_claims_stay_unmatched_when_diffed_again():
    case = dict(CASES)["sibling_raised_with_team"]
    ledger = Ledger.from_dict(case["ledger"])
    r = check(case["text"], ledger, known_tools=case["known_tools"])
    again = diff(r.unmatched, ledger)
    assert again.claimed_but_never_ran == [] and len(again.unmatched) == 1
