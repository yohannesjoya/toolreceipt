"""``fixtures/llm``: recorded or hand-written model output replayed through ``LLMExtractor``.

Each case pins the guards (§7.3) and the merge (§7.4) offline, plus the promise that the
LLM path never changes ``ok`` or whether the verdict is ``fabricated``.
"""

from __future__ import annotations

import pytest

from tests.helpers import load_cases
from toolreceipt import Ledger, LLMExtractor, check
from toolreceipt.claims import normalize_tool_specs

CASES = load_cases("llm")
COMPARED = ("verb", "object", "count", "text", "candidates", "confidence", "source")


def _extractor(case: dict) -> LLMExtractor:
    return LLMExtractor(complete=lambda _prompt: case["llm_output"])


def test_fixture_coverage():
    stems = {s for s, _ in CASES}
    for n in range(1, 7):
        assert any(s.startswith(f"guard{n}_") for s in stems), n
    assert sum(s.startswith("merge_") for s in stems) >= 4
    assert sum(s.startswith("dedupe_") for s in stems) >= 2


@pytest.mark.parametrize("stem,case", CASES, ids=lambda c: c if isinstance(c, str) else "")
def test_llm_fixture(stem, case):
    claims = _extractor(case).extract(case["reply"], normalize_tool_specs(case["known_tools"]))
    assert len(claims) == len(case["expected"]), [c.to_dict() for c in claims]
    for claim, exp in zip(claims, case["expected"], strict=True):
        for key, value in exp.items():
            if key in COMPARED:
                assert getattr(claim, key) == value, (key, claim.to_dict())


@pytest.mark.parametrize("stem,case", CASES, ids=lambda c: c if isinstance(c, str) else "")
def test_ok_and_fabricated_agree_with_rules(stem, case):
    names = [t["name"] for t in case["known_tools"]]
    ledgers = (
        [Ledger.from_dict(case["ledger"])]
        if "ledger" in case
        else [
            Ledger(),
            Ledger.from_dict([{"tool": n, "status": "ok"} for n in names]),
            Ledger.from_dict([{"tool": n, "status": "error", "error": "boom"} for n in names]),
        ]
    )
    for ledger in ledgers:
        with_llm = check(
            case["reply"], ledger, known_tools=case["known_tools"], extractor=_extractor(case)
        )
        rules = check(case["reply"], ledger, known_tools=case["known_tools"])
        assert with_llm.ok == rules.ok, (ledger.to_dict(), with_llm.summary)
        assert (with_llm.verdict == "fabricated") == (rules.verdict == "fabricated")
        assert [c.to_dict() for c in with_llm.claimed_but_never_ran] == [
            c.to_dict() for c in rules.claimed_but_never_ran
        ]


def test_family_alone_rule_claim_still_blocks_beside_llm_claim():
    case = dict(CASES)["merge_family_alone_keeps_both"]
    ledger = Ledger.from_dict([{"tool": "create_ticket", "status": "error", "error": "boom"}])
    r = check(case["reply"], ledger, known_tools=case["known_tools"], extractor=_extractor(case))
    assert r.verdict == "fabricated"
    assert [(c.tool, c.reason, c.source) for c in r.claimed_but_never_ran] == [
        ("create_ticket", "tool_failed", "rules")
    ]
    assert [(c.tool, c.source) for c in r.suspected] == [("escalate_ticket", "llm")]
