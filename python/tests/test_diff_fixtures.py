import pytest

from tests.helpers import load_cases
from toolreceipt.claims import Claim
from toolreceipt.diff import diff
from toolreceipt.ledger import Ledger

CASES = load_cases("diff")
REQUIRED = {
    "fabricated_tool_failed", "fabricated_not_called", "clean_all_confirmed",
    "unreported_lookup", "tie_resolved_by_ledger", "unmatched_only_is_clean",
    "two_fabricated_plural_summary", "ignore_tools_hides_lookup",
    "error_and_ok_same_tool_is_confirmed",
}


def test_required_fixtures_present():
    assert REQUIRED <= {stem for stem, _ in CASES}


@pytest.mark.parametrize(
    "stem,case", CASES, ids=lambda c: c if isinstance(c, str) else ""
)
def test_diff_fixture(stem, case):
    claims = [Claim.from_dict(c) for c in case["claims"]]
    ledger = Ledger.from_dict(case["ledger"])
    r = diff(claims, ledger, known_tools=case["known_tools"], ignore_tools=case["ignore_tools"])
    exp = case["expected"]
    assert r.verdict == exp["verdict"]
    assert r.ok is exp["ok"]
    assert r.summary == exp["summary"]
    for bucket in ("confirmed", "claimed_but_never_ran", "unmatched"):
        got = getattr(r, bucket)
        assert len(got) == len(exp[bucket]), (bucket, [c.to_dict() for c in got])
        for claim, want in zip(got, exp[bucket], strict=True):
            for key, value in want.items():
                assert getattr(claim, key) == value, (bucket, key, claim.to_dict())
    assert [e.tool for e in r.ran_but_unreported] == exp["ran_but_unreported"]
