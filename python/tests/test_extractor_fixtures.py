import pytest

from tests.helpers import load_cases
from toolreceipt.claims import RuleExtractor, normalize_tool_specs

CASES = load_cases("extractor")
# Keys compared when present in a fixture's expected claim.
COMPARED = ("verb", "object", "count", "text", "candidates", "confidence", "source")


def test_fixture_count():
    assert len(CASES) >= 24


@pytest.mark.parametrize(
    "stem,case", CASES, ids=lambda c: c if isinstance(c, str) else ""
)
def test_extractor_fixture(stem, case):
    claims = RuleExtractor().extract(case["text"], normalize_tool_specs(case["known_tools"]))
    assert all(c.source == "rules" for c in claims)
    assert len(claims) == len(case["expected"]), [c.to_dict() for c in claims]
    for claim, exp in zip(claims, case["expected"], strict=True):
        for key, value in exp.items():
            if key in COMPARED:
                assert getattr(claim, key) == value, (key, claim.to_dict())


@pytest.mark.parametrize(
    "stem,case", [c for c in CASES if c[0].startswith("neg_")],
    ids=lambda c: c if isinstance(c, str) else "",
)
def test_negative_fixtures_are_negative(stem, case):
    assert case["expected"] == []


def test_extract_fills_candidates_but_leaves_diff_fields_empty():
    claims = RuleExtractor().extract("I sent the email.", normalize_tool_specs(["send_email"]))
    assert [(c.candidates, c.confidence, c.tool, c.reason) for c in claims] == [
        (["send_email"], "high", None, None)
    ]
