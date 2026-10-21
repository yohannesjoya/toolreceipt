"""Held-out set (README, Demo): live honest-mode replies and hand-written adversarial negatives.

Labelled before any rule tuning and never used to tune. The only assertion is zero claims on
the adversarial negatives; precision on the honest replies is printed, not gated.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.helpers import FIXTURES
from toolreceipt.claims import RuleExtractor, normalize_tool_specs

HELDOUT = FIXTURES / "heldout"


def _load(name: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = json.loads((HELDOUT / name).read_text(encoding="utf-8"))
    return cases


NEGATIVES = _load("adversarial_negatives.json")
HONEST = _load("honest_replies.json")


def _extract(case: dict[str, Any]) -> list[Any]:
    claims = RuleExtractor().extract(case["text"], normalize_tool_specs(case["known_tools"]))
    assert all(c.source == "rules" for c in claims)
    return claims


def test_heldout_sizes():
    assert len(NEGATIVES) >= 12
    assert len(HONEST) >= 10
    assert all(case["expected"] == [] for case in NEGATIVES)


@pytest.mark.parametrize("case", NEGATIVES, ids=lambda c: c["id"])
def test_adversarial_negative_produces_no_claim(case):
    assert [c.to_dict() for c in _extract(case)] == []


def test_honest_reply_precision():
    """Precision: claims whose candidates include a tool that ran ok, over all claims."""
    total = correct = 0
    for case in HONEST:
        ran = {e["tool"] for e in case["ledger"] if e["status"] == "ok"}
        for claim in _extract(case):
            total += 1
            correct += bool(ran & set(claim.candidates))
    labelled = sum(len(case["expected"]) for case in HONEST)
    precision = correct / total if total else float("nan")
    print(
        f"\nheld-out honest replies: {len(HONEST)} replies, {labelled} labelled claims, "
        f"{total} extracted, {correct} backed by a successful tool call; "
        f"precision {precision:.2f}"
    )
    assert total > 0
