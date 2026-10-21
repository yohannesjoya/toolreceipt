"""Shape and labelling checks for the probe sets in ``fixtures/probe``.

``dev.json`` is the 114-case corpus from the 2026-10-21 gap analysis (prompt and guard tuning
may use it). ``heldout.json`` is a fresh 30-case set written and labelled before any tuning
and never tuned against.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.helpers import FIXTURES

PROBE = FIXTURES / "probe"
CATEGORIES = {
    "A_direct", "B_outcome", "C_passive", "D_format", "E_synonym", "F_vague", "G_ambig",
    "H_lang", "I_trap", "J_mixed", "K_tense", "L_long",
}


def _load(name: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = json.loads((PROBE / name).read_text(encoding="utf-8"))
    return cases


def _check_shape(cases: list[dict[str, Any]], prefix: str) -> None:
    ids = [c["id"] for c in cases]
    assert len(set(ids)) == len(ids) and all(i.startswith(prefix) for i in ids)
    for case in cases:
        assert set(case) == {
            "id", "category", "domain", "reply", "known_tools", "ledger", "expected_tools",
        }, case["id"]
        assert case["category"] in CATEGORIES, case["id"]
        names = {t["name"] for t in case["known_tools"]}
        assert all(t in names for t in case["expected_tools"]), case["id"]
        assert all(e["tool"] in names and e["status"] in {"ok", "error"} for e in case["ledger"])
        assert all(e.get("error") for e in case["ledger"] if e["status"] == "error"), case["id"]
        if case["category"] == "I_trap":
            assert case["expected_tools"] == [], case["id"]


def test_dev_set_shape_and_labels():
    cases = _load("dev.json")
    assert len(cases) == 114
    _check_shape(cases, "dev-")
    assert sum(c["category"] == "I_trap" for c in cases) == 23


@pytest.mark.skipif(not (PROBE / "heldout.json").exists(), reason="held-out set not written yet")
def test_heldout_set_shape_and_labels():
    cases = _load("heldout.json")
    assert len(cases) == 30
    _check_shape(cases, "heldout-")
    assert sum(c["category"] == "I_trap" for c in cases) >= 6
    dev_replies = {c["reply"] for c in _load("dev.json")}
    assert not any(c["reply"] in dev_replies for c in cases)
