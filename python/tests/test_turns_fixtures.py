"""Per-turn receipt fixtures (PRD 05 §10): a message list -> one receipt per assistant turn."""

from __future__ import annotations

from tests.helpers import load_cases

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
