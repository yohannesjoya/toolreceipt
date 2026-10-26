"""The argument check over every recorded demo run (PRD 04 §10 gate).

The probe sets and the older fixtures carry no ``args``, so they cannot catch a false alarm
from the argument check. The demo replays and examples are the only honest corpus with real
arguments; none of them may produce ``args_mismatch``.
"""

from __future__ import annotations

import json

import pytest

from tests.helpers import DEMO, demo_tool_specs
from toolreceipt import Ledger, check

CORPUS = sorted((DEMO / "replays").glob("*.json")) + sorted((DEMO / "examples").glob("*.json"))


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.stem)
def test_demo_corpus_has_no_args_mismatch(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    mode = data.get("mode", "pressure")
    r = check(data["reply"], Ledger.from_dict(data["ledger"]), known_tools=demo_tool_specs(mode))
    flagged = [c for c in r.claimed_but_never_ran + r.suspected if c.reason == "args_mismatch"]
    assert flagged == [], [c.to_dict() for c in flagged]


def test_demo_corpus_size():
    assert len(list((DEMO / "replays").glob("*.json"))) >= 19
    assert len(list((DEMO / "examples").glob("*.json"))) >= 6
