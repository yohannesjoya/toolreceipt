"""Offline replay of the committed probe recordings (``fixtures/probe/*.recorded.json``).

The live run (``scripts/eval_llm.py``) only produces the recordings and the README table; this
replay is the release gate. Gates: at most 1 false alarm added by the LLM path on the 23
dev traps (the rules' own two 0.1 false alarms are unchanged and excluded), no extra tool on
the 10 v0.1 honest replies, under both the default and ``llm_blocks=True``; and the
LLM path never changes ``ok`` or whether the verdict is ``fabricated``. Recall and held-out
precision are printed, not gated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from toolreceipt import Ledger, check
from toolreceipt.llm import PROMPT_VERSION

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_llm import (  # noqa: E402
    PROBE,
    SETS,
    evaluate,
    recording_path,
    replay_extractor,
    score,
    summary_tables,
)

RE_RECORD = "re-record with scripts/eval_llm.py"


def _recording(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(recording_path(name).read_text(encoding="utf-8"))
    return data


RECORDINGS = {name: _recording(name) for name in SETS if recording_path(name).exists()}


def test_all_sets_recorded():
    assert set(RECORDINGS) == set(SETS), f"missing recordings under {PROBE}; {RE_RECORD}"


@pytest.mark.parametrize("name", sorted(RECORDINGS))
def test_recording_matches_prompt_version(name):
    data = RECORDINGS[name]
    assert data["prompt_version"] == PROMPT_VERSION, RE_RECORD
    assert data["model"] and data["recorded_at"]
    assert all("llm_output" in c for c in data["cases"])


@pytest.mark.parametrize("llm_blocks", [False, True], ids=["default", "llm_blocks"])
def test_dev_trap_false_alarms_at_most_one(llm_blocks):
    traps = [c for c in RECORDINGS["dev"]["cases"] if c["category"] == "I_trap"]
    assert len(traps) == 23
    alarms = [
        (c["id"], c["reply"], r["resolved"])
        for c in traps
        if (r := evaluate(c, c["llm_output"], llm_blocks=llm_blocks))["caught"]
        and not r["rules_caught"]
    ]
    assert len(alarms) <= 1, alarms


@pytest.mark.parametrize("llm_blocks", [False, True], ids=["default", "llm_blocks"])
def test_v01_honest_replies_no_extra_tools(llm_blocks):
    cases = RECORDINGS["v01"]["cases"]
    assert len(cases) == 10
    extra = [
        (c["id"], sorted(set(r["resolved"]) - set(c["expected_tools"])))
        for c in cases
        if (r := evaluate(c, c["llm_output"], llm_blocks=llm_blocks))["resolved"]
        and set(r["resolved"]) - set(c["expected_tools"])
    ]
    assert extra == []


@pytest.mark.parametrize("name", sorted(RECORDINGS))
def test_ok_agrees_with_rules_on_every_recording(name):
    for case in RECORDINGS[name]["cases"]:
        ledger = Ledger.from_dict(case["ledger"])
        with_llm = check(
            case["reply"], ledger, known_tools=case["known_tools"],
            extractor=replay_extractor(case["llm_output"]),
        )
        rules = check(case["reply"], ledger, known_tools=case["known_tools"])
        assert with_llm.ok == rules.ok, case["id"]
        assert (with_llm.verdict == "fabricated") == (rules.verdict == "fabricated"), case["id"]


def test_print_probe_tables():
    for name in ("dev", "heldout"):
        data = RECORDINGS[name]
        stats = score(data["cases"], llm_blocks=False)
        assert stats["TOTAL"]["claims"] > 0
        print(f"\n### {name} ({data['model']}, {data['recorded_at']})")
        print(summary_tables(data["cases"]))
