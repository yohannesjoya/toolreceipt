"""The settled-entry reader over every recorded honest and dishonest reply (PRD 05 §10).

Each held-out reply, demo replay and demo example is rebuilt as a one-turn transcript. (a)
Every input gives the same verdict and the same confirmed and never-ran claims as ``check()``
on the same reply and ledger. (b) The honest ones are ``ok``: the held-out set, the
``honest-*`` replays and the five honest examples. The nine ``pressure-*`` replays and
``notcalled.json`` are dishonest by design and are ``fabricated`` under both entry points.
"""

from __future__ import annotations

import json

import pytest

from tests.helpers import DEMO, FIXTURES, demo_tool_specs
from toolreceipt import Ledger, check
from toolreceipt.turns import check_turns

HELDOUT = json.loads((FIXTURES / "heldout" / "honest_replies.json").read_text(encoding="utf-8"))
REPLAYS = sorted((DEMO / "replays").glob("*.json"))
EXAMPLES = sorted((DEMO / "examples").glob("*.json"))
HONEST_EXAMPLES = {"admits", "clean", "outcome", "unmatched", "vague"}


def _cases():
    for case in HELDOUT:
        yield case["id"], "", case["text"], case["ledger"], case["known_tools"], True
    for path in REPLAYS + EXAMPLES:
        data = json.loads(path.read_text(encoding="utf-8"))
        honest = path.stem.startswith("honest-") or path.stem in HONEST_EXAMPLES
        tools = demo_tool_specs(data.get("mode", "pressure"))
        user = data.get("customer_message", "")
        yield path.stem, user, data["reply"], data["ledger"], tools, honest


CORPUS = list(_cases())


@pytest.mark.parametrize(
    "cid,user,reply,ledger,tools,honest", CORPUS, ids=[c[0] for c in CORPUS]
)
def test_one_turn_transcript_equals_check(cid, user, reply, ledger, tools, honest):
    transcript = [
        {"role": "user", "content": user}, *ledger, {"role": "assistant", "content": reply},
    ]
    [r] = check_turns(transcript, known_tools=tools)
    ref = check(reply, Ledger.from_dict(ledger), known_tools=tools)

    def key(claims):
        return [(c.tool, c.reason, c.text) for c in claims]

    got = (r.verdict, key(r.confirmed), key(r.claimed_but_never_ran))
    assert got == (ref.verdict, key(ref.confirmed), key(ref.claimed_but_never_ran)), r.to_dict()
    if honest:
        assert r.ok, r.summary


def test_corpus_counts():
    assert len(CORPUS) == 16 + 19 + 6
    assert sum(honest for *_, honest in CORPUS) == 16 + 10 + 5
