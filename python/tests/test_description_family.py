"""Lead verb of a tool description (PRD 03 §7.2 rule 2), pinned by a committed sample."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import toolreceipt
from tests.helpers import FIXTURES
from toolreceipt.claims import _default_lexicon, _stem_verb, description_family, load_lexicon

PACKAGE = Path(toolreceipt.__file__).parent
SAMPLE = json.loads((FIXTURES / "descriptions" / "sample.json").read_text(encoding="utf-8"))
LEX = _default_lexicon()


def test_sample_size():
    assert len(SAMPLE) >= 30


@pytest.mark.parametrize("entry", SAMPLE, ids=lambda e: e["description"][:30] or "<empty>")
def test_description_family_sample(entry):
    assert description_family(entry["description"], LEX) == entry["family"]


def test_stem_verb_third_person():
    words = ("searches", "fetches", "queries", "sends", "tickets", "address", "closes", "processes")
    assert [_stem_verb(w) for w in words] == [
        "search", "fetch", "query", "send", "ticket", "address", "close", "process"
    ]


def test_lexicon_without_skip_key_loads(tmp_path):
    lex = json.loads((PACKAGE / "lexicon.json").read_text(encoding="utf-8"))
    del lex["description_lead_skip"]
    path = tmp_path / "lex.json"
    path.write_text(json.dumps(lex), encoding="utf-8")
    loaded = load_lexicon(path)
    assert description_family("Create a ticket", loaded) == "create"
    assert description_family("Use this tool to create a ticket", loaded) is None
