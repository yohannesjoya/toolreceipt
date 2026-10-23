"""Lead verb of a tool description (PRD 03 §7.2 rule 2), pinned by a committed sample."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import toolreceipt
from tests.helpers import FIXTURES
from toolreceipt.claims import (
    ToolSpec,
    _default_lexicon,
    _description_object,
    _stem_verb,
    _tool_verb_tokens,
    description_family,
    load_lexicon,
)

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
    del lex["description_lead_nouns"]
    path = tmp_path / "lex.json"
    path.write_text(json.dumps(lex), encoding="utf-8")
    loaded = load_lexicon(path)
    assert description_family("Create a ticket", loaded) == "create"
    assert description_family("Use this tool to create a ticket", loaded) is None
    assert description_family("Email the customer", loaded) is None  # no lead-noun list


def test_tool_verb_tokens_empty_description_is_ineligible():
    assert _tool_verb_tokens(ToolSpec("jira", ""), LEX) == set()


def test_camelcase_verbless_name_uses_description():
    assert "create" in _tool_verb_tokens(ToolSpec("JiraV2", "Create a ticket"), LEX)


def test_verb_named_tool_keeps_its_name_tokens():
    assert _tool_verb_tokens(ToolSpec("send_email", "Delete things"), LEX) == {"send", "email"}


def test_digit_verbless_name_uses_description():
    assert "send" in _tool_verb_tokens(ToolSpec("slack2", "Send a message to a channel"), LEX)


@pytest.mark.parametrize(
    "description,expected",
    [
        ("Create a ticket in Jira", ["ticket"]),
        ("Send an email to the customer about their order", ["email"]),
        ("Create a support ticket for the customer", ["support", "ticket"]),
        ("Jira: create a ticket", ["ticket"]),
        ("Email the customer", ["customer"]),
        ("Set up a meeting with the team", ["meeting"]),
        ("Use this tool to refund a payment.", ["payment"]),
        ("Create tickets and check their status", ["ticket"]),
        ("Create", []),
        ("", []),
    ],
)
def test_description_object_is_the_lead_verbs_direct_object(description, expected):
    assert _description_object(description, LEX) == expected
