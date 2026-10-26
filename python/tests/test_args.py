"""Unit tests for the argument check (PRD 04 §8.2): literals, keyed arguments, aggregation."""

from __future__ import annotations

from decimal import Decimal

import pytest

from toolreceipt._args import _clause_text, find_amount, find_reference
from toolreceipt.claims import Claim, _default_lexicon

LEX = _default_lexicon()
CUES = LEX["id_cues"]

AMOUNT = [
    ("I've refunded $64.", Decimal(64)),
    ("Refunded €1,250.00.", Decimal("1250.00")),
    ("Refunded $19.99.", Decimal("19.99")),
    ("64 dollars back to you.", Decimal(64)),
    ("EUR 1,250.00 refunded.", Decimal("1250.00")),
    ("Refunded -$64.", Decimal(64)),
    ("Refunded 64 for you.", None),
    ("Refunded 50% of the order.", None),
    ("Refunded the full amount in USD.", None),
    ("I've refunded $32 of your $64 order.", None),
    ("I've refunded $64 minus the $5 restocking fee.", None),
]

REFERENCE = [
    ("Cancelled order 4821.", ("order", "4821")),
    ("Created ticket #5521.", ("ticket", "5521")),
    ("Updated case no. 2024-118.", ("case", "2024118")),
    ("reservation number 4821 is cancelled", ("reservation", "4821")),
    ("Cancelled order 004821.", ("order", "4821")),
    ("order 4821 cancelled", ("order", "4821")),
    ("Cancelled the order placed 2024-10-21.", None),
    ("Cancelled the order of 1500 units.", None),
    ("Refunded order 4821 placed in 2024.", ("order", "4821")),
    ("Texted the code to number 555-0123.", None),
    ("Updated ticket #5521 for order 4821.", None),
    ("Created #5521.", None),
    ("Refunded $64.", None),
]


@pytest.mark.parametrize("clause,expected", AMOUNT)
def test_find_amount(clause, expected):
    assert find_amount(clause) == expected


@pytest.mark.parametrize("clause,expected", REFERENCE)
def test_find_reference(clause, expected):
    assert find_reference(clause, CUES) == expected


def test_clause_text_prefers_private_clause():
    c = Claim(text="refunded $64 and charged the $5 fee", verb="refund")
    c._clause = "refunded $64"
    assert _clause_text(c) == "refunded $64"
    plain = Claim(text="Refund R‑7731 issued", verb="refund")
    assert _clause_text(plain) == "refund r-7731 issued"
