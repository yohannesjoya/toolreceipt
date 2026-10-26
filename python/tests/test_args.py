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
    ("I've refunded $64 for order 4821.", Decimal(64)),
    ("I've issued a refund for $64.", Decimal(64)),
    ("I've refunded the $64 to your card.", Decimal(64)),
    ("Your $64 has been refunded.", Decimal(64)),
    # an amount that describes another noun, not the action (review B1)
    ("I've refunded half of your $64 order.", None),
    ("I've refunded the shipping on your $64 order.", None),
    ("I've refunded 20% of your $64 order.", None),
    ("I've charged your card for the first installment of your $120 plan.", None),
    ("A refund of $64 has been issued.", None),
    # unit rates, magnitudes and approximations (review B2)
    ("I've refunded $16 each for the four mugs.", None),
    ("I've refunded $16 per mug.", None),
    ("I've refunded $1.5k to your card.", None),
    ("I've refunded $2.5 million to the client.", None),
    ("I've refunded you $64 a month for the last three months.", None),
    ("I've charged your card $64 plus shipping.", None),
    ("I've refunded around $64.", None),
    ("I've refunded roughly $64 to your card.", None),
    ("I've refunded up to $64.", None),
    # "pounds" is a weight as often as a currency (review B3)
    ("I've updated your order to 3 pounds of apples.", None),
    ("Refunded £12.50.", Decimal("12.50")),
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


# ---------------------------------------------------------------------------
# Keyed arguments, agreement and aggregation (§8.2 rules 2 to 4)
# ---------------------------------------------------------------------------

from toolreceipt._args import args_verdict, entry_verdict, literals  # noqa: E402
from toolreceipt.ledger import LedgerEntry  # noqa: E402


def _e(tool="issue_refund", result=None, **args):
    return LedgerEntry(tool=tool, args=args, result=result)


def _claim(text, clause=None):
    c = Claim(text=text, verb="refund", candidates=["issue_refund"])
    c._clause = clause
    return c


AMT = literals(_claim("I've refunded $64"), LEX)
AMT_CENTS = literals(_claim("Refunded $19.99"), LEX)
REF = literals(_claim("Cancelled order 4821"), LEX)
TIC = literals(_claim("Created ticket #5521"), LEX)
BOTH = literals(_claim("Refunded $64 for order 4821"), LEX)


def test_literals_shape():
    assert AMT == (Decimal(64), None)
    assert REF == (None, ("order", "4821"))
    assert BOTH == (Decimal(64), ("order", "4821"))
    assert literals(_claim("Refunded 64 for you"), LEX) == (None, None)


@pytest.mark.parametrize(
    "lits,entry,expected",
    [
        (AMT, _e(amount=32), (False, {"amount": 32})),
        (AMT, _e(order_id="4821"), (None, {})),
        (AMT, _e(amount=6400), (True, {})),
        (AMT, _e(amount="64.00"), (True, {})),
        (AMT, _e(amount="$64"), (True, {})),
        (AMT_CENTS, _e(amount=19.99), (True, {})),
        (AMT, _e(amount=True), (None, {})),
        (AMT, _e(amount=-64), (True, {})),
        (AMT, _e(charge="ch_3N9x"), (None, {})),
        (AMT, _e(amount="ch_3N9x"), (None, {})),
        (AMT, _e(amount=64, items=[{"price": 32}]), (True, {})),
        (AMT, _e(refund_amount=32), (False, {"refund_amount": 32})),
        (AMT, _e(amount_cents=6400), (True, {})),
        (REF, _e("cancel_order", order_id="4822"), (False, {"order_id": "4822"})),
        (REF, _e("cancel_order", order="ORD-2024-4821"), (True, {})),
        (REF, _e("cancel_order", order_id=4821), (True, {})),
        (REF, _e("cancel_order", orderId="004821"), (True, {})),
        (REF, _e("cancel_order", customer_id=77812), (None, {})),
        (REF, _e("cancel_order", id="4822"), (None, {})),
        (REF, _e("cancel_order", order_id="pending"), (None, {})),
        (TIC, _e("create_ticket", "Ticket 5521 created", title="x"), (None, {})),
        (TIC, _e("create_ticket", "Ticket 5521 created", ticket_id=None), (None, {})),
        (TIC, _e("create_ticket", "Ticket 5521 created", ticket_id="TMP-000"), (True, {})),
        (TIC, _e("create_ticket", "ok", order_id="4821"), (None, {})),
        (BOTH, _e(amount=64, order_id="4822"), (False, {"order_id": "4822"})),
        (BOTH, _e(amount=32, order_id="4822"), (False, {"amount": 32, "order_id": "4822"})),
        (BOTH, _e(amount=64), (True, {})),
        (BOTH, _e(amount=64, order_id="4821"), (True, {})),
    ],
)
def test_entry_verdict(lits, entry, expected):
    assert entry_verdict(lits, entry, LEX) == expected


def test_entry_verdict_amount_sum():
    assert entry_verdict(AMT, _e(amount=32), LEX, amount_sum=Decimal(64)) == (True, {})
    assert entry_verdict(AMT, _e(amount=32), LEX, amount_sum=Decimal(6400)) == (True, {})
    assert entry_verdict(AMT, _e(amount=32), LEX, amount_sum=Decimal(50)) == (
        False, {"amount": 32},
    )


def test_args_verdict_aggregation():
    c = _claim("I've refunded $64")
    assert args_verdict(c, [_e(amount=32)], [], LEX) == {"amount": 32}
    assert args_verdict(c, [_e(amount=32), _e(amount=32)], [], LEX) is None
    assert args_verdict(c, [_e(amount=32), _e(order_id="4821")], [], LEX) is None
    assert args_verdict(c, [_e(amount=32), _e(amount=64)], [], LEX) is None
    assert args_verdict(c, [_e(amount=32)], [_e("refund_payment", amount=64)], LEX) is None
    assert args_verdict(c, [_e(amount=32)], [_e("refund_payment", amount=16)], LEX) == {
        "amount": 32
    }
    assert args_verdict(c, [_e(amount=48), _e(amount=32)], [], LEX) == {"amount": 48}
    assert args_verdict(_claim("Refunded 64 for you"), [_e(amount=32)], [], LEX) is None
    assert args_verdict(c, [], [], LEX) is None
