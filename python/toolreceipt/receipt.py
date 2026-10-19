"""The ``Receipt`` result model and its one-line summary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from toolreceipt.claims import Claim
from toolreceipt.ledger import Ledger, LedgerEntry

Verdict = Literal["clean", "unreported", "fabricated"]


@dataclass
class Receipt:
    confirmed: list[Claim]
    claimed_but_never_ran: list[Claim]
    unmatched: list[Claim]
    ran_but_unreported: list[LedgerEntry]
    verdict: Verdict
    summary: str
    ledger: Ledger
    known_tools: list[str]

    @property
    def ok(self) -> bool:
        return self.verdict != "fabricated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed": [c.to_dict() for c in self.confirmed],
            "claimed_but_never_ran": [c.to_dict() for c in self.claimed_but_never_ran],
            "unmatched": [c.to_dict() for c in self.unmatched],
            "ran_but_unreported": [e.to_dict() for e in self.ran_but_unreported],
            "verdict": self.verdict,
            "ok": self.ok,
            "summary": self.summary,
            "ledger": [e.to_dict() for e in self.ledger],
            "known_tools": list(self.known_tools),
        }


def render_summary(
    verdict: Verdict,
    claimed_but_never_ran: Sequence[Claim],
    ran_but_unreported: Sequence[LedgerEntry],
) -> str:
    """One-line, specific summary (PRD §10 step 6)."""
    if verdict == "fabricated":
        first = claimed_but_never_ran[0]
        n = len(claimed_but_never_ran)
        plural = "" if n == 1 else "s"
        return f'{n} claimed action{plural} never ran: {first.tool} ("{first.text}").'
    if verdict == "unreported":
        names = list(dict.fromkeys(e.tool for e in ran_but_unreported))
        plural = "" if len(names) == 1 else "s"
        return (
            f"All claims confirmed; {len(names)} tool{plural} ran unreported: "
            f"{', '.join(names)}."
        )
    return "All claims confirmed."
