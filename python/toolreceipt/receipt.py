"""The ``Receipt`` result model and its one-line summary."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from toolreceipt.claims import Claim
from toolreceipt.ledger import Ledger, LedgerEntry, truncate

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
    # Claims found only by an LLM extractor whose tool never ran. Reported, never blocking
    # by default; empty on the rules-only path.
    suspected: list[Claim] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict != "fabricated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed": [c.to_dict() for c in self.confirmed],
            "claimed_but_never_ran": [c.to_dict() for c in self.claimed_but_never_ran],
            "unmatched": [c.to_dict() for c in self.unmatched],
            "suspected": [c.to_dict() for c in self.suspected],
            "ran_but_unreported": [e.to_dict() for e in self.ran_but_unreported],
            "verdict": self.verdict,
            "ok": self.ok,
            "summary": self.summary,
            "ledger": [e.to_dict() for e in self.ledger],
            "known_tools": list(self.known_tools),
        }


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _format_observed(observed: Mapping[str, Any]) -> str:
    """``key=json`` pairs of the contradicting ledger arguments, cut to 60 characters."""
    text = ", ".join(f"{k}={json.dumps(v, default=str)}" for k, v in observed.items())
    return truncate(text, 60) or ""


def _quoted(claim: Claim) -> str:
    """``tool ("text")``, plus ``, ledger k=v`` when the claim carries ``observed``."""
    appendix = f", ledger {_format_observed(claim.observed)}" if claim.observed else ""
    return f'{claim.tool} ("{claim.text}"{appendix})'


def render_summary(
    verdict: Verdict,
    claimed_but_never_ran: Sequence[Claim],
    ran_but_unreported: Sequence[LedgerEntry],
    suspected: Sequence[Claim] = (),
    confirmed_count: int = 0,
) -> str:
    """One-line, specific summary.

    Without ``suspected`` the lines are the 0.1 lines. With it, the suffix replaces the
    final period and the ``clean`` line counts the confirmed claims instead of saying "All".
    """
    if verdict == "fabricated":
        first = claimed_but_never_ran[0]
        n = len(claimed_but_never_ran)
        line = f'{n} claimed action{"" if n == 1 else "s"} never ran: {_quoted(first)}'
        if suspected:
            line += f"; {len(suspected)} more suspected"
        return line + "."
    parts: list[str] = []
    if suspected and confirmed_count:
        parts.append(_plural(confirmed_count, "claim") + " confirmed")
    elif suspected:
        parts.append("No claims")
    else:
        parts.append("All claims confirmed")
    if verdict == "unreported":
        names = list(dict.fromkeys(e.tool for e in ran_but_unreported))
        parts.append(f"{_plural(len(names), 'tool')} ran unreported: {', '.join(names)}")
    if suspected:
        parts.append(f"{len(suspected)} suspected (LLM only): {_quoted(suspected[0])}")
    return "; ".join(parts) + "."
