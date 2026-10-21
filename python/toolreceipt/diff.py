"""Set-semantics diff between claims and the ledger."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from toolreceipt.claims import (
    Claim,
    Reason,
    _default_lexicon,
    _verb_tokens,
    family_ran,
    tokenize_tool_name,
)
from toolreceipt.ledger import Ledger
from toolreceipt.receipt import Receipt, Verdict, render_summary


def _copy(claim: Claim, **changes: Any) -> Claim:
    """``dataclasses.replace`` that keeps the extractor's private match evidence."""
    out = replace(claim, **changes)
    out._object_matched = claim._object_matched
    return out


def _sibling_ran(
    candidate: str, ok_tools: Iterable[str], lexicon: Mapping[str, Any] | None = None
) -> bool:
    """True when a tool that ran ok shares a lexicon verb token with ``candidate``'s name
    (``send_email`` ran, the claim names ``send_sms``)."""
    lex = _default_lexicon() if lexicon is None else lexicon
    all_verbs: set[str] = set()
    for family in lex["families"]:
        all_verbs |= _verb_tokens(family, lex)
    verbs = set(tokenize_tool_name(candidate)) & all_verbs
    return bool(verbs) and any(verbs & set(tokenize_tool_name(t)) for t in ok_tools)


def diff(
    claims: Sequence[Claim],
    ledger: Ledger,
    *,
    known_tools: Sequence[str] = (),
    ignore_tools: Collection[str] = (),
    llm_blocks: bool = False,
) -> Receipt:
    """Compare ``claims`` with what ``ledger`` recorded.

    Matching is set membership: entries are never consumed. Input claims are not mutated.
    A claim with ``source == "llm"`` whose tool never ran lands in ``suspected`` unless
    ``llm_blocks`` is set; then it blocks like a rule claim.
    """
    ok_tools = ledger.ok_tools()
    error_tools = ledger.error_tools()
    ignored = set(ignore_tools)

    confirmed: list[Claim] = []
    never_ran: list[Claim] = []
    unmatched: list[Claim] = []
    suspected: list[Claim] = []
    for claim in claims:
        if not claim.candidates:
            unmatched.append(_copy(claim, candidates=[]))
            continue
        if claim.source == "llm" and all(c in ignored for c in claim.candidates):
            unmatched.append(_copy(claim, candidates=[]))
            continue
        hit = next((c for c in claim.candidates if c in ok_tools), None)
        if hit is not None:
            confirmed.append(_copy(claim, candidates=list(claim.candidates), tool=hit))
            continue
        failed = any(c in error_tools for c in claim.candidates)
        if claim.source == "llm":
            if not failed and llm_blocks and _sibling_ran(claim.candidates[0], ok_tools):
                unmatched.append(_copy(claim, candidates=list(claim.candidates)))
                continue
            llm_reason: Reason = "tool_failed" if failed else "not_called"
            judged = _copy(
                claim, candidates=list(claim.candidates), tool=claim.candidates[0],
                reason=llm_reason,
            )
            (never_ran if llm_blocks else suspected).append(judged)
        elif failed:
            never_ran.append(_copy(
                claim, candidates=list(claim.candidates), tool=claim.candidates[0],
                reason="tool_failed",
            ))
        elif claim._object_matched is False or family_ran(claim.verb, ok_tools):
            # Not judged: matched on the verb family alone (likely an honest paraphrase of
            # work done with another tool, "raised this with our team" after
            # escalate_ticket), or another tool of the same action ran, so the claim may
            # describe that one ("removed the mug" after remove_item, cancel_order bound).
            unmatched.append(_copy(claim, candidates=list(claim.candidates)))
        else:
            reason: Reason = "not_called"
            never_ran.append(_copy(
                claim, candidates=list(claim.candidates), tool=claim.candidates[0],
                reason=reason,
            ))

    confirmed_tools = {c.tool for c in confirmed}
    unreported = [
        e
        for e in ledger
        if e.status == "ok" and e.tool not in confirmed_tools and e.tool not in ignored
    ]

    verdict: Verdict
    if never_ran:
        verdict = "fabricated"
    elif unreported:
        verdict = "unreported"
    else:
        verdict = "clean"

    return Receipt(
        confirmed=confirmed,
        claimed_but_never_ran=never_ran,
        unmatched=unmatched,
        ran_but_unreported=unreported,
        verdict=verdict,
        summary=render_summary(verdict, never_ran, unreported, suspected, len(confirmed)),
        ledger=ledger,
        known_tools=list(dict.fromkeys(known_tools)),
        suspected=suspected,
    )
