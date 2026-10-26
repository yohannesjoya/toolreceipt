"""Set-semantics diff between claims and the ledger."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import replace
from functools import lru_cache
from typing import Any

from toolreceipt._args import args_verdict
from toolreceipt.claims import (
    Claim,
    Reason,
    _default_lexicon,
    _verb_tokens,
    family_ran,
    tokenize_tool_name,
)
from toolreceipt.ledger import Ledger, LedgerEntry
from toolreceipt.receipt import Receipt, Verdict, render_summary


def _copy(claim: Claim, **changes: Any) -> Claim:
    """``dataclasses.replace`` that keeps the extractor's private match evidence."""
    out = replace(claim, **changes)
    out._object_matched = claim._object_matched
    out._clause = claim._clause
    out._sentence = claim._sentence
    return out


@lru_cache(maxsize=1)
def _all_verb_tokens() -> frozenset[str]:
    """Every verb token of every family in the packaged lexicon. The diff has no extractor
    lexicon in hand, so sibling demotion always uses the packaged families."""
    lex = _default_lexicon()
    tokens: set[str] = set()
    for family in lex["families"]:
        tokens |= _verb_tokens(family, lex)
    return frozenset(tokens)


def _sibling_ran(candidate: str, ok_tools: Iterable[str]) -> bool:
    """True when a tool that ran ok shares a packaged-lexicon verb token with ``candidate``'s
    name (``send_email`` ran, the claim names ``send_sms``)."""
    verbs = set(tokenize_tool_name(candidate)) & _all_verb_tokens()
    return bool(verbs) and any(verbs & set(tokenize_tool_name(t)) for t in ok_tools)


def _mismatch(claim: Claim, ok_entries: Sequence[LedgerEntry]) -> dict[str, Any] | None:
    """The arguments contradicting a confirmed claim, or ``None`` (PRD 04 §8.1).

    A family-only match (``_object_matched is False``) says nothing about which tool did
    the work, so its arguments say nothing either and it is never judged.
    """
    if claim._object_matched is False:
        return None
    candidates = set(claim.candidates)
    hit_entries = [e for e in ok_entries if e.tool in candidates]
    family_entries = [
        e for e in ok_entries if e.tool not in candidates and family_ran(claim.verb, [e.tool])
    ]
    return args_verdict(
        claim, hit_entries, family_entries, _default_lexicon(), context=ok_entries
    )


def diff(
    claims: Sequence[Claim],
    ledger: Ledger,
    *,
    known_tools: Sequence[str] = (),
    ignore_tools: Collection[str] = (),
    llm_blocks: bool = False,
    check_args: bool = True,
) -> Receipt:
    """Compare ``claims`` with what ``ledger`` recorded.

    Matching is set membership: entries are never consumed. Input claims are not mutated.
    A claim with ``source == "llm"`` whose tool never ran lands in ``suspected`` unless
    ``llm_blocks`` is set; then it blocks like a rule claim, except that a sibling tool of
    the same action (by the packaged lexicon's verb tokens) having run makes it ``unmatched``.
    With ``check_args`` (default), a claim whose tool ran is still ``args_mismatch`` when the
    one amount or cued reference number in its clause contradicts every matching call's
    arguments (``toolreceipt._args``); ``check_args=False`` gives the 0.3.0 receipt.
    """
    ok_tools = ledger.ok_tools()
    ok_entries = [e for e in ledger.entries if e.status == "ok"]
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
            observed = _mismatch(claim, ok_entries) if check_args else None
            if observed is None:
                confirmed.append(_copy(claim, candidates=list(claim.candidates), tool=hit))
                continue
            judged = _copy(
                claim, candidates=list(claim.candidates), tool=hit, reason="args_mismatch",
                observed=observed,
            )
            if claim.source == "llm" and not llm_blocks:
                suspected.append(judged)
            else:
                never_ran.append(judged)
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
