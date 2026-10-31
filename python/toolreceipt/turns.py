"""Per-turn receipts (PRD 05 §7): one receipt per assistant turn with text.

``check_turns`` reads the same message lists as ``check_messages`` (OpenAI Chat Completions,
OpenAI Responses, Anthropic, or a plain transcript with settled ledger entries) and returns
one ``TurnReceipt`` per assistant message that carries text. Receipt *n* is computed from
turn *n*'s text and the tool calls made from the start of the list through the message just
before turn *n+1* (its window), so it is a pure function of that prefix; appending messages
never changes an earlier receipt. The ledger is cumulative over the whole list, while
``ran_but_unreported`` lists only the entries whose call sits in the turn's own window. The
last turn closes as-is: a call with no result is ``tool_failed``.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from toolreceipt.claims import ClaimExtractor, normalize_tool_specs
from toolreceipt.ledger import Ledger, LedgerEntry
from toolreceipt.messages import _get, _is_turn, _pair, _text
from toolreceipt.receipt import Receipt, Verdict, render_summary
from toolreceipt.receipts import ToolLike, _acheck, _check

__all__ = ["TurnReceipt", "acheck_turns", "check_turns"]


@dataclass
class TurnReceipt(Receipt):
    """A ``Receipt`` for one assistant turn.

    ``turn`` is the 1-based ordinal of the turn across the list, ``message_index`` the
    position of its message in the input, ``text`` the text that was checked. ``ledger`` is
    cumulative from the start of the list; ``ran_but_unreported`` is scoped to the window.
    """

    turn: int = field(kw_only=True)
    message_index: int = field(kw_only=True)
    text: str = field(kw_only=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "message_index": self.message_index,
            "text": self.text,
            **Receipt.to_dict(self),
        }


def _windows(messages: Sequence[Any]) -> list[tuple[int, int]]:
    """``(turn_index, end_exclusive)`` per turn. Window *n* runs from turn *n*'s message (from
    index 0 for the first turn) through just before turn *n+1*'s message; the windows
    partition the list."""
    turns = [i for i, message in enumerate(messages) if _is_turn(message)]
    if not turns:
        return []
    return list(zip(turns, [*turns[1:], len(messages)], strict=True))


def _reduce(
    receipt: Receipt,
    window: Sequence[LedgerEntry],
    *,
    turn: int,
    message_index: int,
    text: str,
) -> TurnReceipt:
    """``receipt`` with ``ran_but_unreported`` cut to the window's entries (by identity) and
    the verdict and summary recomputed with ``diff()``'s rules."""
    in_window = {id(entry) for entry in window}
    unreported = [e for e in receipt.ran_but_unreported if id(e) in in_window]
    verdict: Verdict
    if receipt.claimed_but_never_ran:
        verdict = "fabricated"
    elif unreported:
        verdict = "unreported"
    else:
        verdict = "clean"
    summary = render_summary(
        verdict, receipt.claimed_but_never_ran, unreported, receipt.suspected,
        len(receipt.confirmed),
    )
    return TurnReceipt(
        confirmed=receipt.confirmed,
        claimed_but_never_ran=receipt.claimed_but_never_ran,
        unmatched=receipt.unmatched,
        ran_but_unreported=unreported,
        verdict=verdict,
        summary=summary,
        ledger=receipt.ledger,
        known_tools=receipt.known_tools,
        suspected=receipt.suspected,
        turn=turn,
        message_index=message_index,
        text=text,
    )


def _turn_inputs(
    messages: Sequence[Any],
) -> Iterator[tuple[int, int, str, list[LedgerEntry], list[LedgerEntry]]]:
    """``(turn, message_index, text, prefix entries, window entries)`` per turn.

    Every message is validated up front (an unsupported shape anywhere raises before any
    receipt is produced). The prefix is re-paired per turn, so each receipt is a function of
    its prefix by construction; the window's entries are the prefix entries added since the
    previous turn, in call order.
    """
    items = list(messages)
    _pair(items)
    seen = 0
    for n, (index, end) in enumerate(_windows(items), start=1):
        entries = _pair(items[:end])
        window, seen = entries[seen:], len(entries)
        yield n, index, _text(_get(items[index], "content")), entries, window


def check_turns(
    messages: Sequence[Any],
    *,
    known_tools: Iterable[ToolLike] = (),
    ignore_tools: Collection[str] = (),
    extractor: ClaimExtractor | None = None,
    llm_blocks: bool = False,
    check_args: bool = True,
) -> list[TurnReceipt]:
    """One ``TurnReceipt`` per assistant turn with text, in message order.

    Same inputs and keywords as ``check_messages``. Receipt *n* checks turn *n*'s text
    against every tool call made before turn *n+1*; a claim whose call only lands after the
    next turn is ``not_called`` on its own receipt. The last turn closes as-is. Raises
    ``ValueError`` on an unsupported message shape anywhere in the list.
    """
    specs = normalize_tool_specs(known_tools)
    receipts: list[TurnReceipt] = []
    for n, index, text, entries, window in _turn_inputs(messages):
        receipt = _check(
            text, Ledger(entries), specs, ignore_tools, extractor, llm_blocks, check_args
        )
        receipts.append(_reduce(receipt, window, turn=n, message_index=index, text=text))
    return receipts


async def acheck_turns(
    messages: Sequence[Any],
    *,
    known_tools: Iterable[ToolLike] = (),
    ignore_tools: Collection[str] = (),
    extractor: ClaimExtractor | None = None,
    llm_blocks: bool = False,
    check_args: bool = True,
) -> list[TurnReceipt]:
    """``check_turns`` for async code: awaits the extractor once per turn, in order."""
    specs = normalize_tool_specs(known_tools)
    receipts: list[TurnReceipt] = []
    for n, index, text, entries, window in _turn_inputs(messages):
        receipt = await _acheck(
            text, Ledger(entries), specs, ignore_tools, extractor, llm_blocks, check_args
        )
        receipts.append(_reduce(receipt, window, turn=n, message_index=index, text=text))
    return receipts
