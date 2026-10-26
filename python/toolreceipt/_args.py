"""Argument check (PRD 04 §8): literals in a claim's clause against the ledger's arguments.

Private module used by ``diff``. Precision first: a kind of literal is judged only when the
clause holds exactly one of it, and only against arguments whose key names that kind.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from toolreceipt.claims import Claim, _stem, normalize_loose, tokenize_tool_name
from toolreceipt.ledger import LedgerEntry

# Unicode hyphens (U+2010 to U+2015) and the minus sign map to "-" before any scan.
_HYPHENS = str.maketrans({c: "-" for c in "‐‑‒–—―−"})
# Words that may sit between a cue and its number ("ticket #5521", "case no. 2024-118").
# Never cues themselves and never matched against argument keys.
SEPARATORS = frozenset({"#", "no", "no.", "number", "id", ":"})

_NUMBER = r"\d+(?:,\d{3})*(?:\.\d+)?"
_SYMBOLS = r"[$€£¥]"
_CODES = r"(?:usd|eur|gbp|cad|aud|jpy)"
# "pounds" is a weight as often as a currency ("3 pounds of apples"), so sterling needs £ or
# gbp; dollars and euros are unambiguous.
_WORDS = r"(?:dollars?|euros?)"
_AMOUNT = re.compile(
    rf"{_SYMBOLS}\s?({_NUMBER})(?![\d%])"
    rf"|\b{_CODES}\s?({_NUMBER})(?![\d%])"
    rf"|\b({_NUMBER})\s?(?:{_CODES}|{_WORDS})\b"
)
# An amount is the action's amount only when the words around it say nothing else (PRD 04
# §8.2 rule 1; review B1 to B4). Before it, a preposition reached through any run of filler
# words means it prices another noun ("half of your original $64 charge"), and an
# approximation word means it is not exact. After it, only a short allowlist (or the end of
# the sentence) keeps it the action's amount; any other word ("$64 jacket", "$16 apiece",
# "$64 less the fee", "$2.5 million") disqualifies it.
_PREPOSITIONS = frozenset({
    "of", "on", "from", "in", "at", "with", "per", "by", "over", "under", "about", "around",
})
_FILLERS = frozenset({
    "your", "the", "a", "an", "my", "our", "this", "that", "their", "its", "his", "her",
    "original", "full", "entire", "first", "whole", "same", "total", "remaining", "initial",
})
_APPROXIMATIONS = frozenset({
    "about", "around", "roughly", "approximately", "nearly", "almost", "over", "under", "~",
    "circa",
})
_TWO_WORD_APPROXIMATIONS = frozenset({("up", "to"), ("at", "least"), ("at", "most")})
_ALLOWED_AFTER = frozenset({
    "to", "for", "on", "onto", "back", "into", "via", "as", "in", "from", "has", "have",
    "had", "was", "is", "were", "been", "will", "which", "that", "so", "because", "since",
    "today", "yesterday", "now", "already", "directly", "straight", "immediately", "and",
})
# "and" stays allowed only when it joins a second action ("$64 and emailed you"); a second
# amount after it ("$64 and the $5 fee") already fails the one-amount count.
_RATE_AFTER_FOR = frozenset({"each", "every"})
_REFERENCE = re.compile(r"\d[\d-]*\d|\d{3,}")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$|^\d{2}/\d{2}/\d{4}$")
_EDGE_PUNCT = ".,;!?()\"'"


def _clause_text(claim: Claim) -> str:
    """The clause the literals come from: the extractor's recorded clause, else the text."""
    clause = claim._clause or claim.text
    return normalize_loose(clause).translate(_HYPHENS)


def _qualified(low: str, start: int, end: int) -> bool:
    """False when the words around ``low[start:end]`` say the amount is not the action's."""
    tail = low[end:]
    if tail[:1] and (tail[:1].isalpha() or tail[:1] in "+/"):
        return False  # "$1.5k", "$64/mo", "$64+"
    if re.match(r"[,.' ]\d", tail):
        return False  # "€1.250,00", "€1 250": another number format, not judged
    before = [t.strip(_EDGE_PUNCT) for t in low[:start].split()]
    while before and before[-1] in _FILLERS:
        before.pop()
    if before:
        last = before[-1]
        if last in _PREPOSITIONS or last in _APPROXIMATIONS:
            return False
        if len(before) >= 2 and (before[-2], last) in _TWO_WORD_APPROXIMATIONS:
            return False
    after = [t.strip(_EDGE_PUNCT) for t in tail.split()[:2]]
    after = [t for t in after if t]
    if not after:
        return True
    # A past participle keeps it the action's amount too ("EUR 1,250.00 refunded",
    # "$64 credited back").
    if after[0] not in _ALLOWED_AFTER and not (len(after[0]) > 4 and after[0].endswith("ed")):
        return False
    return not (after[0] in {"for", "to"} and len(after) > 1 and after[1] in _RATE_AFTER_FOR)


def find_amount(clause: str, sentence: str | None = None) -> Decimal | None:
    """The clause's one money amount (currency mark required), judged in its sentence.

    ``None`` when the sentence holds any other amount, when the amount is not in the clause,
    or when the words around it qualify it (§8.2 rule 1).
    """
    low_clause = normalize_loose(clause).translate(_HYPHENS)
    low = normalize_loose(sentence).translate(_HYPHENS) if sentence else low_clause
    if len(list(_AMOUNT.finditer(low))) != 1:
        return None
    in_clause = list(_AMOUNT.finditer(low_clause))
    if len(in_clause) != 1:
        return None
    m = in_clause[0]
    offset = low.find(low_clause)
    if offset < 0:
        low, offset = low_clause, 0
    if not _qualified(low, offset + m.start(), offset + m.end()):
        return None
    return abs(Decimal(next(g for g in m.groups() if g).replace(",", "")))


def _tokens(low: str) -> list[str]:
    return low.replace("#", " # ").replace(":", " : ").split()


def find_reference(clause: str, cues: Sequence[str]) -> tuple[str, str] | None:
    """The clause's one cued reference number as ``(cue stem, digits)``, else ``None``.

    The number follows a cue word directly or through one separator, starts with a digit,
    holds three or more digits and is not shaped like a date. Digits are returned with
    hyphens removed and leading zeros stripped.
    """
    cue_stems = {_stem(c) for c in cues}
    tokens = _tokens(normalize_loose(clause).translate(_HYPHENS))
    found: list[tuple[str, str]] = []
    skip = 0
    for i, raw in enumerate(tokens):
        if skip:
            skip -= 1
            continue
        tok = raw.strip(_EDGE_PUNCT)
        digits = tok.replace("-", "")
        if not (tok[:1].isdigit() and _REFERENCE.fullmatch(tok) and len(digits) >= 3):
            continue
        if _DATE.match(tok) or not digits.isdigit():
            continue
        # "tracking 9400 1118 9922 3334": digit groups of three or more are one number.
        for nxt in tokens[i + 1 :]:
            group = nxt.strip(_EDGE_PUNCT)
            if len(group) >= 3 and group.isdigit():
                digits += group
                skip += 1
            else:
                break
        j = i - 1
        if j >= 0 and (tokens[j] in SEPARATORS or tokens[j].rstrip(".") in SEPARATORS):
            j -= 1
        if j < 0:
            continue
        cue = _stem(tokens[j].strip(_EDGE_PUNCT))
        if cue in cue_stems:
            found.append((cue, digits.lstrip("0") or "0"))
    return found[0] if len(found) == 1 else None


# ---------------------------------------------------------------------------
# Keyed arguments and agreement (§8.2 rules 2 and 3)
# ---------------------------------------------------------------------------

Literals = tuple[Decimal | None, tuple[str, str] | None]
"""``(amount, (cue, digits))`` found in a claim's clause; ``None`` where the clause has none."""

# Applied to a stripped value; single optional spaces keep the pattern free of backtracking.
_VALUE_AMOUNT = re.compile(rf"-?\s?(?:{_SYMBOLS}|{_CODES})?\s?({_NUMBER})\s?(?:{_CODES})?")
_MAX_VALUE_CHARS = 64
_DIGIT_RUN = re.compile(r"\d[\d-]*\d|\d")


def literals(claim: Claim, lexicon: Mapping[str, Any]) -> Literals:
    """The claim clause's one amount and one cued reference, each ``None`` when absent."""
    clause = _clause_text(claim)
    return (
        find_amount(clause, claim._sentence),
        find_reference(clause, lexicon.get("id_cues", [])),
    )


def _amount_value(value: Any) -> Decimal | None:
    """A keyed amount's value as a positive ``Decimal``; ``None`` when it is not a number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = Decimal(str(value))
        return abs(number) if number.is_finite() else None
    if isinstance(value, str) and len(value) <= _MAX_VALUE_CHARS:
        m = _VALUE_AMOUNT.fullmatch(value.strip().lower())
        if m:
            return abs(Decimal(m.group(1).replace(",", "")))
    return None


def _digit_runs(text: str) -> set[str]:
    """Digit runs of three or more digits, leading zeros stripped: each plain run and each
    hyphen-joined run ("ORD-2024-4821" gives 2024, 4821 and 20244821)."""
    text = text.translate(_HYPHENS)
    joined = re.sub(r"(?<=\d) (?=\d)", "", text)  # "1234 5678" is also 12345678
    runs = {r.replace("-", "") for r in _DIGIT_RUN.findall(text) + _DIGIT_RUN.findall(joined)}
    runs |= set(re.findall(r"\d+", text))
    return {r.lstrip("0") or "0" for r in runs if len(r) >= 3}


def _reference_runs(value: Any) -> set[str] | None:
    """A keyed reference's digit runs; ``None`` when the value holds no run of three digits."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        runs = _digit_runs(str(value))
    elif isinstance(value, str):
        runs = _digit_runs(value)
    else:
        return None
    return runs or None


def _amount_keyed(key: str, amount_keys: set[str]) -> bool:
    tokens = tokenize_tool_name(key)
    return _stem(key.lower()) in amount_keys or bool(tokens) and tokens[-1] in amount_keys


def _reference_keyed(key: str, cue: str) -> bool:
    return cue in tokenize_tool_name(key)


def entry_verdict(
    lits: Literals,
    entry: LedgerEntry,
    lexicon: Mapping[str, Any],
    *,
    amount_sum: Decimal | None = None,
    context: Sequence[LedgerEntry] = (),
) -> tuple[bool | None, dict[str, Any]]:
    """Judge one ``ok`` entry against the claim's literals.

    ``(True, {})`` when the entry agrees on every literal it can judge and judges at least
    one; ``(False, observed)`` when it has a keyed argument for every literal present and
    at least one disagrees, ``observed`` holding the disagreeing arguments by original key;
    ``(None, {})`` otherwise. Only top-level keys of ``entry.args`` are read. ``context`` is
    every ``ok`` entry of the ledger: a reference also agrees when one of them holds both the
    claim's number and the keyed value (a lookup that mapped a display number to an id).
    """
    amount, reference = lits
    present = judged = agreed = 0
    observed: dict[str, Any] = {}
    if amount is not None:
        present += 1
        amount_keys = {_stem(k) for k in lexicon.get("amount_keys", [])}
        keyed = {
            k: val for k, v in entry.args.items()
            if _amount_keyed(k, amount_keys) and (val := _amount_value(v)) is not None
        }
        if keyed:
            judged += 1
            targets = {amount, amount * 100}
            if targets & set(keyed.values()) or amount_sum in targets:
                agreed += 1
            else:
                observed.update({k: entry.args[k] for k in keyed})
    if reference is not None:
        present += 1
        cue, digits = reference
        keyed_runs = {
            k: runs for k, v in entry.args.items()
            if _reference_keyed(k, cue) and (runs := _reference_runs(v)) is not None
        }
        if keyed_runs:
            judged += 1
            if (
                any(digits in runs for runs in keyed_runs.values())
                or digits in _digit_runs(entry.result or "")
                or _mapped(digits, keyed_runs, context)
            ):
                agreed += 1
            else:
                observed.update({k: entry.args[k] for k in keyed_runs})
    if judged == 0:
        return None, {}
    if agreed == judged:
        return True, {}
    if judged < present:
        return None, {}
    return False, observed


def _mapped(
    digits: str, keyed_runs: Mapping[str, set[str]], context: Sequence[LedgerEntry]
) -> bool:
    """True when some entry's args or result hold both ``digits`` and a keyed value's digits."""
    wanted = set().union(*keyed_runs.values()) if keyed_runs else set()
    for other in context:
        runs = _digit_runs(json.dumps(other.args, default=str)) | _digit_runs(other.result or "")
        if digits in runs and runs & wanted:
            return True
    return False


def args_verdict(
    claim: Claim,
    hit_entries: Sequence[LedgerEntry],
    family_entries: Sequence[LedgerEntry],
    lexicon: Mapping[str, Any],
    *,
    context: Sequence[LedgerEntry] = (),
) -> dict[str, Any] | None:
    """The disagreeing arguments when every ``ok`` entry of the claim's candidates contradicts
    the claim and no same-family tool's entry agrees; ``None`` confirms the claim (§8.2 rule 4).

    ``hit_entries`` come in ``Ledger.entries`` order; the first ``False`` entry's arguments
    are returned. When two or more hit entries carry a keyed amount, the sum of one keyed
    amount per entry also agrees (a refund split across calls). ``context`` is every ``ok``
    entry of the ledger, for the display-number excuse in ``entry_verdict``.
    """
    lits = literals(claim, lexicon)
    if lits == (None, None) or not hit_entries:
        return None
    amount_keys = {_stem(k) for k in lexicon.get("amount_keys", [])}
    sums = [
        next(
            val
            for k, v in e.args.items()
            if _amount_keyed(k, amount_keys) and (val := _amount_value(v)) is not None
        )
        for e in hit_entries
        if any(
            _amount_keyed(k, amount_keys) and _amount_value(v) is not None
            for k, v in e.args.items()
        )
    ]
    amount_sum = sum(sums, Decimal(0)) if len(sums) >= 2 else None
    verdicts = [
        entry_verdict(lits, e, lexicon, amount_sum=amount_sum, context=context)
        for e in hit_entries
    ]
    if any(v is not False for v, _ in verdicts):
        return None
    if any(entry_verdict(lits, e, lexicon, context=context)[0] is True for e in family_entries):
        return None
    return verdicts[0][1]
