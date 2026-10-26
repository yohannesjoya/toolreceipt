"""Argument check (PRD 04 §8): literals in a claim's clause against the ledger's arguments.

Private module used by ``diff``. Precision first: a kind of literal is judged only when the
clause holds exactly one of it, and only against arguments whose key names that kind.
"""

from __future__ import annotations

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
_WORDS = r"(?:dollars?|euros?|pounds?)"
_AMOUNT = re.compile(
    rf"{_SYMBOLS}\s?({_NUMBER})(?![\d%])"
    rf"|\b{_CODES}\s?({_NUMBER})(?![\d%])"
    rf"|\b({_NUMBER})\s?(?:{_CODES}|{_WORDS})\b"
)
_REFERENCE = re.compile(r"\d[\d-]*\d|\d{3,}")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$|^\d{2}/\d{2}/\d{4}$")
_EDGE_PUNCT = ".,;!?()\"'"


def _clause_text(claim: Claim) -> str:
    """The clause the literals come from: the extractor's recorded clause, else the text."""
    clause = getattr(claim, "_clause", None) or claim.text
    return normalize_loose(clause).translate(_HYPHENS)


def find_amount(clause: str) -> Decimal | None:
    """The clause's one money amount (currency mark required), else ``None``."""
    low = normalize_loose(clause).translate(_HYPHENS)
    found = [next(g for g in m.groups() if g) for m in _AMOUNT.finditer(low)]
    if len(found) != 1:
        return None
    return abs(Decimal(found[0].replace(",", "")))


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
    for i, raw in enumerate(tokens):
        tok = raw.strip(_EDGE_PUNCT)
        digits = tok.replace("-", "")
        if not (tok[:1].isdigit() and _REFERENCE.fullmatch(tok) and len(digits) >= 3):
            continue
        if _DATE.match(tok) or not digits.isdigit():
            continue
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

_VALUE_AMOUNT = re.compile(rf"\s*-?\s*(?:{_SYMBOLS}|{_CODES})?\s*({_NUMBER})\s*(?:{_CODES})?\s*")
_DIGIT_RUN = re.compile(r"\d[\d-]*\d|\d")


def literals(claim: Claim, lexicon: Mapping[str, Any]) -> Literals:
    """The claim clause's one amount and one cued reference, each ``None`` when absent."""
    clause = _clause_text(claim)
    return find_amount(clause), find_reference(clause, lexicon.get("id_cues", []))


def _amount_value(value: Any) -> Decimal | None:
    """A keyed amount's value as a positive ``Decimal``; ``None`` when it is not a number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return abs(Decimal(str(value)))
    if isinstance(value, str):
        m = _VALUE_AMOUNT.fullmatch(value.lower())
        if m:
            return abs(Decimal(m.group(1).replace(",", "")))
    return None


def _digit_runs(text: str) -> set[str]:
    """Digit runs of three or more digits, leading zeros stripped: each plain run and each
    hyphen-joined run ("ORD-2024-4821" gives 2024, 4821 and 20244821)."""
    text = text.translate(_HYPHENS)
    runs = {r.replace("-", "") for r in _DIGIT_RUN.findall(text)} | set(re.findall(r"\d+", text))
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
) -> tuple[bool | None, dict[str, Any]]:
    """Judge one ``ok`` entry against the claim's literals.

    ``(True, {})`` when the entry agrees on every literal it can judge and judges at least
    one; ``(False, observed)`` when it has a keyed argument for every literal present and
    at least one disagrees, ``observed`` holding the disagreeing arguments by original key;
    ``(None, {})`` otherwise. Only top-level keys of ``entry.args`` are read.
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
            if any(digits in runs for runs in keyed_runs.values()) or digits in _digit_runs(
                entry.result or ""
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


def args_verdict(
    claim: Claim,
    hit_entries: Sequence[LedgerEntry],
    family_entries: Sequence[LedgerEntry],
    lexicon: Mapping[str, Any],
) -> dict[str, Any] | None:
    """The disagreeing arguments when every ``ok`` entry of the claim's candidates contradicts
    the claim and no same-family tool's entry agrees; ``None`` confirms the claim (§8.2 rule 4).

    ``hit_entries`` come in ``Ledger.entries`` order; the first ``False`` entry's arguments
    are returned. When two or more hit entries carry a keyed amount, their sum also agrees
    (a refund split across calls).
    """
    lits = literals(claim, lexicon)
    if lits == (None, None) or not hit_entries:
        return None
    amount_keys = {_stem(k) for k in lexicon.get("amount_keys", [])}
    sums = [
        val
        for e in hit_entries
        for k, v in e.args.items()
        if _amount_keyed(k, amount_keys) and (val := _amount_value(v)) is not None
    ]
    amount_sum = sum(sums, Decimal(0)) if len(sums) >= 2 else None
    verdicts = [entry_verdict(lits, e, lexicon, amount_sum=amount_sum) for e in hit_entries]
    if any(v is not False for v, _ in verdicts):
        return None
    if any(entry_verdict(lits, e, lexicon)[0] is True for e in family_entries):
        return None
    return verdicts[0][1]
