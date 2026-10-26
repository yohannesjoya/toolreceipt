"""Argument check (PRD 04 §8): literals in a claim's clause against the ledger's arguments.

Private module used by ``diff``. Precision first: a kind of literal is judged only when the
clause holds exactly one of it, and only against arguments whose key names that kind.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal

from toolreceipt.claims import Claim, _stem, normalize_loose

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
