"""Claim data model, extractor protocol, lexicon loading and the rule extractor."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol

Confidence = Literal["high", "medium"]
Reason = Literal["tool_failed", "not_called"]


class ToolSpec(NamedTuple):
    name: str
    description: str = ""


@dataclass
class Claim:
    text: str
    verb: str
    object: str | None = None
    count: int | None = None
    candidates: list[str] = field(default_factory=list)
    tool: str | None = None
    confidence: Confidence | None = None
    reason: Reason | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "verb": self.verb,
            "object": self.object,
            "count": self.count,
            "candidates": list(self.candidates),
            "tool": self.tool,
            "confidence": self.confidence,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Claim:
        return cls(
            text=data["text"],
            verb=data["verb"],
            object=data.get("object"),
            count=data.get("count"),
            candidates=list(data.get("candidates") or []),
            tool=data.get("tool"),
            confidence=data.get("confidence"),
            reason=data.get("reason"),
        )


class ClaimExtractor(Protocol):
    def extract(self, text: str, known_tools: Sequence[ToolSpec]) -> list[Claim]: ...


def normalize_tool_specs(
    tools: Iterable[str | ToolSpec | tuple[str, str] | Mapping[str, Any]],
) -> list[ToolSpec]:
    """Normalize names, pairs, ToolSpecs and (OpenAI-style) dicts to ``ToolSpec``."""
    specs: list[ToolSpec] = []
    for tool in tools:
        if isinstance(tool, str):
            specs.append(ToolSpec(tool, ""))
        elif isinstance(tool, Mapping):
            inner = tool.get("function")
            src: Mapping[str, Any] = inner if isinstance(inner, Mapping) else tool
            specs.append(ToolSpec(str(src["name"]), str(src.get("description") or "")))
        else:
            name, *rest = tool
            specs.append(ToolSpec(name, rest[0] if rest else ""))
    return specs


@lru_cache(maxsize=1)
def _default_lexicon() -> dict[str, Any]:
    raw = (files("toolreceipt") / "lexicon.json").read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(raw)
    return data


def load_lexicon(path: Path | None = None) -> dict[str, Any]:
    """Load the lexicon (packaged default when ``path`` is None).

    The result is always a fresh copy, so callers may mutate it freely.
    """
    if path is None:
        return copy.deepcopy(_default_lexicon())
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


# ---------------------------------------------------------------------------
# Rule extractor (PRD §9.2). Steps 1-5: sentence split, clause split, action
# detection, filters, object extraction. Tool matching and dedupe come later.
# ---------------------------------------------------------------------------

_BULLET = re.compile(r"^\s*(?:[-*•+]|\d+[.)])\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CONJUNCTIONS = r"(?:and|but|so|because|however|then)"
_CLAUSE_SPLIT = re.compile(rf"\s*[;]\s*|\s*,(?=\s|$)\s*|\s+{_CONJUNCTIONS}\b\s*", re.IGNORECASE)
_LEADING_CONJ = re.compile(rf"^(?:{_CONJUNCTIONS}\b[\s,]*)+", re.IGNORECASE)
_TRAILING_PUNCT = ".!?,;:"
_TOKEN = re.compile(r"'s|[a-z0-9#$@_-]+")
_CONTRACTIONS = (("'ve", " have"), ("'re", " are"), ("'m", " am"), ("'d", " had"), ("'ll", " will"))

_PERFECT = frozenset({"have", "has", "had"})
_BE_DONE = frozenset({"been", "was", "were", "got"})
_BE_PRESENT = frozenset({"is", "are", "am", "'s", "be", "being", "get", "gets", "getting"})
_ADVERBS = frozenset({"already", "just", "also", "now", "all", "both"})
_AUX = _PERFECT | _BE_DONE | _BE_PRESENT
# A determiner or preposition right before a participle makes it an adjective
# ("your scheduled call", "with saved card"), not a completed action.
_ADJECTIVAL_BEFORE = frozenset({
    "the", "a", "an", "your", "my", "our", "their", "his", "her", "its", "this", "that",
    "these", "those", "any", "no", "some", "every", "each", "with", "of", "for", "from",
})
_FIRST_PERSON = frozenset({"i", "we"})
_OTHER_PRONOUNS = frozenset({"you", "they", "he", "she", "it", "someone", "somebody", "everyone"})
_SUBJECT_DETERMINERS = frozenset({
    "the", "a", "an", "your", "their", "his", "her", "its", "this", "that", "these", "those",
    "my", "our", "some", "someone", "somebody", "everyone",
})
_OBJECT_STRIP = frozenset({"you", "me", "the", "a", "an", "your", "our"})


def _normalize(text: str) -> str:
    """Straighten curly quotes; keeps every character position."""
    return (
        text.replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
    )


def _tokens(text: str) -> list[str]:
    low = _normalize(text).lower()
    for short, full in _CONTRACTIONS:
        low = low.replace(short, full)
    return _TOKEN.findall(low.replace("'s", " 's"))


def _phrase(entry: str) -> str:
    """Regex for a lexicon phrase, word-bounded on whichever ends are alphanumeric.

    Contraction suffixes such as ``n't`` match inside a word ("haven't").
    """
    e = entry.strip().lower()
    suffix = e.startswith("n'") or not e[:1].isalnum()
    left = "" if suffix else r"(?<![a-z0-9_'])"
    right = r"(?![a-z0-9_])" if e[-1:].isalnum() else ""
    return left + r"\s+".join(re.escape(w) for w in e.split()) + right


def _contains_any(text: str, entries: Iterable[str]) -> bool:
    low = _normalize(text).lower()
    return any(re.search(_phrase(e), low) for e in entries if e.strip())


def split_sentences(text: str) -> list[str]:
    """Split a reply into sentences on ``.!?`` and newlines, stripping list bullets."""
    out: list[str] = []
    for line in text.splitlines():
        line = _BULLET.sub("", line).strip()
        out.extend(s.strip() for s in _SENTENCE_END.split(line) if s.strip())
    return out


def split_clauses(sentence: str) -> list[str]:
    """Split a sentence on ``,`` ``;`` and the conjunctions and/but/so/because/however/then."""
    out: list[str] = []
    for piece in _CLAUSE_SPLIT.split(sentence):
        piece = _LEADING_CONJ.sub("", piece.strip()).strip()
        if piece.strip(_TRAILING_PUNCT + " "):
            out.append(piece)
    return out


class _Action(NamedTuple):
    family: str
    start: int
    end: int
    subject: list[str]  # tokens before the verb, auxiliaries removed
    aux: list[str]  # auxiliary/adverb chain directly before the verb


def _context(clause: str, start: int) -> tuple[list[str], list[str]]:
    before = _tokens(clause[:start])
    aux: list[str] = []
    while before and len(aux) < 4 and (before[-1] in _AUX or before[-1] in _ADVERBS):
        aux.insert(0, before.pop())
    return before, aux


def _completed(subject: list[str], aux: list[str]) -> bool:
    if subject and subject[-1] in _ADJECTIVAL_BEFORE and not set(aux) & _AUX:
        return False  # adjectival: "your scheduled call", "the already sent email"
    present = set(aux) & _BE_PRESENT
    return not present or bool(set(aux) & {"been", "now"})


def _scan(clause: str, lexicon: Mapping[str, Any]) -> _Action | None:
    entries: list[tuple[str, str | None]] = [
        (form, name)
        for name, fam in lexicon["families"].items()
        for form in fam["forms"]
    ]
    entries += [(form, None) for form in lexicon["exclusions"]]
    entries.sort(key=lambda e: -len(e[0]))
    low = _normalize(clause).lower()
    hits: list[tuple[int, int, str | None]] = []
    for form, family in entries:
        for m in re.finditer(_phrase(form), low):
            if not any(m.start() < e and s < m.end() for s, e, _ in hits):
                hits.append((m.start(), m.end(), family))
    for start, end, family in sorted(hits):
        if low.count('"', 0, start) % 2:
            continue  # inside a quotation
        if family is None:
            return None  # a non-effectful verb comes first
        subject, aux = _context(clause, start)
        if _completed(subject, aux):
            return _Action(family, start, end, subject, aux)
    return None


def find_action(clause: str, lexicon: Mapping[str, Any]) -> tuple[str, int, int] | None:
    """Return ``(family, start, end)`` of the first completed action verb in ``clause``."""
    action = _scan(clause, lexicon)
    return None if action is None else (action.family, action.start, action.end)


def _by_other_agent(clause: str, lexicon: Mapping[str, Any]) -> bool:
    exempt = set(lexicon.get("by_exempt", ()))
    tokens = _tokens(clause)
    for i, tok in enumerate(tokens[:-1]):
        if tok != "by":
            continue
        rest = [t for t in tokens[i + 1:] if t not in {"the", "a", "an"}]
        if rest and rest[0] not in exempt:
            return True
    return False


def _subject_kind(action: _Action, clause: str, lexicon: Mapping[str, Any]) -> str:
    """Classify who did the action: agent, implicit, other or state."""
    subject, aux = action.subject, set(action.aux)
    passive = bool(aux & (_BE_DONE | _BE_PRESENT))
    if not subject:
        return "implicit"
    last = subject[-1]
    if last in _FIRST_PERSON:
        return "agent"
    if last in _OTHER_PRONOUNS:
        if not passive:
            return "other"  # "you sent", "they created"
        if last == "you" and "been" not in aux:
            return "state"  # "you're booked", "you were charged"
    elif not passive and subject[0] in _SUBJECT_DETERMINERS:
        return "other"  # "the customer created", "the nightly sync ran"
    # Passive or headline ("Refund issued") with no first-person agent.
    timed = any(re.search(p, _normalize(clause).lower()) for p in lexicon.get("time_markers", ()))
    if timed or _by_other_agent(clause, lexicon):
        return "state"
    return "passive"


def _lexically_filtered(clause: str, lexicon: Mapping[str, Any]) -> bool:
    if clause.strip().endswith("?"):
        return True
    return any(
        _contains_any(clause, lexicon.get(key, ()))
        for key in ("negation", "future", "advice", "other_subject")
    )


def is_filtered(clause: str, lexicon: Mapping[str, Any]) -> bool:
    """True when the clause is negated, future/conditional, a question, advice, or the
    action belongs to someone other than the agent (other subject or a state description)."""
    if _lexically_filtered(clause, lexicon):
        return True
    action = _scan(clause, lexicon)
    return action is not None and _subject_kind(action, clause, lexicon) in {"other", "state"}


def extract_object(
    clause: str, verb_end: int, lexicon: Mapping[str, Any]
) -> tuple[str | None, int | None]:
    """Return ``(object phrase, count)`` from the text after the verb to the clause end."""
    words = clause[verb_end:].strip(_TRAILING_PUNCT + " -").split()
    while words and words[0].lower() in _OBJECT_STRIP:
        words.pop(0)
    if not words:
        return None, None
    head = words[0].lower()
    count: int | None = None
    if re.fullmatch(r"\d+", head):
        count = int(head)
    elif head not in {"a", "an"} and head in lexicon.get("number_words", {}):
        count = int(lexicon["number_words"][head])
    return " ".join(words), count


def _clean(clause: str) -> str:
    return clause.strip().rstrip(_TRAILING_PUNCT).strip()


class RuleExtractor:
    """Deterministic, stdlib-only claim extractor driven by the lexicon."""

    def __init__(self, lexicon: Mapping[str, Any] | None = None) -> None:
        self._lexicon: Mapping[str, Any] = _default_lexicon() if lexicon is None else lexicon

    def _sentence_filtered(self, sentence: str) -> bool:
        if sentence.strip().endswith("?"):
            return True
        low = _normalize(sentence).lower().lstrip(" \"'(*_")
        return any(
            re.match(_phrase(lead), low) for lead in self._lexicon.get("conditional_lead", ())
        )

    def extract(self, text: str, known_tools: Sequence[ToolSpec]) -> list[Claim]:
        lex = self._lexicon
        claims: list[Claim] = []
        for sentence in split_sentences(_normalize(text)):
            if self._sentence_filtered(sentence):
                continue
            prev_other = False
            for clause in split_clauses(sentence):
                action = _scan(clause, lex)
                if action is None:
                    continue
                kind = _subject_kind(action, clause, lex)
                inherited = kind == "implicit" and prev_other and not action.aux
                prev_other = kind == "other"
                if inherited or kind in {"other", "state"} or _lexically_filtered(clause, lex):
                    continue
                obj, count = extract_object(clause, action.end, lex)
                claims.append(Claim(
                    text=_clean(clause), verb=action.family, object=obj, count=count,
                ))
        return claims
