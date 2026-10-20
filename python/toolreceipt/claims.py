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


class NamedTool(Protocol):
    """Any tool object with a ``name`` and a ``description``, such as LangChain's ``BaseTool``."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...


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

    def __post_init__(self) -> None:
        # Extractor evidence, deliberately not a dataclass field: the field set is fixed and
        # ``to_dict()`` never emits it. True when the candidates matched on an object word of
        # their tool name, False when on the verb family alone, None when unknown (a claim
        # built by hand or by ``from_dict``). ``diff`` only reports a family-alone match whose
        # tool never ran as ``unmatched``, never as ``not_called``.
        self._object_matched: bool | None = None

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
    tools: Iterable[str | ToolSpec | tuple[str, str] | Mapping[str, Any] | NamedTool],
) -> list[ToolSpec]:
    """Normalize names, pairs, ToolSpecs, (OpenAI-style) dicts and tool objects to ``ToolSpec``."""
    specs: list[ToolSpec] = []
    for tool in tools:
        if isinstance(tool, str):
            specs.append(ToolSpec(tool, ""))
        elif isinstance(tool, Mapping):
            inner = tool.get("function")
            src: Mapping[str, Any] = inner if isinstance(inner, Mapping) else tool
            specs.append(ToolSpec(str(src["name"]), str(src.get("description") or "")))
        elif isinstance(tool, (tuple, list)):
            name, *rest = tool
            specs.append(ToolSpec(name, rest[0] if rest else ""))
        else:
            tool_name = getattr(tool, "name", None)
            if not isinstance(tool_name, str):
                raise TypeError(
                    "known_tools entries must be a name, a (name, description) pair, a ToolSpec,"
                    f" a dict or an object with a name attribute; got {type(tool).__name__}"
                )
            specs.append(ToolSpec(tool_name, str(getattr(tool, "description", "") or "")))
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
# Rule extractor. Steps 1-5: sentence split, clause split, action
# detection, filters, object extraction. Steps 6-7 (tool matching, dedupe) follow
# the extraction helpers below.
# ---------------------------------------------------------------------------

_BULLET = re.compile(r"^\s*(?:[-*•+]|\d+[.)])\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CONJUNCTIONS = r"(?:and|but|so|because|however|then)"
# Sentences are whitespace-collapsed before splitting, so single spaces suffice here
# (unbounded leading \s* made long whitespace runs quadratic).
# Also splits on " - ", en/em dashes and ": " (not "e-mail" or "3:00").
_CLAUSE_SPLIT = re.compile(
    rf" ?; ?|,(?= |$) ?| {_CONJUNCTIONS}\b ?| - | ?[\u2013\u2014] ?|:(?= |$) ?", re.IGNORECASE
)
_WHITESPACE = re.compile(r"\s+")
_LEADING_CONJ = re.compile(rf"^(?:{_CONJUNCTIONS}\b[\s,]*)+", re.IGNORECASE)
_TRAILING_PUNCT = ".!?,;:"
_TOKEN = re.compile(r"'s|[a-z0-9#$@_-]+")
_CONTRACTIONS = (("'ve", " have"), ("'re", " are"), ("'m", " am"), ("'d", " had"), ("'ll", " will"))

_PERFECT = frozenset({"have", "has", "had"})
_BE_DONE = frozenset({"been", "was", "were", "got"})
_BE_PRESENT = frozenset({"is", "are", "am", "'s", "be", "being", "get", "gets", "getting"})
_HEADLINE_ADVERBS = frozenset({"already", "just", "now", "successfully", "also"})
_ADVERBS = _HEADLINE_ADVERBS | {"all", "both", "since"}
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
    "my", "some", "someone", "somebody", "everyone",
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
    for piece in _CLAUSE_SPLIT.split(_WHITESPACE.sub(" ", sentence).strip()):
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


def _is_agent_subject(subject: list[str], lexicon: Mapping[str, Any]) -> bool:
    """``I``/``we`` right before the verb chain, or ``our <noun>`` ("our team") unless the
    noun is customer-like ("our customer")."""
    if subject[-1] in _FIRST_PERSON:
        return True
    return "our" in subject[-3:-1] and subject[-1] not in lexicon.get("non_agent_nouns", ())


def _subject_kind(action: _Action, clause: str, lexicon: Mapping[str, Any]) -> str:
    """Classify who did the action: agent, implicit, other or state."""
    subject, aux = action.subject, set(action.aux)
    passive = bool(aux & (_BE_DONE | _BE_PRESENT))
    if not subject:
        return "implicit"
    if _is_agent_subject(subject, lexicon):
        return "agent"
    last = subject[-1]
    if not passive:
        # Active voice with any other subject is someone else's action ("John refunded",
        # "the customer has cancelled"), except a headline whose participle ends the
        # clause ("Refund issued").
        headline = (
            aux <= _HEADLINE_ADVERBS  # "Refund issued", "Refund already issued"
            and last not in _OTHER_PRONOUNS
            and not _tokens(clause[action.end:])
        )
        if not headline:
            return "other"
    elif last == "you" and "been" not in aux:
        return "state"  # "you were charged"
    # Passive or headline with no first-person agent.
    timed = any(re.search(p, _normalize(clause).lower()) for p in lexicon.get("time_markers", ()))
    if timed or _by_other_agent(clause, lexicon):
        return "state"
    return "passive"


def _verbless_other_subject(clause: str, lexicon: Mapping[str, Any]) -> bool | None:
    """For a clause with no lexicon verb: does it open with a non-agent subject?

    True for "The warehouse has shipped", False for "I looked", None when unknown.
    """
    tokens = _tokens(clause)
    if not tokens:
        return None
    if tokens[0] == "our":
        return tokens[1:2] != [] and tokens[1] in lexicon.get("non_agent_nouns", ())
    if tokens[0] in _FIRST_PERSON:
        return False
    if tokens[0] in _SUBJECT_DETERMINERS or tokens[0] in _OTHER_PRONOUNS:
        return True
    if any(t in _PERFECT or t in _BE_DONE for t in tokens[1:3]):
        return True  # "Stripe has shipped"
    return None


def _lexically_filtered(clause: str, lexicon: Mapping[str, Any]) -> bool:
    if clause.strip().endswith("?"):
        return True
    low = _normalize(clause).lower()
    if any(_contains_any(low, lexicon.get(k, ())) for k in ("negation", "other_subject")):
        return True
    # "I can confirm ..." / "Please note ..." are not future or advice.
    for phrase in lexicon.get("filter_exempt", ()):
        low = re.sub(_phrase(phrase), " ", low)
    return any(_contains_any(low, lexicon.get(k, ())) for k in ("future", "advice"))


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


# ---------------------------------------------------------------------------
# Tool matching (step 6; see docs/decisions.md)
# ---------------------------------------------------------------------------

_NAME_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+[0-9]*")
_NAME_SPLIT = re.compile(r"[^A-Za-z0-9]+")
_WORD = re.compile(r"[a-z0-9]+")
# Words that never identify a tool (object filler, pronouns, prepositions).
_MATCH_STOP = frozenset({
    "a", "an", "the", "to", "for", "of", "in", "on", "at", "by", "with", "from", "into",
    "and", "or", "it", "its", "this", "that", "these", "those", "you", "your", "me", "my",
    "i", "we", "us", "our", "they", "them", "their", "he", "she", "him", "her", "is", "are",
    "was", "were", "be", "been", "has", "have", "had", "all", "any", "some", "up", "out",
    "now", "again", "just", "already", "also", "as", "so", "here", "there",
})
_SUBJECT_BOUNDARY = _SUBJECT_DETERMINERS | _OBJECT_STRIP | _FIRST_PERSON | {"that", "which", "who"}

_SUBJECT_PREPOSITIONS = frozenset({"to", "for", "of", "from", "with"})

_Term = tuple[str, ...]


def _stem(word: str) -> str:
    """Lowercase and strip one plural ``s`` ("tickets" -> "ticket"; keeps "address", "sms")."""
    w = word.lower()
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _words(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(_normalize(text).lower())]


def tokenize_tool_name(name: str) -> list[str]:
    """Split a tool name on ``_``, ``-``, other separators and camelCase; lowercase, singular."""
    return [_stem(part) for chunk in _NAME_SPLIT.split(name) for part in _NAME_PART.findall(chunk)]


def _occurs(term: _Term, words: Sequence[str]) -> set[int]:
    """Indices of ``words`` covered by contiguous occurrences of ``term``."""
    n = len(term)
    hits: set[int] = set()
    for i in range(len(words) - n + 1):
        if tuple(words[i:i + n]) == term:
            hits.update(range(i, i + n))
    return hits


def _synonym_groups(lexicon: Mapping[str, Any]) -> list[set[_Term]]:
    return [
        {tuple(_words(w)) for w in (key, *values)}
        for key, values in lexicon.get("synonyms", {}).items()
    ]


def _noun_families(lexicon: Mapping[str, Any]) -> dict[str, str]:
    """Noun-like family tokens ("email", "issue", "record") mapped to their family."""
    nouns = set(lexicon.get("noun_tokens", ()))
    return {
        _stem(tok): name
        for name, fam in lexicon["families"].items()
        for tok in fam.get("tokens", ())
        if tok in nouns
    }


def _verb_tokens(family: str, lexicon: Mapping[str, Any]) -> set[str]:
    """A family's verb tokens: its name and every token not listed in ``noun_tokens``."""
    nouns = set(lexicon.get("noun_tokens", ()))
    fam = lexicon["families"].get(family, {})
    return {_stem(family)} | {_stem(t) for t in fam.get("tokens", ()) if t not in nouns}


def family_ran(family: str, tools: Iterable[str], lexicon: Mapping[str, Any] | None = None) -> bool:
    """True when any of ``tools`` carries a verb token of ``family`` in its name."""
    lex = _default_lexicon() if lexicon is None else lexicon
    if family not in lex["families"]:
        return False
    verbs = _verb_tokens(family, lex)
    return any(verbs & set(tokenize_tool_name(t)) for t in tools)


def _conflicts(family: str, name: Sequence[str], lexicon: Mapping[str, Any]) -> bool:
    """True when the tool name says it does something else than the claim.

    Its name holds a read verb ("read_email", "lookup_order"), or a verb of another family
    without one of the claim's own ("delete_email" for a send claim). Noun tokens never
    conflict, and the claim's own verb keeps the tool ("create_refund" for a refund).
    """
    tokens = set(name)
    if tokens & {_stem(v) for v in lexicon.get("read_verbs", ())}:
        return True
    if tokens & _verb_tokens(family, lexicon):
        return False
    return any(
        tokens & _verb_tokens(other, lexicon) for other in lexicon["families"] if other != family
    )


def _family_terms(family: str, lexicon: Mapping[str, Any]) -> set[_Term]:
    """Family tokens; verb tokens are also expanded through every synonym group holding them.

    Noun tokens stay literal, so "issue" (a refund token) never reaches "ticket".
    """
    fam = lexicon["families"].get(family, {})
    verbs = _verb_tokens(family, lexicon)
    terms: set[_Term] = {(v,) for v in verbs} | {tuple(_words(t)) for t in fam.get("tokens", ())}
    for group in _synonym_groups(lexicon):
        if any(len(term) == 1 and term[0] in verbs for term in group):
            terms |= group
    return {t for t in terms if t}


def _object_terms(
    family: str, words: Sequence[str], lexicon: Mapping[str, Any]
) -> set[_Term]:
    """Content words of the object plus synonyms of any group they (or a phrase) hit.

    Expansion never yields another family's noun token: "ticket" does not reach the
    "issue" of ``issue_refund`` for a create claim. A literal "issue" still matches.
    """
    foreign = {n for n, fam in _noun_families(lexicon).items() if fam != family}
    content = [w for w in words if w not in _MATCH_STOP]
    terms: set[_Term] = {(w,) for w in content}
    for group in _synonym_groups(lexicon):
        if any(_occurs(member, content) for member in group if member):
            terms |= {m for m in group if m and " ".join(m) not in foreign}
    return terms


def _names_object(verb_stem: str | None, name: Sequence[str], family_tokens: set[str]) -> bool:
    """True when the claim's verb stem is the object of the tool name: its last token, after
    another token of the family ("refunded" -> the ``refund`` of ``issue_refund``)."""
    return (
        verb_stem is not None and len(name) > 1 and name[-1] == verb_stem
        and any(t in family_tokens for t in name[:-1])
    )


def _score(
    family: str,
    object_words: list[str],
    spec: ToolSpec,
    lexicon: Mapping[str, Any],
    verb_stem: str | None = None,
) -> tuple[int, bool]:
    """``(score, object component earned)``; see ``score_tool``."""
    name = tokenize_tool_name(spec.name)
    fam_terms = _family_terms(family, lexicon)
    obj_terms = _object_terms(family, [_stem(w) for w in object_words], lexicon)
    fam_hits: set[int] = set().union(*(_occurs(t, name) for t in fam_terms))
    obj_hits: set[int] = set().union(*(_occurs(t, name) for t in obj_terms))
    object_matched = bool(obj_hits) and (not fam_hits or len(fam_hits | obj_hits) > 1)
    score = (2 if fam_hits else 0) + (2 if object_matched else 0)
    desc = _words(spec.description)
    if any(_occurs(t, desc) and not _occurs(t, name) for t in obj_terms):
        score += 1
    # Evidence only, never score: a sibling ``<verb>_<stem>`` tool must not outrank the
    # tool that ran ("refunded you" vs ``issue_refund`` and ``refund_payment``).
    family_tokens = {_stem(t) for t in lexicon["families"].get(family, {}).get("tokens", ())}
    return score, object_matched or _names_object(verb_stem, name, family_tokens)


def score_tool(
    family: str, object_words: list[str], spec: ToolSpec, lexicon: Mapping[str, Any]
) -> int:
    """Score a claim against one tool.

    +2 when a family token (synonym-expanded) is a tool-name token, +2 when an object
    word (synonym-expanded) is a tool-name token, +1 when the description holds an object
    word (synonym-expanded) that the tool name lacks. One tool-name token cannot earn both
    +2s ("sent the email" scores the ``email`` of ``email_report`` once).
    """
    return _score(family, object_words, spec, lexicon)[0]


def _match(
    family: str,
    obj: str | None,
    known_tools: Sequence[ToolSpec],
    lexicon: Mapping[str, Any],
    verb_stem: str | None = None,
) -> tuple[list[str], Confidence | None, bool]:
    """``match_tools`` plus whether the candidates earned the object component."""
    words = _words(obj) if obj else []
    verbs = _verb_tokens(family, lexicon)
    scored: list[tuple[str, int, bool]] = []
    for spec in known_tools:
        name = tokenize_tool_name(spec.name)
        if verbs & set(name) and not _conflicts(family, name, lexicon):
            scored.append((spec.name, *_score(family, words, spec, lexicon, verb_stem)))
    best = max((score for _, score, _ in scored), default=0)
    if best < 2:
        return [], None, False
    top = [(name, matched) for name, score, matched in scored if score == best]
    candidates = list(dict.fromkeys(name for name, _ in top))
    return candidates, "high" if best >= 3 else "medium", any(m for _, m in top)


def match_tools(
    family: str, obj: str | None, known_tools: Sequence[ToolSpec], lexicon: Mapping[str, Any]
) -> tuple[list[str], Confidence | None]:
    """All tools at the maximum score (ties kept, in ``known_tools`` order) when it is
    at least 2; ``high`` confidence from 3.

    Only tools whose name carries one of the family's verb tokens are eligible. An
    object, description or noun-token match alone ("deleted two tickets" vs
    ``create_ticket``, "emailed" vs ``email_customer``) never makes a candidate, nor
    does a tool whose name conflicts (``_conflicts``).
    """
    candidates, confidence, _ = _match(family, obj, known_tools, lexicon)
    return candidates, confidence


def _subject_phrase(subject: Sequence[str]) -> str | None:
    """The noun phrase ending the subject tokens, cut at the first preposition
    ("your refund" -> "refund", "the email to your manager" -> "email")."""
    head = list(subject)
    for i, tok in enumerate(head):
        if tok in _SUBJECT_PREPOSITIONS:
            head = head[:i]
            break
    phrase: list[str] = []
    for tok in reversed(head):
        if tok in _SUBJECT_BOUNDARY or len(phrase) == 4:
            break
        phrase.insert(0, tok)
    return " ".join(phrase) or None


def _verb_stem(form: str, family: str, lexicon: Mapping[str, Any]) -> str | None:
    """The family verb token a matched form is built on ("refunded" -> "refund")."""
    compact = re.sub(r"[^a-z]", "", form.lower())
    found = [v for v in _verb_tokens(family, lexicon) if compact.startswith(v)]
    return max(found, key=len) if found else None


def _surface_stem(form: str, family: str, lexicon: Mapping[str, Any]) -> str | None:
    """The noun token a matched form is built on ("emailed" -> "email", "texted" -> "text")."""
    compact = re.sub(r"[^a-z]", "", form.lower())
    stems = [n for n, fam in _noun_families(lexicon).items() if fam == family]
    found = [n for n in stems if compact.startswith(n) and n != _stem(family)]
    return max(found, key=len) if found else None


def _dedupe(claims: list[Claim]) -> list[Claim]:
    """One claim per candidate set; unmatched claims collapse on ``(verb, object)``.

    The first claim's span is kept, but it carries object evidence when any duplicate has
    it, so the order of sentences never decides whether a claim is judged.
    """
    seen: dict[tuple[str | None, ...], Claim] = {}
    out: list[Claim] = []
    for claim in claims:
        key: tuple[str | None, ...] = (
            ("tools", *sorted(claim.candidates)) if claim.candidates
            else ("verb", claim.verb, claim.object)
        )
        first = seen.get(key)
        if first is None:
            seen[key] = claim
            out.append(claim)
        elif claim._object_matched:
            first._object_matched = True
    return out


def _clean(clause: str) -> str:
    return clause.strip().rstrip(_TRAILING_PUNCT).strip()


class RuleExtractor:
    """Deterministic, stdlib-only claim extractor driven by the lexicon."""

    def __init__(self, lexicon: Mapping[str, Any] | None = None) -> None:
        self._lexicon: Mapping[str, Any] = _default_lexicon() if lexicon is None else lexicon

    def _sentence_filtered(self, sentence: str) -> bool:
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
                    other = _verbless_other_subject(clause, lex)
                    if other is not None:
                        prev_other = other
                    continue
                kind = _subject_kind(action, clause, lex)
                # A subject-less clause carries the previous clause's non-agent subject
                # ("The customer sent the form and has since cancelled the order").
                inherited = kind == "implicit" and prev_other
                if kind != "implicit":
                    prev_other = kind == "other"
                if inherited or kind in {"other", "state"} or _lexically_filtered(clause, lex):
                    continue
                obj, count = extract_object(clause, action.end, lex)
                match_text = obj
                if kind == "passive":
                    # "Your refund has been issued": the subject names the object.
                    subject = _subject_phrase(action.subject)
                    obj = obj or subject
                    match_text = " ".join(p for p in (subject, match_text) if p) or None
                # "I emailed you": the verb itself names the channel.
                stem = _surface_stem(clause[action.start:action.end], action.family, lex)
                match_text = " ".join(p for p in (match_text, stem) if p) or None
                form = clause[action.start:action.end]
                candidates, confidence, matched = _match(
                    action.family, match_text, known_tools, lex,
                    _verb_stem(form, action.family, lex),
                )
                claim = Claim(
                    text=_clean(clause), verb=action.family, object=obj, count=count,
                    candidates=candidates, confidence=confidence,
                )
                claim._object_matched = matched
                claims.append(claim)
        return _dedupe(claims)
