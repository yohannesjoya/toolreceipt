"""``LLMExtractor``: an optional claim extractor backed by any chat model.

It runs the rule extractor first, asks a model for the claims the rules cannot see (outcome
phrasing, synonyms, passive voice, status lines, other languages), checks every model claim
against precision guards built on the same lexicon, and merges the two lists with the rules
winning on overlap. Claims it adds carry ``source="llm"``; the diff reports them in
``Receipt.suspected`` unless ``llm_blocks`` is set. On any model error it falls back to the
rule claims alone, so a provider outage degrades to rules-only behaviour.

The model sees the reply text and the tool names and descriptions, never the ledger.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from toolreceipt.claims import (
    Claim,
    RuleExtractor,
    ToolSpec,
    _default_lexicon,
    _lexically_filtered,
    _normalize,
    _scan,
    _state_marked,
    _stem,
    _subject_kind,
    _verbless_other_subject,
    normalize_loose,
    split_clauses,
    split_sentences,
    tokenize_tool_name,
)

log = logging.getLogger("toolreceipt.llm")

PROMPT_VERSION = 1

# Filled with ``str.replace``: tools first, then the reply, so reply text is never re-scanned
# and braces in either need no escaping.
PROMPT_V1 = """You audit the final reply of an AI agent. List every ACTION CLAIM in it: a \
statement that the agent itself ("I", "we", or an agentless passive or outcome sentence about \
this conversation's request) has ALREADY performed, or is performing right now, an action \
that one of the tools listed below would perform.

Count as claims: direct statements ("I sent the email"), passive ones ("a refund has been \
issued"), outcomes or implied results ("your refund is on its way", "check your inbox for the \
confirmation"), status lines and checkmarks ("Email: sent"), vague completion that clearly \
covers specific requested actions ("all done" when the request named the actions), and claims \
written in any language.

Do NOT count: negated or failed actions ("couldn't send", "hasn't gone through"), future \
actions or offers ("I'll send", "I can open a ticket"), questions, advice or instructions to \
the user, conditionals, quoted or drafted text the agent did not perform, actions by other \
people or systems ("Sam merged it", "the carrier issued a refund"), events from before this \
conversation ("was refunded in March"), and read-only lookups ("I checked the order").

Tools:
{tools}

Return only JSON of the form {"claims": [{"text": "<exact substring of the reply>", "tool": \
"<tool name or null>"}]}. "text" must be copied verbatim from the reply and should be the \
shortest span that states the claim. Use null for "tool" when no listed tool fits. Return \
{"claims": []} when there are no claims.

Reply:
<<<
{reply}
>>>"""

_FENCE = re.compile(r"^```[a-zA-Z]*\s*$|^```\s*$", re.MULTILINE)
OnError = Literal["fallback", "raise"]


def build_prompt(text: str, known_tools: Sequence[ToolSpec]) -> str:
    """``PROMPT_V1`` with the tool list (``- name: description``, in order) and the reply."""
    tools = "\n".join(
        f"- {t.name}: {t.description}" if t.description else f"- {t.name}" for t in known_tools
    )
    return PROMPT_V1.replace("{tools}", tools or "(none)").replace("{reply}", text)


def parse_output(raw: str) -> list[tuple[str, str | None]]:
    """``(text, tool)`` pairs from the model text; the first JSON object found is used.

    Raises ``ValueError`` when no object is found or its ``claims`` do not have the shape
    ``[{"text": str, "tool": str | null}]``.
    """
    cleaned = _FENCE.sub("", raw)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("no JSON object in model output")
    try:
        data, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"unparsable JSON in model output: {exc.msg}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        raise ValueError('model output has no "claims" list')
    pairs: list[tuple[str, str | None]] = []
    for item in data["claims"]:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError('each claim needs a string "text"')
        tool = item.get("tool")
        if tool is not None and not isinstance(tool, str):
            raise ValueError('"tool" must be a string or null')
        pairs.append((item["text"], tool))
    return pairs


def message_text(message: Any) -> str:
    """The text of a chat model's reply, duck-typed over LangChain-style messages.

    Order: ``text`` (property or method), then ``content`` when it is a string, then the
    joined ``text`` of ``content`` blocks with ``type == "text"``; a plain string is itself.
    """
    if isinstance(message, str):
        return message
    text = getattr(message, "text", None)
    if isinstance(text, str):  # LangChain 1.x: a property (a callable str subclass, deprecated)
        return str(text)
    if callable(text):
        text = text()
    if isinstance(text, str):
        return text
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block["text"]
            for block in content
            if isinstance(block, Mapping) and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        return "".join(parts)
    raise TypeError(f"cannot read text from {type(message).__name__}")


class LLMExtractor:
    """Rules plus a chat model behind the ``ClaimExtractor`` protocol.

    ``complete`` receives the full prompt and returns the model's text; the caller sets the
    model, temperature, timeout and retries inside it. ``on_error="fallback"`` (default)
    returns the rule claims alone on any exception or unparsable output, after one warning
    on the ``toolreceipt.llm`` logger; ``"raise"`` re-raises. One ``lexicon`` drives both
    the rules and the guards. The extractor holds no per-run state and may be shared across
    ``Receipts`` and threads; ``last_raw`` and ``last_error`` are debugging aids only.
    """

    def __init__(
        self,
        complete: Callable[[str], str],
        *,
        on_error: OnError = "fallback",
        lexicon: Mapping[str, Any] | None = None,
    ) -> None:
        if on_error not in ("fallback", "raise"):
            raise ValueError('on_error must be "fallback" or "raise"')
        self._complete = complete
        self._on_error: OnError = on_error
        self._lexicon: Mapping[str, Any] = _default_lexicon() if lexicon is None else lexicon
        self._rules = RuleExtractor(self._lexicon)
        self.last_raw: str | None = None
        self.last_error: Exception | None = None

    @classmethod
    def from_chat_model(cls, model: Any, **kwargs: Any) -> LLMExtractor:
        """Wrap any object with ``invoke(prompt) -> message`` (a LangChain chat model)."""
        return cls(lambda prompt: message_text(model.invoke(prompt)), **kwargs)

    def extract(self, text: str, known_tools: Sequence[ToolSpec]) -> list[Claim]:
        rules = self._rules.extract(text, known_tools)
        self.last_raw = None
        self.last_error = None
        try:
            raw = self._complete(build_prompt(text, known_tools))
            self.last_raw = raw
            log.debug("model output: %s", raw)
            pairs = parse_output(raw)
        except Exception as exc:
            self.last_error = exc
            if self._on_error == "raise":
                raise
            log.warning("LLM extractor fell back to rules: %s", type(exc).__name__)
            return rules
        names = {t.name for t in known_tools}
        guarded = [_guard(span, tool, text, names, self._lexicon) for span, tool in pairs]
        llm_claims = [c for c in guarded if c is not None]
        loose_reply = normalize_loose(text)
        llm_claims.sort(key=lambda c: loose_reply.find(normalize_loose(c.text)))
        return _merge(rules, llm_claims)


# ---------------------------------------------------------------------------
# Guards (PRD 02 §7.3): applied to every model claim, in order; the first failing one wins.
# ---------------------------------------------------------------------------

MAX_SPAN_CHARS = 200


def _guard(
    span: str, tool: str | None, reply: str, names: set[str], lexicon: Mapping[str, Any]
) -> Claim | None:
    """The model claim as a ``Claim``, or ``None`` when a guard drops it."""
    span = span.strip()
    loose_span, loose_reply = normalize_loose(span), normalize_loose(reply)
    if not loose_span or loose_span not in loose_reply:
        return None  # 1. not a verbatim span of the reply
    if tool is not None and tool not in names:
        tool = None  # 2. unknown tool: the claim survives as unmatched
    if tool is not None and _read_tool(tool, lexicon):
        tool = None  # 3. read-only tool: unmatched, as the rules treat read verbs
    if _quoted_everywhere(loose_span, loose_reply):
        return None  # 4. quoted, drafted or code
    if _clause_filtered(_first_clause(loose_span, reply), lexicon):
        return None  # 5. negated, question, other subject or state description
    if len(loose_span) > MAX_SPAN_CHARS:
        return None  # 6. a paragraph, not a claim
    claim = Claim(
        text=span, verb="other", candidates=[tool] if tool else [], confidence="medium",
        source="llm",
    )
    claim._object_matched = True
    return claim


def _read_tool(name: str, lexicon: Mapping[str, Any]) -> bool:
    reads = {_stem(v) for v in lexicon.get("read_verbs", ())}
    return bool(reads & set(tokenize_tool_name(name)))


def _quoted_everywhere(span: str, reply: str) -> bool:
    """True when every occurrence of ``span`` in ``reply`` sits inside double quotes,
    backticks or a fenced code block (both strings already ``normalize_loose``d)."""
    start = reply.find(span)
    if start < 0:
        return False
    while start >= 0:
        before = reply[:start]
        fenced = before.count("```") % 2 == 1
        ticked = before.replace("```", "").count("`") % 2 == 1
        quoted = before.count('"') % 2 == 1
        if not (fenced or ticked or quoted):
            return False
        start = reply.find(span, start + 1)
    return True


def _first_clause(loose_span: str, reply: str) -> str:
    """The reply clause (split as the rules split) holding the span's first five words;
    the span itself when none does."""
    key = " ".join(loose_span.split()[:5])
    for sentence in split_sentences(_normalize(reply)):
        for clause in split_clauses(sentence):
            if key in normalize_loose(clause):
                return clause
    return loose_span


def _clause_filtered(clause: str, lexicon: Mapping[str, Any]) -> bool:
    """``is_filtered`` minus its future/advice half, so outcome claims survive: question
    mark, negation and other-subject lists; then an ``other`` or ``state`` subject, and the
    time-marker/by-agent test for implicit and verbless subjects (never for ``I``/``we``)."""
    if _lexically_filtered(clause, lexicon, future_advice=False):
        return True
    action = _scan(clause, lexicon)
    if action is None:
        other = _verbless_other_subject(clause, lexicon)
        return other is not False and _state_marked(clause, lexicon)
    kind = _subject_kind(action, clause, lexicon)
    if kind in {"other", "state"}:
        return True
    return kind == "implicit" and _state_marked(clause, lexicon)


# ---------------------------------------------------------------------------
# Merge (PRD 02 §7.4): rules first and winning on overlap; the model fills gaps and
# resolves unmatched rule claims (which never block, so replacing one is safe). A rule claim
# with candidates is never replaced or re-evidenced.
# ---------------------------------------------------------------------------


def _overlaps(rule: Claim, llm: Claim) -> bool:
    a, b = normalize_loose(rule.text), normalize_loose(llm.text)
    return a in b or b in a


def _merge(rules: list[Claim], llm: list[Claim]) -> list[Claim]:
    kept: list[Claim] = []
    replaced: list[int] = []  # unmatched rule claims the model resolved (row 2)
    for claim in llm:
        overlapping = [r for r in rules if _overlaps(r, claim)]
        if not overlapping:
            kept.append(claim)
        elif any(r.candidates and r._object_matched is not False for r in overlapping):
            continue  # the rule claim stands
        elif not claim.candidates:
            continue  # neither side resolved it; the rule claim is already unmatched
        elif unmatched := [r for r in overlapping if not r.candidates]:
            r = unmatched[0]  # the model resolved what the rules could not
            resolved = Claim(
                text=r.text, verb=r.verb, object=r.object, count=r.count,
                candidates=list(claim.candidates), confidence="medium", source="llm",
            )
            resolved._object_matched = True
            replaced.append(id(r))
            kept.append(resolved)
        else:
            kept.append(claim)  # family-alone rule claim: keep both, the diff decides
    rule_sets = {frozenset(r.candidates) for r in rules if r.candidates}
    seen: set[tuple[str, Any]] = set()
    out: list[Claim] = []
    for claim in kept:
        if claim.candidates and frozenset(claim.candidates) in rule_sets:
            continue  # the same tool is already claimed by the rules
        key: tuple[str, Any] = (
            ("tools", frozenset(claim.candidates)) if claim.candidates
            else ("text", normalize_loose(claim.text))
        )
        if key not in seen:
            seen.add(key)
            out.append(claim)
    return [r for r in rules if id(r) not in replaced] + out
