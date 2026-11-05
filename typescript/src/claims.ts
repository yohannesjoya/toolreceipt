/** Claim data model, extractor interface, tool specs and the rule extractor (port of claims.py). */

import { type Family, type Lexicon, defaultLexicon } from "./lexicon";
import {
  cpCompare, cpFirst, cpLast, cpLength, cpSlice, fullmatch, isAlnum, isMapping, matchAt0, pyInt, pyLower, pyLstrip, pyRegex,
  pyRstrip, pySplit, pySplitlines, pyStrip, pyTruthy, reEscape,
} from "./pystr";
import type { Json } from "./types";

export type Confidence = "high" | "medium";
export type Reason = "tool_failed" | "not_called" | "args_mismatch";
export type Source = "rules" | "llm";

export class ToolSpec {
  constructor(
    readonly name: string,
    readonly description: string = "",
  ) {}
}

/** Any tool object with a `name` and a `description`, such as a LangChain or AI SDK tool. */
export interface NamedTool {
  name: string;
  description?: string | null | undefined;
}

export type ToolLike =
  | string
  | ToolSpec
  | readonly [string]
  | readonly [string, string]
  | Record<string, unknown>
  | Map<string, unknown>
  | NamedTool
  | { readonly type: string }; // SDK tool interfaces, which have no index signature

export interface ClaimInit {
  text: string;
  verb: string;
  object?: string | null | undefined;
  count?: number | null | undefined;
  candidates?: readonly string[] | undefined;
  tool?: string | null | undefined;
  confidence?: Confidence | null | undefined;
  reason?: Reason | null | undefined;
  source?: Source | undefined;
  observed?: Record<string, Json> | null | undefined;
}

export interface ClaimEvidence {
  /** True when the candidates matched on an object word of their tool name, false when on the
   * verb family alone, null when unknown (a claim built by hand or by `fromDict`). */
  objectMatched: boolean | null;
  /** The reply clause the argument check reads literals from, when the extractor knows it. */
  clause: string | null;
  /** The reply sentence the clause came from (amounts are counted over it). */
  sentence: string | null;
}

export class Claim {
  text: string;
  verb: string;
  object: string | null;
  count: number | null;
  candidates: string[];
  tool: string | null;
  confidence: Confidence | null;
  reason: Reason | null;
  source: Source;
  /** The ledger arguments that contradict the claim when `reason === "args_mismatch"`. */
  observed: Record<string, Json> | null;

  // Extractor evidence: deliberately not enumerable, so toDict(), JSON.stringify and object
  // spread never emit it (the field set is fixed by the Python package).
  private declare _objectMatched: boolean | null;
  private declare _clause: string | null;
  private declare _sentence: string | null;

  constructor(init: ClaimInit) {
    this.text = init.text;
    this.verb = init.verb;
    this.object = init.object ?? null;
    this.count = init.count ?? null;
    this.candidates = [...(init.candidates ?? [])];
    this.tool = init.tool ?? null;
    this.confidence = init.confidence ?? null;
    this.reason = init.reason ?? null;
    this.source = init.source ?? "rules";
    this.observed = init.observed ?? null;
    for (const key of ["_objectMatched", "_clause", "_sentence"]) {
      Object.defineProperty(this, key, { value: null, writable: true, enumerable: false, configurable: true });
    }
  }

  toDict(): Record<string, Json> {
    return {
      text: this.text,
      verb: this.verb,
      object: this.object,
      count: this.count,
      candidates: [...this.candidates],
      tool: this.tool,
      confidence: this.confidence,
      reason: this.reason,
      source: this.source,
      observed: this.observed,
    };
  }

  static fromDict(data: Record<string, unknown>): Claim {
    const text = data["text"];
    const verb = data["verb"];
    if (typeof text !== "string") throw new TypeError("claim needs a string 'text'"); // Python: KeyError('text')
    if (typeof verb !== "string") throw new TypeError("claim needs a string 'verb'"); // Python: KeyError('verb')
    return new Claim({
      text,
      verb,
      object: (data["object"] as string | null | undefined) ?? null,
      count: (data["count"] as number | null | undefined) ?? null,
      candidates: (data["candidates"] as string[] | null | undefined) ?? [],
      tool: (data["tool"] as string | null | undefined) ?? null,
      confidence: (data["confidence"] as Confidence | null | undefined) ?? null,
      reason: (data["reason"] as Reason | null | undefined) ?? null,
      source: (data["source"] as Source | undefined) ?? "rules",
      observed: (data["observed"] as Record<string, Json> | null | undefined) ?? null,
    });
  }

  evidence(): ClaimEvidence {
    return { objectMatched: this._objectMatched, clause: this._clause, sentence: this._sentence };
  }

  setEvidence(e: Partial<ClaimEvidence>): void {
    if (e.objectMatched !== undefined) this._objectMatched = e.objectMatched;
    if (e.clause !== undefined) this._clause = e.clause;
    if (e.sentence !== undefined) this._sentence = e.sentence;
  }
}

/** `dataclasses.replace` that keeps the extractor's private match evidence (diff.py `_copy`). */
export function cloneClaim(claim: Claim, changes: Partial<ClaimInit> = {}): Claim {
  const out = new Claim({
    text: claim.text,
    verb: claim.verb,
    object: claim.object,
    count: claim.count,
    candidates: claim.candidates,
    tool: claim.tool,
    confidence: claim.confidence,
    reason: claim.reason,
    source: claim.source,
    observed: claim.observed,
    ...changes,
  });
  out.setEvidence(claim.evidence());
  return out;
}

export interface ClaimExtractor {
  extract(text: string, knownTools: readonly ToolSpec[]): Claim[];
  aextract?(text: string, knownTools: readonly ToolSpec[]): Promise<Claim[]>;
}

const TYPE_ERROR =
  "known_tools entries must be a name, a (name, description) pair, a ToolSpec, a dict or an object with a name attribute; got ";

export function typeName(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "object") {
    const ctor = (value as { constructor?: { name?: string } }).constructor;
    return ctor?.name ?? "object";
  }
  return typeof value;
}

/** Normalize names, pairs, ToolSpecs, (OpenAI-style) dicts and tool objects to `ToolSpec`. */
export function normalizeToolSpecs(tools: Iterable<ToolLike>): ToolSpec[] {
  const specs: ToolSpec[] = [];
  for (const tool of tools) {
    if (typeof tool === "string") {
      specs.push(new ToolSpec(tool, ""));
    } else if (tool instanceof ToolSpec) {
      specs.push(tool);
    } else if (Array.isArray(tool)) {
      const [name, ...rest] = tool as readonly [string, ...string[]];
      specs.push(new ToolSpec(name, rest[0] ?? ""));
    } else if (isMapping(tool)) {
      const get = (obj: unknown, key: string): unknown =>
        obj instanceof Map ? obj.get(key) : isMapping(obj) ? (obj as Record<string, unknown>)[key] : undefined;
      const fn = get(tool, "function"); // Python: tool.get("function") or tool.get("custom")
      const inner = pyTruthy(fn) ? fn : get(tool, "custom");
      const src = isMapping(inner) ? inner : tool;
      const name = get(src, "name");
      if (typeof name !== "string" || name === "") continue; // a built-in tool spec such as {"type": "web_search"}
      const description = get(src, "description");
      // Python: str(src.get("description") or ""); String() is the accepted counterpart (PRD 06 §7.3 item 11).
      specs.push(new ToolSpec(name, pyTruthy(description) ? String(description) : ""));
    } else {
      const toolName = (tool as { name?: unknown } | null | undefined)?.name;
      if (typeof toolName !== "string") throw new TypeError(TYPE_ERROR + typeName(tool));
      const description = (tool as unknown as { description?: unknown }).description;
      specs.push(new ToolSpec(toolName, pyTruthy(description) ? String(description) : ""));
    }
  }
  return specs;
}

// ---------------------------------------------------------------------------
// Rule extractor. Steps 1-5: sentence split, clause split, action detection, filters,
// object extraction (claims.py:146-451). Steps 6-7 (tool matching, dedupe) follow.
// ---------------------------------------------------------------------------

const BULLET = pyRegex("^\\s*(?:[-*•+]|\\d+[.)])\\s+");
const SENTENCE_END = pyRegex("(?<=[.!?])\\s+");
const CONJUNCTIONS = "(?:and|but|so|because|however|then)";
// Sentences are whitespace-collapsed before splitting, so single spaces suffice here. Also
// splits on " - ", en/em dashes and ": " (not "e-mail" or "3:00").
const CLAUSE_SPLIT = pyRegex(` ?; ?|,(?= |$) ?| ${CONJUNCTIONS}\\b ?| - | ?[\\u2013\\u2014] ?|:(?= |$) ?`, "i");
const WHITESPACE = pyRegex("\\s+", "g");
const LEADING_CONJ = pyRegex(`^(?:${CONJUNCTIONS}\\b[\\s,]*)+`, "i");
const TRAILING_PUNCT = ".!?,;:";
const TOKEN = pyRegex("'s|[a-z0-9#$@_-]+", "g");
const CONTRACTIONS: Array<[string, string]> = [["'ve", " have"], ["'re", " are"], ["'m", " am"], ["'d", " had"], ["'ll", " will"]];

const PERFECT = new Set(["have", "has", "had"]);
const BE_DONE = new Set(["been", "was", "were", "got"]);
const BE_PRESENT = new Set(["is", "are", "am", "'s", "be", "being", "get", "gets", "getting"]);
const HEADLINE_ADVERBS = new Set(["already", "just", "now", "successfully", "also"]);
const ADVERBS = new Set([...HEADLINE_ADVERBS, "all", "both", "since"]);
const AUX = new Set([...PERFECT, ...BE_DONE, ...BE_PRESENT]);
// A determiner or preposition right before a participle makes it an adjective
// ("your scheduled call", "with saved card"), not a completed action.
const ADJECTIVAL_BEFORE = new Set([
  "the", "a", "an", "your", "my", "our", "their", "his", "her", "its", "this", "that",
  "these", "those", "any", "no", "some", "every", "each", "with", "of", "for", "from",
]);
const FIRST_PERSON = new Set(["i", "we"]);
const OTHER_PRONOUNS = new Set(["you", "they", "he", "she", "it", "someone", "somebody", "everyone"]);
const SUBJECT_DETERMINERS = new Set([
  "the", "a", "an", "your", "their", "his", "her", "its", "this", "that", "these", "those",
  "my", "some", "someone", "somebody", "everyone",
]);
const OBJECT_STRIP = new Set(["you", "me", "the", "a", "an", "your", "our"]);

function has(set: Set<string>, value: string | undefined): boolean {
  return value !== undefined && set.has(value);
}

function intersects(a: Set<string>, b: Set<string>): boolean {
  for (const v of a) if (b.has(v)) return true;
  return false;
}

function lexList(lexicon: Lexicon, key: string): string[] {
  const value = lexicon[key];
  return Array.isArray(value) ? (value as string[]) : [];
}

/** Straighten curly quotes; keeps every character position. */
export function normalize(text: string): string {
  return text.replace(/’/g, "'").replace(/‘/g, "'").replace(/“/g, '"').replace(/”/g, '"');
}

const ODD_SPACES = /[    ]/g;

/**
 * Lower-case, straight-quoted, single-spaced text for containment tests. Unlike `normalize`
 * it does not keep character positions.
 */
export function normalizeLoose(text: string): string {
  return pyLower(pyStrip(normalize(text).replace(ODD_SPACES, " ").replace(WHITESPACE, " ")));
}

export function tokens(text: string): string[] {
  let low = pyLower(normalize(text));
  for (const [short, full] of CONTRACTIONS) low = low.replaceAll(short, full);
  return [...low.replaceAll("'s", " 's").matchAll(TOKEN)].map((m) => m[0]);
}

/**
 * Regex source for a lexicon phrase, word-bounded on whichever ends are alphanumeric.
 * Contraction suffixes such as `n't` match inside a word ("haven't").
 */
export function phrase(entry: string): string {
  const e = pyLower(pyStrip(entry));
  const suffix = e.startsWith("n'") || !isAlnum(cpFirst(e));
  const left = suffix ? "" : "(?<![a-z0-9_'])";
  const right = isAlnum(cpLast(e)) ? "(?![a-z0-9_])" : "";
  return left + pySplit(e).map(reEscape).join("\\s+") + right;
}

const PHRASE_CACHE = new Map<string, RegExp>();

/** `pyRegex(phrase(entry), flags)`, cached: `scan` compiles every lexicon form per clause. */
function phraseRegex(entry: string, flags = ""): RegExp {
  const key = `${flags}\u0000${entry}`;
  let re = PHRASE_CACHE.get(key);
  if (!re) {
    re = pyRegex(phrase(entry), flags);
    PHRASE_CACHE.set(key, re);
  }
  return re;
}

export function containsAny(text: string, entries: readonly string[]): boolean {
  const low = pyLower(normalize(text));
  return entries.some((e) => pyStrip(e) !== "" && phraseRegex(e).test(low));
}

/** Split a reply into sentences on `.!?` and newlines, stripping list bullets. */
export function splitSentences(text: string): string[] {
  const out: string[] = [];
  for (const raw of pySplitlines(text)) {
    const line = pyStrip(raw.replace(BULLET, ""));
    for (const s of line.split(SENTENCE_END)) if (pyStrip(s) !== "") out.push(pyStrip(s));
  }
  return out;
}

/** Split a sentence on `,` `;` and the conjunctions and/but/so/because/however/then. */
export function splitClauses(sentence: string): string[] {
  const out: string[] = [];
  for (const raw of pyStrip(sentence.replace(WHITESPACE, " ")).split(CLAUSE_SPLIT)) {
    const piece = pyStrip(pyStrip(raw).replace(LEADING_CONJ, ""));
    if (pyStrip(piece, TRAILING_PUNCT + " ") !== "") out.push(piece);
  }
  return out;
}

export interface Action {
  family: string;
  start: number;
  end: number;
  /** tokens before the verb, auxiliaries removed */
  subject: string[];
  /** auxiliary/adverb chain directly before the verb */
  aux: string[];
}

function context(clause: string, start: number): [string[], string[]] {
  const before = tokens(clause.slice(0, start));
  const aux: string[] = [];
  while (before.length > 0 && aux.length < 4 && (has(AUX, before.at(-1)) || has(ADVERBS, before.at(-1)))) {
    aux.unshift(before.pop()!);
  }
  return [before, aux];
}

function completed(subject: string[], aux: string[]): boolean {
  const auxSet = new Set(aux);
  if (subject.length > 0 && has(ADJECTIVAL_BEFORE, subject.at(-1)) && !intersects(auxSet, AUX)) {
    return false; // adjectival: "your scheduled call", "the already sent email"
  }
  const present = intersects(auxSet, BE_PRESENT);
  return !present || auxSet.has("been") || auxSet.has("now");
}

type Entry = [form: string, family: string | null];

/** The `(form, family)` list, longest form first. Rebuilt on every call, as Python does, so a
 * lexicon edited after the extractor was built is seen (compiled phrases are cached by text). */
function scanEntries(lexicon: Lexicon): Entry[] {
  const entries: Entry[] = [];
  for (const [name, fam] of Object.entries(lexicon.families)) for (const form of fam.forms) entries.push([form, name]);
  for (const form of lexList(lexicon, "exclusions")) entries.push([form, null]);
  entries.sort((a, b) => cpLength(b[0]) - cpLength(a[0])); // stable, as Python's sort; code points
  return entries;
}

function countQuotes(low: string, end: number): number {
  let n = 0;
  for (let i = 0; i < end; i++) if (low[i] === '"') n++;
  return n;
}

export function scan(clause: string, lexicon: Lexicon): Action | null {
  const low = pyLower(normalize(clause));
  const hits: Array<[number, number, string | null]> = [];
  for (const [form, family] of scanEntries(lexicon)) {
    for (const m of low.matchAll(phraseRegex(form, "g"))) {
      const start = m.index;
      const end = start + m[0].length;
      if (!hits.some(([s, e]) => start < e && s < end)) hits.push([start, end, family]);
    }
  }
  hits.sort((a, b) => a[0] - b[0] || a[1] - b[1] || cpCompare(a[2] ?? "", b[2] ?? ""));
  for (const [start, end, family] of hits) {
    if (countQuotes(low, start) % 2 === 1) continue; // inside a quotation
    if (family === null) return null; // a non-effectful verb comes first
    const [subject, aux] = context(clause, start);
    if (completed(subject, aux)) return { family, start, end, subject, aux };
  }
  return null;
}

/** `(family, start, end)` of the first completed action verb in `clause`. */
export function findAction(clause: string, lexicon: Lexicon): [string, number, number] | null {
  const action = scan(clause, lexicon);
  return action === null ? null : [action.family, action.start, action.end];
}

function byOtherAgent(clause: string, lexicon: Lexicon): boolean {
  const exempt = new Set(lexList(lexicon, "by_exempt"));
  const toks = tokens(clause);
  for (let i = 0; i < toks.length - 1; i++) {
    if (toks[i] !== "by") continue;
    const rest = toks.slice(i + 1).filter((t) => t !== "the" && t !== "a" && t !== "an");
    if (rest.length > 0 && !exempt.has(rest[0]!)) return true;
  }
  return false;
}

/** `I`/`we` right before the verb chain, or `our <noun>` ("our team") unless the noun is
 * customer-like ("our customer"). */
function isAgentSubject(subject: string[], lexicon: Lexicon): boolean {
  if (has(FIRST_PERSON, subject.at(-1))) return true;
  const window = subject.slice(Math.max(0, subject.length - 3), subject.length - 1);
  return window.includes("our") && !lexList(lexicon, "non_agent_nouns").includes(subject.at(-1)!);
}

export type SubjectKind = "agent" | "implicit" | "other" | "state" | "passive";

/** Classify who did the action: agent, implicit, other or state. */
export function subjectKind(action: Action, clause: string, lexicon: Lexicon): SubjectKind {
  const subject = action.subject;
  const aux = new Set(action.aux);
  const passive = intersects(aux, BE_DONE) || intersects(aux, BE_PRESENT);
  if (subject.length === 0) return "implicit";
  if (isAgentSubject(subject, lexicon)) return "agent";
  const last = subject.at(-1)!;
  if (!passive) {
    // Active voice with any other subject is someone else's action ("John refunded",
    // "the customer has cancelled"), except a headline whose participle ends the clause.
    const headline =
      [...aux].every((a) => HEADLINE_ADVERBS.has(a)) && !OTHER_PRONOUNS.has(last) && tokens(clause.slice(action.end)).length === 0;
    if (!headline) return "other";
  } else if (last === "you" && !aux.has("been")) {
    return "state"; // "you were charged"
  }
  // Passive or headline with no first-person agent.
  if (stateMarked(clause, lexicon)) return "state";
  return "passive";
}

const TIME_MARKER_CACHE = new Map<string, RegExp>();

/** A time marker ("on the 1st", "yesterday") or a non-agent `by` phrase: the clause describes
 * a state of affairs, not the agent's own action. */
export function stateMarked(clause: string, lexicon: Lexicon): boolean {
  const low = pyLower(normalize(clause));
  const timed = lexList(lexicon, "time_markers").some((p) => {
    let re = TIME_MARKER_CACHE.get(p);
    if (!re) {
      re = pyRegex(p);
      TIME_MARKER_CACHE.set(p, re);
    }
    return re.test(low);
  });
  return timed || byOtherAgent(clause, lexicon);
}

/** For a clause with no lexicon verb: does it open with a non-agent subject? True for "The
 * warehouse has shipped", false for "I looked", null when unknown. */
export function verblessOtherSubject(clause: string, lexicon: Lexicon): boolean | null {
  const toks = tokens(clause);
  if (toks.length === 0) return null;
  const first = toks[0]!;
  if (first === "our") return toks.length > 1 && lexList(lexicon, "non_agent_nouns").includes(toks[1]!);
  if (FIRST_PERSON.has(first)) return false;
  if (SUBJECT_DETERMINERS.has(first) || OTHER_PRONOUNS.has(first)) return true;
  if (toks.slice(1, 3).some((t) => PERFECT.has(t) || BE_DONE.has(t))) return true; // "Stripe has shipped"
  return null;
}

/** The sentence opens with a `conditional_lead` phrase ("If ...", "Once ..."). */
export function conditionalLead(sentence: string, lexicon: Lexicon): boolean {
  const low = pyLstrip(pyLower(normalize(sentence)), " \"'(*_");
  return lexList(lexicon, "conditional_lead").some((lead) => matchAt0(phraseRegex(lead), low) !== null);
}

/** `prevOther` after a clause with no lexicon verb: updated when the clause names a subject. */
export function carryOther(prevOther: boolean, clause: string, lexicon: Lexicon): boolean {
  const other = verblessOtherSubject(clause, lexicon);
  return other === null ? prevOther : other;
}

/** The last clause is subject-less and the clauses before it leave a non-agent subject in force. */
export function inheritsOtherSubject(clauses: readonly string[], lexicon: Lexicon): boolean {
  let prevOther = false;
  for (const clause of clauses.slice(0, -1)) {
    const action = scan(clause, lexicon);
    if (action === null) {
      prevOther = carryOther(prevOther, clause, lexicon);
      continue;
    }
    const kind = subjectKind(action, clause, lexicon);
    if (kind !== "implicit") prevOther = kind === "other";
  }
  if (!prevOther || clauses.length === 0) return false;
  const last = clauses.at(-1)!;
  const action = scan(last, lexicon);
  return action !== null && subjectKind(action, last, lexicon) === "implicit";
}

/** Question mark, `negation` and `other_subject` phrases; with `futureAdvice` (the rules'
 * setting) also the `future` and `advice` lists. */
export function lexicallyFiltered(clause: string, lexicon: Lexicon, futureAdvice = true): boolean {
  if (pyStrip(clause).endsWith("?")) return true;
  let low = pyLower(normalize(clause));
  if (["negation", "other_subject"].some((k) => containsAny(low, lexList(lexicon, k)))) return true;
  if (!futureAdvice) return false;
  // "I can confirm ..." / "Please note ..." are not future or advice.
  for (const p of lexList(lexicon, "filter_exempt")) low = low.replace(phraseRegex(p, "g"), " ");
  return ["future", "advice"].some((k) => containsAny(low, lexList(lexicon, k)));
}

/** True when the clause is negated, future/conditional, a question, advice, or the action
 * belongs to someone other than the agent (other subject or a state description). */
export function isFiltered(clause: string, lexicon: Lexicon): boolean {
  if (lexicallyFiltered(clause, lexicon)) return true;
  const action = scan(clause, lexicon);
  if (action === null) return false;
  const kind = subjectKind(action, clause, lexicon);
  return kind === "other" || kind === "state";
}

const DIGITS = pyRegex("\\d+");

/** `(object phrase, count)` from the text after the verb to the clause end. */
export function extractObject(clause: string, verbEnd: number, lexicon: Lexicon): [string | null, number | null] {
  const words = pySplit(pyStrip(clause.slice(verbEnd), TRAILING_PUNCT + " -"));
  while (words.length > 0 && OBJECT_STRIP.has(pyLower(words[0]!))) words.shift();
  if (words.length === 0) return [null, null];
  const head = pyLower(words[0]!);
  let count: number | null = null;
  const numberWords = (lexicon["number_words"] ?? {}) as Record<string, number>;
  if (fullmatch(DIGITS, head)) count = pyInt(head);
  else if (head !== "a" && head !== "an" && Object.hasOwn(numberWords, head)) count = Math.trunc(numberWords[head]!);
  return [words.join(" "), count];
}

export function clean(clause: string): string {
  return pyStrip(pyRstrip(pyStrip(clause), TRAILING_PUNCT));
}

// ---------------------------------------------------------------------------
// Tool matching (step 6; see docs/decisions.md). claims.py:453-950.
// ---------------------------------------------------------------------------

// ASCII classes in Python too: a non-ASCII character is a separator, never a token.
const NAME_PART = /[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+[0-9]*/g;
const NAME_SPLIT = /[^A-Za-z0-9]+/;
const WORD = /[a-z0-9]+/g;
// Words that never identify a tool (object filler, pronouns, prepositions).
const MATCH_STOP = new Set([
  "a", "an", "the", "to", "for", "of", "in", "on", "at", "by", "with", "from", "into",
  "and", "or", "it", "its", "this", "that", "these", "those", "you", "your", "me", "my",
  "i", "we", "us", "our", "they", "them", "their", "he", "she", "him", "her", "is", "are",
  "was", "were", "be", "been", "has", "have", "had", "all", "any", "some", "up", "out",
  "now", "again", "just", "already", "also", "as", "so", "here", "there",
]);
const SUBJECT_BOUNDARY = new Set([...SUBJECT_DETERMINERS, ...OBJECT_STRIP, ...FIRST_PERSON, "that", "which", "who"]);
const SUBJECT_PREPOSITIONS = new Set(["to", "for", "of", "from", "with"]);

/** A term is a tuple of stemmed words; `termKey` gives it set semantics. */
type Term = readonly string[];
const termKey = (t: Term): string => t.join("\u0000");

class TermSet {
  private readonly map = new Map<string, Term>();
  add(t: Term): void {
    this.map.set(termKey(t), t);
  }
  addAll(ts: Iterable<Term>): void {
    for (const t of ts) this.add(t);
  }
  values(): Term[] {
    return [...this.map.values()];
  }
}

function famOf(lexicon: Lexicon, family: string): Family {
  return lexicon.families[family] ?? { forms: [], tokens: [] };
}

/** Lowercase and strip one plural `s` ("tickets" -> "ticket"; keeps "address", "sms"). */
export function stem(word: string): string {
  const w = pyLower(word);
  return cpLength(w) > 3 && w.endsWith("s") && !w.endsWith("ss") ? cpSlice(w, 0, -1) : w;
}

export function words(text: string): string[] {
  return [...pyLower(normalize(text)).matchAll(WORD)].map((m) => stem(m[0]));
}

/** `stem` plus third-person handling: "searches" -> "search", "queries" -> "query", "closes" -> "close". */
export function stemVerb(word: string): string {
  const w = pyLower(word);
  if (w.length > 4 && w.endsWith("ies")) return w.slice(0, -3) + "y";
  if (w.length > 4 && w.endsWith("es") && /(?:sh|ch|x|z|ss|o)$/.test(w.slice(0, -2))) return w.slice(0, -2);
  return stem(w);
}

const LEAD_PREFIX = pyRegex("^\\s*[A-Za-z0-9_-]+:\\s*");
const LEAD_NOUN_FOLLOWERS = new Set([
  "the", "a", "an", "your", "our", "my", "his", "her", "their", "its", "this", "that",
  "these", "those", "me", "you", "us", "them", "it", "all", "each", "every", "any", "one",
]);
const CLAUSE_END = /[.;]| and /;

/** Text up to the first `.`, `;` or ` and `. */
function descriptionClause(description: string): string {
  const text = normalize(description);
  const m = CLAUSE_END.exec(text);
  return m === null ? text : text.slice(0, m.index);
}

/** Lowercase words of the first clause after an optional single-word `<Name>:` prefix and
 * one skip phrase (longest first). */
function leadWords(description: string, skip: readonly string[]): string[] {
  let text = pyStrip(pyLower(descriptionClause(description).replace(LEAD_PREFIX, "")));
  for (const p of [...skip].sort((a, b) => b.length - a.length)) {
    if (text === p || text.startsWith(p + " ")) {
      text = pyStrip(text.slice(p.length));
      break;
    }
  }
  return [...text.matchAll(WORD)].map((m) => m[0]);
}

/**
 * The family named by a description's lead verb (PRD 03 §7.2 rule 2), or `null`. The lead
 * verb is the first content word of the first clause, after an optional `<Name>:` prefix and
 * one `description_lead_skip` phrase; a read verb anywhere in the first clause gives `null`.
 */
export function descriptionFamily(description: string, lexicon: Lexicon = defaultLexicon()): string | null {
  const ws = leadWords(description, lexList(lexicon, "description_lead_skip"));
  if (ws.length === 0) return null;
  const reads = new Set(lexList(lexicon, "read_verbs").map(stem));
  if (ws.some((w) => reads.has(stemVerb(w)))) return null;
  const two = ws.slice(0, 2).map(stemVerb).join(" ");
  for (const [name, fam] of Object.entries(lexicon.families)) {
    if (fam.forms.some((f) => f.includes(" ") && pySplit(f).map(stemVerb).join(" ") === two)) return name;
  }
  const s = stemVerb(ws[0]!);
  for (const [name, fam] of Object.entries(lexicon.families)) {
    const single = new Set(fam.forms.filter((f) => !f.includes(" ")).map(stem));
    if (verbTokens(name, lexicon).has(s) || single.has(s)) return name;
  }
  // A noun token is an imperative verb only when an object follows it ("Email the customer").
  const follower = ws[1];
  const leadNouns = new Set(lexList(lexicon, "description_lead_nouns").map(stem));
  if (leadNouns.has(s) && follower !== undefined && LEAD_NOUN_FOLLOWERS.has(follower)) {
    return nounFamilies(lexicon).get(s) ?? null;
  }
  return null;
}

// Words that end a description's direct object: prepositions and clause openers.
const OBJECT_END = new Set([
  "to", "for", "of", "in", "on", "at", "by", "with", "from", "into", "about", "through",
  "via", "as", "using", "when", "if", "that", "which", "who", "where", "so", "or",
]);

/** Stemmed words of the lead verb's direct object (PRD 03 §7.2 rule 3), or `[]`. */
export function descriptionObject(description: string, lexicon: Lexicon = defaultLexicon()): string[] {
  if (descriptionFamily(description, lexicon) === null) return [];
  const ws = leadWords(description, lexList(lexicon, "description_lead_skip"));
  const two = ws.slice(0, 2).map(stemVerb).join(" ");
  const twoWordForms = new Set(
    Object.values(lexicon.families).flatMap((fam) =>
      fam.forms.filter((f) => f.includes(" ")).map((f) => pySplit(f).map(stemVerb).join(" ")),
    ),
  );
  const obj: string[] = [];
  for (const w of twoWordForms.has(two) ? ws.slice(2) : ws.slice(1)) {
    if (OBJECT_END.has(w)) break;
    if (!LEAD_NOUN_FOLLOWERS.has(w)) obj.push(stem(w));
  }
  return obj;
}

/** Split a tool name on `_`, `-`, other separators and camelCase; lowercase, singular. */
export function tokenizeToolName(name: string): string[] {
  return name.split(NAME_SPLIT).flatMap((chunk) => [...chunk.matchAll(NAME_PART)].map((m) => stem(m[0])));
}

/** Indices of `words` covered by contiguous occurrences of `term`. */
function occurs(term: Term, ws: readonly string[]): Set<number> {
  const n = term.length;
  const hits = new Set<number>();
  for (let i = 0; i + n <= ws.length; i++) {
    let same = true;
    for (let j = 0; j < n; j++) if (ws[i + j] !== term[j]) { same = false; break; }
    if (same) for (let j = 0; j < n; j++) hits.add(i + j);
  }
  return hits;
}

export function synonymGroups(lexicon: Lexicon): TermSet[] {
  const synonyms = (lexicon["synonyms"] ?? {}) as Record<string, string[]>;
  return Object.entries(synonyms).map(([key, values]) => {
    const group = new TermSet();
    for (const w of [key, ...values]) group.add(words(w));
    return group;
  });
}

/** Noun-like family tokens ("email", "issue", "record") mapped to their family. */
export function nounFamilies(lexicon: Lexicon): Map<string, string> {
  const nouns = new Set(lexList(lexicon, "noun_tokens"));
  const out = new Map<string, string>();
  for (const [name, fam] of Object.entries(lexicon.families)) {
    for (const tok of fam.tokens) if (nouns.has(tok)) out.set(stem(tok), name);
  }
  return out;
}

/** A family's verb tokens: its name and every token not listed in `noun_tokens`. */
export function verbTokens(family: string, lexicon: Lexicon): Set<string> {
  const nouns = new Set(lexList(lexicon, "noun_tokens"));
  const out = new Set([stem(family)]);
  for (const t of famOf(lexicon, family).tokens) if (!nouns.has(t)) out.add(stem(t));
  return out;
}

/** True when any of `tools` carries a verb token of `family` in its name. */
export function familyRan(family: string, tools: Iterable<string>, lexicon: Lexicon = defaultLexicon()): boolean {
  if (!Object.hasOwn(lexicon.families, family)) return false;
  const verbs = verbTokens(family, lexicon);
  for (const t of tools) if (tokenizeToolName(t).some((tok) => verbs.has(tok))) return true;
  return false;
}

/** True when a tool name carries a verb token of any family or a read verb. */
function isVerbNamed(nameTokens: readonly string[], lexicon: Lexicon): boolean {
  const toks = new Set(nameTokens);
  const reads = new Set(lexList(lexicon, "read_verbs").map(stem));
  if (intersects(toks, reads)) return true;
  return Object.keys(lexicon.families).some((f) => intersects(toks, verbTokens(f, lexicon)));
}

/** `(tokens the tool is matched on, byDescription)` (PRD 03 §7.2 rule 1). */
export function eligibility(spec: ToolSpec, lexicon: Lexicon): [Set<string>, boolean] {
  const name = tokenizeToolName(spec.name);
  if (isVerbNamed(name, lexicon)) return [new Set(name), false];
  const family = descriptionFamily(spec.description, lexicon);
  return [family ? verbTokens(family, lexicon) : new Set<string>(), true];
}

/** The tokens a tool is matched on; see `eligibility`. */
export function toolVerbTokens(spec: ToolSpec, lexicon: Lexicon = defaultLexicon()): Set<string> {
  return eligibility(spec, lexicon)[0];
}

/** True when the tool's tokens say it does something else than the claim: a read verb, or a
 * verb of another family without one of the claim's own. */
function conflicts(family: string, name: Iterable<string>, lexicon: Lexicon): boolean {
  const toks = new Set(name);
  const reads = new Set(lexList(lexicon, "read_verbs").map(stem));
  if (intersects(toks, reads)) return true;
  if (intersects(toks, verbTokens(family, lexicon))) return false;
  return Object.keys(lexicon.families).some((other) => other !== family && intersects(toks, verbTokens(other, lexicon)));
}

/** Family tokens; verb tokens are also expanded through every synonym group holding them. */
function familyTerms(family: string, lexicon: Lexicon): Term[] {
  const verbs = verbTokens(family, lexicon);
  const terms = new TermSet();
  for (const v of verbs) terms.add([v]);
  for (const t of famOf(lexicon, family).tokens) terms.add(words(t));
  for (const group of synonymGroups(lexicon)) {
    if (group.values().some((term) => term.length === 1 && verbs.has(term[0]!))) terms.addAll(group.values());
  }
  return terms.values().filter((t) => t.length > 0);
}

/** Content words of the object plus synonyms of any group they (or a phrase) hit. Expansion
 * never yields another family's noun token. */
function objectTerms(family: string, ws: readonly string[], lexicon: Lexicon): Term[] {
  const foreign = new Set([...nounFamilies(lexicon)].filter(([, fam]) => fam !== family).map(([n]) => n));
  const content = ws.filter((w) => !MATCH_STOP.has(w));
  const terms = new TermSet();
  for (const w of content) terms.add([w]);
  for (const group of synonymGroups(lexicon)) {
    if (group.values().some((member) => member.length > 0 && occurs(member, content).size > 0)) {
      for (const m of group.values()) if (m.length > 0 && !foreign.has(m.join(" "))) terms.add(m);
    }
  }
  return terms.values();
}

/** True when the claim's verb stem is the object of the tool name: its last token, after
 * another token of the family ("refunded" -> the `refund` of `issue_refund`). */
function namesObject(verbStemWord: string | null, name: readonly string[], familyTokens: Set<string>): boolean {
  return (
    verbStemWord !== null && name.length > 1 && name.at(-1) === verbStemWord && name.slice(0, -1).some((t) => familyTokens.has(t))
  );
}

function union(sets: Iterable<Set<number>>): Set<number> {
  const out = new Set<number>();
  for (const s of sets) for (const v of s) out.add(v);
  return out;
}

/** `(score, object component earned)`; see `scoreTool`. */
function score(
  family: string,
  objectWords: readonly string[],
  spec: ToolSpec,
  lexicon: Lexicon,
  verbStemWord: string | null = null,
  byDescription = false,
): [number, boolean] {
  const name = tokenizeToolName(spec.name);
  const famTerms = familyTerms(family, lexicon);
  const objTerms = objectTerms(family, objectWords.map(stem), lexicon);
  const famHits = union(famTerms.map((t) => occurs(t, name)));
  const objHits = union(objTerms.map((t) => occurs(t, name)));
  let objectMatched = objHits.size > 0 && (famHits.size === 0 || union([famHits, objHits]).size > 1);
  if (byDescription) {
    // PRD 03 §7.2 rule 3: the lead verb earns the family component; an object word in the
    // name or in the description's direct object earns the object component, once.
    const directObject = descriptionObject(spec.description, lexicon);
    const descHit = objTerms.some((t) => occurs(t, directObject).size > 0 && occurs(t, name).size === 0);
    objectMatched = objHits.size > 0 || descHit;
    return [2 + (objectMatched ? 2 : 0), objectMatched];
  }
  const desc = words(spec.description);
  const descHit = objTerms.some((t) => occurs(t, desc).size > 0 && occurs(t, name).size === 0);
  let total = (famHits.size > 0 ? 2 : 0) + (objectMatched ? 2 : 0);
  if (descHit) total += 1;
  // Evidence only, never score: a sibling `<verb>_<stem>` tool must not outrank the tool that ran.
  const familyTokens = new Set(famOf(lexicon, family).tokens.map(stem));
  return [total, objectMatched || namesObject(verbStemWord, name, familyTokens)];
}

/** Score a claim against one tool: +2 family token in the name, +2 object word in the name,
 * +1 object word in the description that the name lacks. */
export function scoreTool(family: string, objectWords: readonly string[], spec: ToolSpec, lexicon: Lexicon): number {
  return score(family, objectWords, spec, lexicon)[0];
}

/** `matchTools` plus whether the candidates earned the object component. */
export function matchCandidates(
  family: string,
  obj: string | null,
  knownTools: readonly ToolSpec[],
  lexicon: Lexicon,
  verbStemWord: string | null = null,
): [string[], Confidence | null, boolean] {
  const ws = obj ? words(obj) : [];
  const verbs = verbTokens(family, lexicon);
  const scored: Array<[string, number, boolean]> = [];
  const described: ToolSpec[] = [];
  for (const spec of knownTools) {
    const [toks, byDescription] = eligibility(spec, lexicon);
    if (!intersects(verbs, toks) || conflicts(family, toks, lexicon)) continue;
    if (byDescription) described.push(spec);
    else scored.push([spec.name, ...score(family, ws, spec, lexicon, verbStemWord)]);
  }
  // PRD 03 §7.2 rule 4: a description-eligible tool joins the scoring only when no verb-named
  // tool scored, or when the claim's object names it ("a ticket in Jira").
  const namedBest = Math.max(0, ...scored.map(([, s]) => s));
  const wordSet = new Set(ws);
  for (const spec of described) {
    if (namedBest < 2 || tokenizeToolName(spec.name).some((t) => wordSet.has(t))) {
      let [s, matched] = score(family, ws, spec, lexicon, verbStemWord, true);
      if (namedBest >= 2) {
        // It may join a name match as a tie, never displace it.
        [s, matched] = [Math.min(s, namedBest), matched && s <= namedBest];
      }
      scored.push([spec.name, s, matched]);
    }
  }
  const best = Math.max(0, ...scored.map(([, s]) => s));
  if (best < 2) return [[], null, false];
  const top = scored.filter(([, s]) => s === best);
  const candidates = [...new Set(top.map(([name]) => name))];
  return [candidates, best >= 3 ? "high" : "medium", top.some(([, , m]) => m)];
}

/** All tools at the maximum score (ties kept, in `knownTools` order) when it is at least 2;
 * `high` confidence from 3. */
export function matchTools(
  family: string,
  obj: string | null,
  knownTools: readonly ToolSpec[],
  lexicon: Lexicon,
): [string[], Confidence | null] {
  const [candidates, confidence] = matchCandidates(family, obj, knownTools, lexicon);
  return [candidates, confidence];
}

/** The noun phrase ending the subject tokens, cut at the first preposition. */
function subjectPhrase(subject: readonly string[]): string | null {
  let head = [...subject];
  for (let i = 0; i < head.length; i++) {
    if (SUBJECT_PREPOSITIONS.has(head[i]!)) {
      head = head.slice(0, i);
      break;
    }
  }
  const out: string[] = [];
  for (const tok of [...head].reverse()) {
    if (SUBJECT_BOUNDARY.has(tok) || out.length === 4) break;
    out.unshift(tok);
  }
  return out.length > 0 ? out.join(" ") : null;
}

function longest(found: string[]): string | null {
  if (found.length === 0) return null;
  return found.reduce((a, b) => (b.length > a.length ? b : a));
}

/** The family verb token a matched form is built on ("refunded" -> "refund"). */
function verbStem(form: string, family: string, lexicon: Lexicon): string | null {
  const compact = pyLower(form).replace(/[^a-z]/g, "");
  return longest([...verbTokens(family, lexicon)].filter((v) => compact.startsWith(v)));
}

/** The noun token a matched form is built on ("emailed" -> "email", "texted" -> "text"). */
function surfaceStem(form: string, family: string, lexicon: Lexicon): string | null {
  const compact = pyLower(form).replace(/[^a-z]/g, "");
  const stems = [...nounFamilies(lexicon)].filter(([, fam]) => fam === family).map(([n]) => n);
  return longest(stems.filter((n) => compact.startsWith(n) && n !== stem(family)));
}

/** One claim per candidate set; unmatched claims collapse on `(verb, object)`. The first
 * claim's span is kept, but it carries object evidence when any duplicate has it. */
export function dedupe(claims: readonly Claim[]): Claim[] {
  const seen = new Map<string, Claim>();
  const out: Claim[] = [];
  for (const claim of claims) {
    const key =
      claim.candidates.length > 0
        ? ["tools", ...[...claim.candidates].sort(cpCompare)].join("\u0000")
        : ["verb", claim.verb, claim.object ?? "\u0001null"].join("\u0000");
    const first = seen.get(key);
    if (first === undefined) {
      seen.set(key, claim);
      out.push(claim);
    } else if (claim.evidence().objectMatched) {
      first.setEvidence({ objectMatched: true });
    }
  }
  return out;
}

/** Deterministic, dependency-free claim extractor driven by the lexicon. */
export class RuleExtractor implements ClaimExtractor {
  private readonly lexicon: Lexicon;

  constructor(lexicon?: Lexicon) {
    this.lexicon = lexicon ?? defaultLexicon();
  }

  private sentenceFiltered(sentence: string): boolean {
    return conditionalLead(sentence, this.lexicon);
  }

  extract(text: string, knownTools: readonly ToolSpec[]): Claim[] {
    const lex = this.lexicon;
    const claims: Claim[] = [];
    for (const sentence of splitSentences(normalize(text))) {
      if (this.sentenceFiltered(sentence)) continue;
      let prevOther = false;
      for (const clause of splitClauses(sentence)) {
        const action = scan(clause, lex);
        if (action === null) {
          prevOther = carryOther(prevOther, clause, lex);
          continue;
        }
        const kind = subjectKind(action, clause, lex);
        // A subject-less clause carries the previous clause's non-agent subject.
        const inherited = kind === "implicit" && prevOther;
        if (kind !== "implicit") prevOther = kind === "other";
        if (inherited || kind === "other" || kind === "state" || lexicallyFiltered(clause, lex)) continue;
        const extracted = extractObject(clause, action.end, lex);
        let obj = extracted[0];
        const count = extracted[1];
        let matchText = obj;
        if (kind === "passive") {
          // "Your refund has been issued": the subject names the object.
          const subject = subjectPhrase(action.subject);
          obj = obj ?? subject;
          matchText = [subject, matchText].filter((p): p is string => !!p).join(" ") || null;
        }
        // "I emailed you": the verb itself names the channel.
        const form = clause.slice(action.start, action.end);
        const surface = surfaceStem(form, action.family, lex);
        matchText = [matchText, surface].filter((p): p is string => !!p).join(" ") || null;
        const [candidates, confidence, matched] = matchCandidates(
          action.family, matchText, knownTools, lex, verbStem(form, action.family, lex),
        );
        const claim = new Claim({ text: clean(clause), verb: action.family, object: obj, count, candidates, confidence });
        claim.setEvidence({ objectMatched: matched, sentence });
        claims.push(claim);
      }
    }
    return dedupe(claims);
  }
}
