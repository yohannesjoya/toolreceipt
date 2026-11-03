/** Claim data model, extractor interface, tool specs and the rule extractor (port of claims.py). */

import type { Lexicon } from "./lexicon";
import {
  cpCompare, cpFirst, cpLast, fullmatch, isAlnum, matchAt0, pyInt, pyLower, pyLstrip, pyRegex, pyRstrip, pySplit,
  pySplitlines, pyStrip, reEscape,
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
  | NamedTool;

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
    return new Claim({
      text: data["text"] as string,
      verb: data["verb"] as string,
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

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object") return false;
  const proto = Object.getPrototypeOf(value) as unknown;
  return proto === Object.prototype || proto === null;
}

function typeName(value: unknown): string {
  if (value === null) return "NoneType";
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
    } else if (tool instanceof Map || isPlainObject(tool)) {
      const get = (obj: unknown, key: string): unknown =>
        obj instanceof Map ? obj.get(key) : isPlainObject(obj) ? obj[key] : undefined;
      const inner = get(tool, "function") ?? get(tool, "custom");
      const src = inner instanceof Map || isPlainObject(inner) ? inner : tool;
      const name = get(src, "name");
      if (typeof name !== "string" || name === "") continue; // a built-in tool spec such as {"type": "web_search"}
      const description = get(src, "description");
      // Python: str(src.get("description") or ""); String() is the accepted counterpart (PRD 06 §7.3 item 11).
      // eslint-disable-next-line @typescript-eslint/no-base-to-string
      specs.push(new ToolSpec(name, description ? String(description) : ""));
    } else {
      const toolName = (tool as { name?: unknown }).name;
      if (typeof toolName !== "string") throw new TypeError(TYPE_ERROR + typeName(tool));
      const description = (tool as { description?: unknown }).description;
      // eslint-disable-next-line @typescript-eslint/no-base-to-string
      specs.push(new ToolSpec(toolName, description ? String(description) : ""));
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
const ENTRY_CACHE = new WeakMap<Lexicon, Entry[]>();

function scanEntries(lexicon: Lexicon): Entry[] {
  let entries = ENTRY_CACHE.get(lexicon);
  if (!entries) {
    entries = [];
    for (const [name, fam] of Object.entries(lexicon.families)) for (const form of fam.forms) entries.push([form, name]);
    for (const form of lexList(lexicon, "exclusions")) entries.push([form, null]);
    entries.sort((a, b) => b[0].length - a[0].length); // stable, as Python's sort
    ENTRY_CACHE.set(lexicon, entries);
  }
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
// STUBS (Task 5): replaced by the rule extractor port in Tasks 6 and 7.
// ---------------------------------------------------------------------------

function notPorted(): never {
  throw new Error("not ported yet");
}

export class RuleExtractor implements ClaimExtractor {
  constructor(_lexicon?: unknown) {}
  extract(_text: string, _knownTools: readonly ToolSpec[]): Claim[] {
    return notPorted();
  }
}
export function descriptionFamily(_description: string, _lexicon?: unknown): string | null {
  return notPorted();
}
export function descriptionObject(_description: string, _lexicon?: unknown): string[] {
  return notPorted();
}
export function stemVerb(_word: string): string {
  return notPorted();
}
export function toolVerbTokens(_spec: ToolSpec, _lexicon?: unknown): Set<string> {
  return notPorted();
}
