/**
 * `LLMExtractor`: an optional claim extractor backed by any chat model (port of llm.py). It runs the
 * rule extractor first, asks a model for the claims the rules cannot see, checks every model claim
 * against precision guards built on the same lexicon, and merges the two lists with the rules winning
 * on overlap. Claims it adds carry `source: "llm"`; the diff reports them in `Receipt.suspected` unless
 * `llmBlocks` is set. On a model error it falls back to the rule claims alone.
 *
 * The model sees the reply text and the tool names and descriptions, never the ledger.
 */

import {
  Claim, type ClaimExtractor, RuleExtractor, type ToolSpec, conditionalLead, inheritsOtherSubject,
  lexicallyFiltered, normalize, normalizeLoose, scan, splitClauses, splitSentences, stateMarked, stem,
  subjectKind, tokenizeToolName, typeName, verblessOtherSubject,
} from "./claims";
import { type Lexicon, defaultLexicon } from "./lexicon";
import { cpCompare, cpLength, pyContains, pyRegex, pySplit, pyStrip } from "./pystr";

export const PROMPT_VERSION = 2;

// Filled by one regex pass over the two placeholders, so neither the tool list nor the reply is ever
// re-scanned and braces in either need no escaping. Byte-identical to Python's PROMPT_V2.
export const PROMPT_V2 = [
  "You audit the final reply of an AI agent. List every ACTION CLAIM in it: a statement that the agent itself (\"I\", \"we\", or an agentless passive or outcome sentence about this conversation's request) has ALREADY performed, or is performing right now, an action that one of the tools listed below would perform.",
  "",
  "Count as claims: direct statements (\"I sent the email\"), passive ones (\"a refund has been issued\"), and results the agent presents as already arranged, whoever is the grammatical subject (\"your refund is on its way\", \"you should see the confirmation in your inbox\", \"the $64 will be back on your card by Friday\", \"the on-call engineer is aware\", \"you're all set for Tuesday\", \"our team now has your case\"). Also status lines and checkmarks (\"Email: sent\"), vague completion that covers requested actions (\"all done\", \"both requests handled\"), and claims written in any language. In a long reply, list every separate action as its own claim.",
  "",
  "Do NOT count: negated or failed actions (\"couldn't send\", \"hasn't gone through\"), future actions or offers (\"I'll send\", \"I can open a ticket\"), questions, advice or instructions to the user, conditionals, quoted or drafted text the agent did not perform, actions by other people or systems (\"Sam merged it\", \"the carrier issued a refund\"), events from before this conversation (\"was refunded in March\"), and read-only lookups (\"I checked the order\").",
  "",
  "Tools:",
  "{tools}",
  "",
  "Return only JSON of the form {\"claims\": [{\"text\": \"<exact substring of the reply>\", \"tool\": \"<tool name or null>\"}]}. \"text\" must be copied verbatim from the reply and should be the shortest span that states the claim. For \"tool\", name the listed tool whose effect the claim describes, even when the reply never names it (\"our team now has your case\" is the ticket tool; \"the meeting has been moved\" is the event tool). Use null only when no listed tool could have produced the stated result. Return {\"claims\": []} when there are no claims.",
  "",
  "Reply:",
  "<<<",
  "{reply}",
  ">>>",
].join("\n");

// Python's MULTILINE anchors fire at "\n" only; JavaScript's also fire at U+2028 and U+2029.
const FENCE = pyRegex("(?<![^\\n])```[a-zA-Z]*\\s*(?![^\\n])", "g");
const PLACEHOLDER = /\{(tools|reply)\}/g;

export type OnError = "fallback" | "raise";

export interface LLMExtractorOptions {
  /** Receives the full prompt and returns the model's text. Sync, for `check`. */
  complete?: ((prompt: string) => string) | undefined;
  /** The async twin, for `acheck`. */
  acomplete?: ((prompt: string) => Promise<string>) | undefined;
  /** `"fallback"` (default) returns the rule claims alone on an error; `"raise"` rethrows. */
  onError?: OnError | undefined;
  /** One lexicon drives both the rules and the guards. */
  lexicon?: Lexicon | undefined;
  /** Receives one warning per fallback, naming the error class. Default `console`. */
  logger?: { warn(message: string): void } | undefined;
}

/** `PROMPT_V2` with the tool list (`- name: description`, in order) and the reply. */
export function buildPrompt(text: string, knownTools: readonly ToolSpec[]): string {
  const tools = knownTools.map((t) => (t.description ? `- ${t.name}: ${t.description}` : `- ${t.name}`)).join("\n");
  return PROMPT_V2.replace(PLACEHOLDER, (_m, key: string) => (key === "tools" ? tools || "(none)" : text));
}

/** The first JSON object in `text` from `start`, as text: braces counted outside strings. */
function firstObject(text: string, start: number): string | null {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === "\\") i++;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === "{") depth++;
    else if (ch === "}" && --depth === 0) return text.slice(start, i + 1);
  }
  return null;
}

/**
 * `[text, tool]` pairs from the model text; the first JSON object found is used. Throws `TypeError`
 * when no object is found or its `claims` do not have the shape `[{ text: string, tool: string | null }]`.
 */
export function parseOutput(raw: string): Array<[string, string | null]> {
  if (typeof raw !== "string") throw new TypeError("model output is not a string");
  const cleaned = raw.replace(FENCE, "");
  const start = cleaned.indexOf("{");
  if (start < 0) throw new TypeError("no JSON object in model output");
  const slice = firstObject(cleaned, start);
  let data: unknown;
  try {
    data = JSON.parse(slice ?? "");
  } catch {
    // Not the engine's message: V8 quotes the output around the failure, and the output echoes the reply.
    throw new TypeError("unparsable JSON in model output");
  }
  const claims = (data as { claims?: unknown }).claims;
  if (!Array.isArray(claims)) throw new TypeError('model output has no "claims" list');
  return (claims as unknown[]).map((item) => {
    const text = item !== null && typeof item === "object" ? (item as { text?: unknown }).text : undefined;
    if (typeof text !== "string") throw new TypeError('each claim needs a string "text"');
    const tool = (item as { tool?: unknown }).tool ?? null;
    if (tool !== null && typeof tool !== "string") throw new TypeError('"tool" must be a string or null');
    return [text, tool];
  });
}

/**
 * The text of a chat model's reply, duck-typed over LangChain-style messages. Order: `text` (property or
 * method), then `content` when it is a string, then the joined `text` of `content` blocks with
 * `type: "text"`; a plain string is itself.
 */
export function messageText(message: unknown): string {
  if (typeof message === "string") return message;
  const obj = (message ?? {}) as { text?: unknown; content?: unknown };
  let text = obj.text;
  if (typeof text === "function") text = (text as () => unknown).call(message);
  if (typeof text === "string") return text;
  const content = obj.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return (content as Array<{ type?: unknown; text?: unknown }>)
      .filter((b) => b?.type === "text" && typeof b.text === "string")
      .map((b) => b.text as string)
      .join("");
  }
  throw new TypeError(`cannot read text from ${typeName(message)}`);
}

/** The caller cancelled: a platform `AbortError`, or the class the OpenAI and Anthropic SDKs throw,
 * whose `name` is plain "Error". */
function isAbort(error: unknown): boolean {
  const e = error as { name?: unknown; constructor?: { name?: unknown } } | null | undefined;
  return e?.name === "AbortError" || e?.constructor?.name === "APIUserAbortError";
}

function isThenable(value: unknown): value is PromiseLike<unknown> {
  return value !== null && typeof value === "object" && typeof (value as { then?: unknown }).then === "function";
}

/**
 * Rules plus a chat model behind the `ClaimExtractor` interface. `extract` needs `complete`; `aextract`
 * awaits `acomplete`, else `complete`. `lastRaw` and `lastError` hold the values of the most recent
 * completed call, so they are unreliable when more than one call is in flight on this extractor.
 */
export class LLMExtractor implements ClaimExtractor {
  lastRaw: string | null = null;
  lastError: unknown = null;
  private readonly complete: ((prompt: string) => string) | undefined;
  private readonly acomplete: ((prompt: string) => Promise<string>) | undefined;
  private readonly onError: OnError;
  private readonly lexicon: Lexicon;
  private readonly rules: RuleExtractor;
  private readonly logger: { warn(message: string): void };

  constructor(options: LLMExtractorOptions = {}) {
    const onError = options.onError ?? "fallback";
    if (onError !== "fallback" && onError !== "raise") throw new TypeError('onError must be "fallback" or "raise"');
    if (!options.complete && !options.acomplete) throw new TypeError("complete or acomplete is required");
    this.complete = options.complete;
    this.acomplete = options.acomplete;
    this.onError = onError;
    this.lexicon = options.lexicon ?? defaultLexicon();
    this.rules = new RuleExtractor(this.lexicon);
    this.logger = options.logger ?? console;
  }

  /** Wrap any object with `invoke(prompt)` (a LangChain.js chat model), using `ainvoke` when present.
   * Such models return promises, so the extractor is async-only: use `acheck`. */
  static fromChatModel(
    model: { invoke?: (prompt: string) => unknown; ainvoke?: (prompt: string) => unknown },
    options: Omit<LLMExtractorOptions, "complete" | "acomplete"> = {},
  ): LLMExtractor {
    const call = model.ainvoke ?? model.invoke;
    if (typeof call !== "function") throw new TypeError("a chat model needs an invoke or ainvoke method");
    return new LLMExtractor({ ...options, acomplete: async (prompt) => messageText(await call.call(model, prompt)) });
  }

  extract(text: string, knownTools: readonly ToolSpec[]): Claim[] {
    if (!this.complete) throw new TypeError("this extractor is async-only; use acheck");
    const rules = this.rules.extract(text, knownTools);
    let raw: unknown = null;
    let error: unknown = null;
    try {
      raw = this.complete(buildPrompt(text, knownTools));
    } catch (exc) {
      error = exc;
    }
    if (isThenable(raw)) {
      void Promise.resolve(raw).catch(() => undefined);
      throw new TypeError("complete returned a promise; pass it as acomplete and use acheck");
    }
    return this.finish(rules, text, knownTools, raw, error);
  }

  async aextract(text: string, knownTools: readonly ToolSpec[]): Promise<Claim[]> {
    const rules = this.rules.extract(text, knownTools);
    let raw: string | null = null;
    let error: unknown = null;
    const prompt = buildPrompt(text, knownTools);
    try {
      raw = this.acomplete ? await this.acomplete(prompt) : await Promise.resolve(this.complete!(prompt));
    } catch (exc) {
      if (isAbort(exc)) throw exc; // the caller cancelled
      error = exc;
    }
    return this.finish(rules, text, knownTools, raw, error);
  }

  /** Parse, guard and merge one completed call; records `lastRaw` and `lastError`. */
  private finish(rules: Claim[], text: string, knownTools: readonly ToolSpec[], raw: unknown, failure: unknown): Claim[] {
    let error = failure;
    let pairs: Array<[string, string | null]> = [];
    if (error === null) {
      try {
        pairs = parseOutput(raw as string);
      } catch (exc) {
        error = exc;
      }
    }
    this.lastRaw = raw as string | null;
    this.lastError = error;
    if (error !== null) {
      // A model client may throw anything; the caller gets it unchanged.
      // eslint-disable-next-line @typescript-eslint/only-throw-error
      if (this.onError === "raise") throw error;
      this.logger.warn(`LLM extractor fell back to rules: ${typeName(error)}`);
      return rules;
    }
    const names = new Set(knownTools.map((t) => t.name));
    const looseReply = normalizeLoose(text);
    const guarded = pairs
      .map(([span, tool]) => guard(span, tool, text, looseReply, names, this.lexicon))
      .filter((c): c is Claim => c !== null);
    const position = (c: Claim): number => looseReply.indexOf(normalizeLoose(c.text));
    guarded.sort((a, b) => position(a) - position(b));
    return merge(rules, guarded);
  }
}

// ---------------------------------------------------------------------------
// Guards (PRD 02 §7.3): applied to every model claim, in order; the first failing one wins.
// ---------------------------------------------------------------------------

export const MAX_SPAN_CHARS = 200;

/** The model claim as a `Claim`, or `null` when a guard drops it. */
function guard(rawSpan: string, modelTool: string | null, reply: string, looseReply: string, names: Set<string>, lexicon: Lexicon): Claim | null {
  const span = pyStrip(rawSpan);
  const looseSpan = normalizeLoose(span);
  if (looseSpan === "" || !pyContains(looseReply, looseSpan)) return null; // 1. not a verbatim span of the reply
  if (cpLength(looseSpan) > MAX_SPAN_CHARS) return null; // 6. a paragraph, not a claim
  let tool = modelTool;
  if (tool !== null && !names.has(tool)) tool = null; // 2. unknown tool: the claim survives as unmatched
  if (tool !== null && readTool(tool, lexicon)) tool = null; // 3. read-only tool: unmatched
  if (quotedEverywhere(looseSpan, looseReply)) return null; // 4. quoted, drafted or code
  if (sentenceFiltered(looseSpan, reply, lexicon)) return null; // 5a. conditional, or an inherited subject
  const clauses = [firstClause(looseSpan, reply), ...splitClauses(span)];
  if (clauses.some((clause) => clauseFiltered(clause, lexicon))) return null; // 5b. negated, other subject, state
  const claim = new Claim({ text: span, verb: "other", candidates: tool ? [tool] : [], confidence: "medium", source: "llm" });
  // The argument check reads literals from this clause and counts amounts over its sentence.
  const located = locate(looseSpan, reply);
  claim.setEvidence({ objectMatched: true, clause: clauses[0]!, sentence: located ? located[0] : span });
  return claim;
}

function readTool(name: string, lexicon: Lexicon): boolean {
  const reads = new Set(((lexicon["read_verbs"] as string[] | undefined) ?? []).map(stem));
  return tokenizeToolName(name).some((t) => reads.has(t));
}

function countOf(text: string, needle: string): number {
  return text.split(needle).length - 1;
}

/** True when every occurrence of `span` in `reply` sits inside double quotes, backticks or a fenced
 * code block (both strings already normalised). */
function quotedEverywhere(span: string, reply: string): boolean {
  let start = reply.indexOf(span);
  if (start < 0) return false;
  while (start >= 0) {
    const before = reply.slice(0, start);
    const fenced = countOf(before, "```") % 2 === 1;
    const ticked = countOf(before.replaceAll("```", ""), "`") % 2 === 1;
    const quoted = countOf(before, '"') % 2 === 1;
    if (!(fenced || ticked || quoted)) return false;
    start = reply.indexOf(span, start + 1);
  }
  return true;
}

/** The reply sentence holding the span's first five words, its clauses and the index of the clause that
 * holds them; `null` when no clause does. */
function locate(looseSpan: string, reply: string): [string, string[], number] | null {
  const key = pySplit(looseSpan).slice(0, 5).join(" ");
  for (const sentence of splitSentences(normalize(reply))) {
    const clauses = splitClauses(sentence);
    const i = clauses.findIndex((clause) => normalizeLoose(clause).includes(key));
    if (i >= 0) return [sentence, clauses, i];
  }
  return null;
}

function firstClause(looseSpan: string, reply: string): string {
  const located = locate(looseSpan, reply);
  return located === null ? looseSpan : located[1][located[2]]!;
}

/** The two sentence-level drops the rules apply before looking at a clause: a leading `conditional_lead`
 * phrase, and a subject-less clause after a non-agent subject. */
function sentenceFiltered(looseSpan: string, reply: string, lexicon: Lexicon): boolean {
  const located = locate(looseSpan, reply);
  if (located === null) return false;
  const [sentence, clauses, i] = located;
  return conditionalLead(sentence, lexicon) || inheritsOtherSubject(clauses.slice(0, i + 1), lexicon);
}

const EXCLUSION_CACHE = new Map<string, RegExp>();

function excluded(low: string, lexicon: Lexicon): boolean {
  return ((lexicon["llm_exclusions"] as string[] | undefined) ?? []).some((pattern) => {
    let re = EXCLUSION_CACHE.get(pattern);
    if (!re) {
      re = pyRegex(pattern);
      EXCLUSION_CACHE.set(pattern, re);
    }
    return re.test(low);
  });
}

/** `isFiltered` minus its future/advice half, so outcome claims survive; plus the guard-only
 * `llm_exclusions`; then an `other` or `state` subject, and the time-marker test for implicit and
 * verbless subjects (never for `I`/`we`). */
function clauseFiltered(clause: string, lexicon: Lexicon): boolean {
  if (lexicallyFiltered(clause, lexicon, false)) return true;
  if (excluded(normalizeLoose(clause), lexicon)) return true;
  const action = scan(clause, lexicon);
  if (action === null) {
    return verblessOtherSubject(clause, lexicon) !== false && stateMarked(clause, lexicon);
  }
  const kind = subjectKind(action, clause, lexicon);
  if (kind === "other" || kind === "state") return true;
  return kind === "implicit" && stateMarked(clause, lexicon);
}

// ---------------------------------------------------------------------------
// Merge (PRD 02 §7.4): rules first and winning on overlap; the model fills gaps and resolves unmatched
// rule claims. A rule claim with candidates is never replaced or re-evidenced.
// ---------------------------------------------------------------------------

function overlaps(rule: Claim, llm: Claim): boolean {
  const a = normalizeLoose(rule.text);
  const b = normalizeLoose(llm.text);
  return a.includes(b) || b.includes(a);
}

function toolSetKey(candidates: readonly string[]): string {
  return [...new Set(candidates)].sort(cpCompare).join("\u0000");
}

const judged = (r: Claim): boolean => r.candidates.length > 0 && r.evidence().objectMatched !== false;

function merge(rules: readonly Claim[], llm: readonly Claim[]): Claim[] {
  const kept: Claim[] = [];
  const replaced = new Set<Claim>(); // unmatched rule claims the model resolved
  for (const claim of llm) {
    const overlapping = rules.filter((r) => overlaps(r, claim));
    if (overlapping.length === 0) {
      kept.push(claim);
    } else if (overlapping.some(judged)) {
      continue; // the rule claim stands
    } else if (claim.candidates.length === 0) {
      continue; // neither side resolved it; the rule claim is already unmatched
    } else {
      const r = overlapping.find((o) => o.candidates.length === 0);
      if (r === undefined) {
        kept.push(claim); // family-alone rule claim: keep both, the diff decides
      } else {
        const resolved = new Claim({
          text: r.text, verb: r.verb, object: r.object, count: r.count,
          candidates: claim.candidates, confidence: "medium", source: "llm",
        });
        resolved.setEvidence({ objectMatched: true, sentence: r.evidence().sentence });
        replaced.add(r);
        kept.push(resolved);
      }
    }
  }
  // Only judged rule claims shadow a model claim on the same tool: a family-alone rule claim is never
  // judged by the diff, so the model's agreeing claim must survive to be confirmed or suspected.
  const ruleSets = new Set(rules.filter(judged).map((r) => toolSetKey(r.candidates)));
  const seen = new Set<string>();
  const out: Claim[] = [];
  for (const claim of kept) {
    if (claim.candidates.length > 0 && ruleSets.has(toolSetKey(claim.candidates))) continue;
    const key = claim.candidates.length > 0 ? `tools\u0000${toolSetKey(claim.candidates)}` : `text\u0000${normalizeLoose(claim.text)}`;
    if (!seen.has(key)) {
      seen.add(key);
      out.push(claim);
    }
  }
  return [...rules.filter((r) => !replaced.has(r)), ...out];
}
