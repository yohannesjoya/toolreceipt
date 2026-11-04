/**
 * Argument check (PRD 04 §8): literals in a claim's clause against the ledger's arguments.
 * Port of _args.py. Private module used by `diff`. Precision first: a kind of literal is
 * judged only when the clause holds exactly one of it, and only against arguments whose key
 * names that kind.
 */

import { type Claim, normalizeLoose, stem, tokenizeToolName } from "./claims";
import { Dec } from "./decimal";
import type { LedgerEntry } from "./ledger";
import type { Lexicon } from "./lexicon";
import { cpFirst, cpLength, isAlpha, matchAt0, pyDumps, pyLower, pyRegex, pySplit, pyStrip } from "./pystr";
import type { Json } from "./types";

// Unicode hyphens (U+2010 to U+2015) and the minus sign map to "-" before any scan.
const HYPHENS = /[‐‑‒–—―−]/g;
const dehyphen = (s: string): string => s.replace(HYPHENS, "-");
// Words that may sit between a cue and its number ("ticket #5521", "case no. 2024-118").
export const SEPARATORS = new Set(["#", "no", "no.", "number", "id", ":"]);

const NUMBER = "\\d+(?:,\\d{3})*(?:\\.\\d+)?";
const SYMBOLS = "[$€£¥]";
const CODES = "(?:usd|eur|gbp|cad|aud|jpy)";
// "pounds" is a weight as often as a currency, so sterling needs £ or gbp.
const WORDS = "(?:dollars?|euros?)";
const AMOUNT = pyRegex(
  `${SYMBOLS}\\s?(${NUMBER})(?![\\d%])` + `|\\b${CODES}\\s?(${NUMBER})(?![\\d%])` + `|\\b(${NUMBER})\\s?(?:${CODES}|${WORDS})\\b`,
  "g",
);
// An amount is the action's amount only when the words around it say nothing else (PRD 04
// §8.2 rule 1; review B1 to B4).
const PREPOSITIONS = new Set(["of", "on", "from", "in", "at", "with", "per", "by", "over", "under", "about", "around"]);
const FILLERS = new Set([
  "your", "the", "a", "an", "my", "our", "this", "that", "their", "its", "his", "her",
  "original", "full", "entire", "first", "whole", "same", "total", "remaining", "initial",
]);
const APPROXIMATIONS = new Set(["about", "around", "roughly", "approximately", "nearly", "almost", "over", "under", "~", "circa"]);
const TWO_WORD_APPROXIMATIONS = new Set(["up to", "at least", "at most"]);
const ALLOWED_AFTER = new Set([
  "to", "for", "on", "onto", "back", "into", "via", "as", "in", "from", "has", "have",
  "had", "was", "is", "were", "been", "will", "which", "that", "so", "because", "since",
  "today", "yesterday", "now", "already", "directly", "straight", "immediately", "and",
]);
const RATE_AFTER_FOR = new Set(["each", "every"]);
const REFERENCE_FULL = pyRegex("^(?:\\d[\\d-]*\\d|\\d{3,})$");
const DATE = pyRegex("^\\d{4}-\\d{2}-\\d{2}$|^\\d{2}/\\d{2}/\\d{4}$");
const EDGE_PUNCT = ".,;!?()\"'";
const ANOTHER_FORMAT = pyRegex("[,.' ]\\d");
const ALL_DIGITS = pyRegex("^\\d+$");
const ONE_DIGIT = pyRegex("^\\d$");

/** The clause the literals come from: the extractor's recorded clause, else the text. */
function clauseText(claim: Claim): string {
  const clause = claim.evidence().clause ?? claim.text;
  return dehyphen(normalizeLoose(clause));
}

/** False when the words around `low[start:end]` say the amount is not the action's. */
function qualified(low: string, start: number, end: number): boolean {
  const tail = low.slice(end);
  const first = cpFirst(tail);
  if (first !== "" && (isAlpha(first) || first === "+" || first === "/")) return false; // "$1.5k", "$64/mo", "$64+"
  if (matchAt0(ANOTHER_FORMAT, tail)) return false; // "€1.250,00", "€1 250": another number format
  const before = pySplit(low.slice(0, start)).map((t) => pyStrip(t, EDGE_PUNCT));
  while (before.length > 0 && FILLERS.has(before.at(-1)!)) before.pop();
  if (before.length > 0) {
    const last = before.at(-1)!;
    if (PREPOSITIONS.has(last) || APPROXIMATIONS.has(last)) return false;
    if (before.length >= 2 && TWO_WORD_APPROXIMATIONS.has(`${before.at(-2)!} ${last}`)) return false;
  }
  const after = pySplit(tail).slice(0, 2).map((t) => pyStrip(t, EDGE_PUNCT)).filter((t) => t !== "");
  if (after.length === 0) return true;
  const a0 = after[0]!;
  // A past participle keeps it the action's amount too ("EUR 1,250.00 refunded").
  if (!ALLOWED_AFTER.has(a0) && !(cpLength(a0) > 4 && a0.endsWith("ed"))) return false;
  return !((a0 === "for" || a0 === "to") && after.length > 1 && RATE_AFTER_FOR.has(after[1]!));
}

function amountMatches(low: string): RegExpExecArray[] {
  return [...low.matchAll(AMOUNT)];
}

/**
 * The clause's one money amount (currency mark required), judged in its sentence. `null`
 * when the sentence holds any other amount, when the amount is not in the clause, or when the
 * words around it qualify it (§8.2 rule 1).
 */
export function findAmount(clause: string, sentence?: string | null): Dec | null {
  const lowClause = dehyphen(normalizeLoose(clause));
  let low = sentence ? dehyphen(normalizeLoose(sentence)) : lowClause;
  if (amountMatches(low).length !== 1) return null;
  const inClause = amountMatches(lowClause);
  if (inClause.length !== 1) return null;
  const m = inClause[0]!;
  let offset = low.indexOf(lowClause);
  if (offset < 0) {
    low = lowClause;
    offset = 0;
  }
  if (!qualified(low, offset + m.index, offset + m.index + m[0].length)) return null;
  const group = m.slice(1).find((g) => g !== undefined && g !== "") ?? "";
  return Dec.parse(group.replaceAll(",", ""))?.abs() ?? null;
}

function refTokens(low: string): string[] {
  return pySplit(low.replaceAll("#", " # ").replaceAll(":", " : "));
}

/**
 * The clause's one cued reference number as `[cue stem, digits]`, else `null`. The number
 * follows a cue word directly or through one separator, starts with a digit, holds three or
 * more digits and is not shaped like a date. Digits are returned with hyphens removed and
 * leading zeros stripped.
 */
export function findReference(clause: string, cues: readonly string[]): [string, string] | null {
  const cueStems = new Set(cues.map(stem));
  const toks = refTokens(dehyphen(normalizeLoose(clause)));
  const found: Array<[string, string]> = [];
  let skip = 0;
  for (let i = 0; i < toks.length; i++) {
    if (skip > 0) {
      skip--;
      continue;
    }
    const tok = pyStrip(toks[i]!, EDGE_PUNCT);
    let digits = tok.replaceAll("-", "");
    if (!(ONE_DIGIT.test(cpFirst(tok)) && REFERENCE_FULL.test(tok) && cpLength(digits) >= 3)) continue;
    if (matchAt0(DATE, tok) || !ALL_DIGITS.test(digits)) continue;
    // "tracking 9400 1118 9922 3334": digit groups of three or more are one number.
    for (const nxt of toks.slice(i + 1)) {
      const group = pyStrip(nxt, EDGE_PUNCT);
      if (cpLength(group) >= 3 && ALL_DIGITS.test(group)) {
        digits += group;
        skip++;
      } else break;
    }
    let j = i - 1;
    if (j >= 0 && (SEPARATORS.has(toks[j]!) || SEPARATORS.has(toks[j]!.replace(/\.+$/, "")))) j--;
    if (j < 0) continue;
    const cue = stem(pyStrip(toks[j]!, EDGE_PUNCT));
    if (cueStems.has(cue)) found.push([cue, digits.replace(/^0+/, "") || "0"]);
  }
  return found.length === 1 ? found[0]! : null;
}

// ---------------------------------------------------------------------------
// Keyed arguments and agreement (§8.2 rules 2 and 3)
// ---------------------------------------------------------------------------

/** `[amount, [cue, digits]]` found in a claim's clause; `null` where the clause has none. */
export type Literals = [Dec | null, [string, string] | null];

// Applied to a stripped value; single optional spaces keep the pattern free of backtracking.
const VALUE_AMOUNT_FULL = pyRegex(`^(?:-?\\s?(?:${SYMBOLS}|${CODES})?\\s?(${NUMBER})\\s?(?:${CODES})?)$`);
const MAX_VALUE_CHARS = 64;
const DIGIT_RUN = pyRegex("\\d[\\d-]*\\d|\\d", "g");
const DIGIT_SPACE_DIGIT = pyRegex("(?<=\\d) (?=\\d)", "g");
const DIGITS_RUN = pyRegex("\\d+", "g");

/** The claim clause's one amount and one cued reference, each `null` when absent. */
export function literals(claim: Claim, lexicon: Lexicon): Literals {
  const clause = clauseText(claim);
  const cues = (lexicon["id_cues"] as string[] | undefined) ?? [];
  return [findAmount(clause, claim.evidence().sentence), findReference(clause, cues)];
}

/** A keyed amount's value as a positive `Dec`; `null` when it is not a number. A JS number is a
 * Python int iff `Number.isInteger` (PRD 06 §7.3 item 8); both go through `Decimal(str(value))`. */
function amountValue(value: unknown): Dec | null {
  if (typeof value === "boolean") return null;
  if (typeof value === "number") return Dec.fromNumber(value)?.abs() ?? null;
  if (typeof value === "string" && cpLength(value) <= MAX_VALUE_CHARS) {
    const m = VALUE_AMOUNT_FULL.exec(pyLower(pyStrip(value)));
    if (m) return Dec.parse(m[1]!.replaceAll(",", ""))?.abs() ?? null;
  }
  return null;
}

/** Digit runs of three or more digits, leading zeros stripped: each plain run and each
 * hyphen-joined run ("ORD-2024-4821" gives 2024, 4821 and 20244821). */
function digitRuns(text: string): Set<string> {
  const t = dehyphen(text);
  const joined = t.replace(DIGIT_SPACE_DIGIT, ""); // "1234 5678" is also 12345678
  const runs = new Set<string>();
  for (const m of [...t.matchAll(DIGIT_RUN), ...joined.matchAll(DIGIT_RUN)]) runs.add(m[0].replaceAll("-", ""));
  for (const m of t.matchAll(DIGITS_RUN)) runs.add(m[0]);
  return new Set([...runs].filter((r) => cpLength(r) >= 3).map((r) => r.replace(/^0+/, "") || "0"));
}

/** A keyed reference's digit runs; `null` when the value holds no run of three digits. Only a
 * Python `int` (an integer-valued JS number) or a string is read. */
function referenceRuns(value: unknown): Set<string> | null {
  if (typeof value === "boolean") return null;
  let runs: Set<string>;
  if (typeof value === "number" && Number.isInteger(value)) runs = digitRuns(String(value));
  else if (typeof value === "string") runs = digitRuns(value);
  else return null;
  return runs.size > 0 ? runs : null;
}

function amountKeyed(key: string, amountKeys: Set<string>): boolean {
  const toks = tokenizeToolName(key);
  return amountKeys.has(stem(pyLower(key))) || (toks.length > 0 && amountKeys.has(toks.at(-1)!));
}

function referenceKeyed(key: string, cue: string): boolean {
  return tokenizeToolName(key).includes(cue);
}

function amountKeySet(lexicon: Lexicon): Set<string> {
  return new Set(((lexicon["amount_keys"] as string[] | undefined) ?? []).map(stem));
}

function keyedAmounts(entry: LedgerEntry, amountKeys: Set<string>): Map<string, Dec> {
  const keyed = new Map<string, Dec>();
  for (const [k, v] of Object.entries(entry.args)) {
    if (!amountKeyed(k, amountKeys)) continue;
    const val = amountValue(v);
    if (val !== null) keyed.set(k, val);
  }
  return keyed;
}

/**
 * Judge one `ok` entry against the claim's literals: `[true, {}]` when the entry agrees on
 * every literal it can judge and judges at least one; `[false, observed]` when it has a keyed
 * argument for every literal present and at least one disagrees; `[null, {}]` otherwise.
 */
export function entryVerdict(
  lits: Literals,
  entry: LedgerEntry,
  lexicon: Lexicon,
  options: { amountSum?: Dec | null | undefined; context?: readonly LedgerEntry[] | undefined } = {},
): [boolean | null, Record<string, Json>] {
  const [amount, reference] = lits;
  const amountSum = options.amountSum ?? null;
  const context = options.context ?? [];
  let present = 0;
  let judged = 0;
  let agreed = 0;
  const observed: Record<string, Json> = {};
  if (amount !== null) {
    present++;
    const keyed = keyedAmounts(entry, amountKeySet(lexicon));
    if (keyed.size > 0) {
      judged++;
      const hundred = amount.times100();
      const inTargets = (d: Dec): boolean => d.equals(amount) || d.equals(hundred);
      if ([...keyed.values()].some(inTargets) || (amountSum !== null && inTargets(amountSum))) agreed++;
      else for (const k of keyed.keys()) observed[k] = entry.args[k]!;
    }
  }
  if (reference !== null) {
    present++;
    const [cue, digits] = reference;
    const keyedRuns = new Map<string, Set<string>>();
    for (const [k, v] of Object.entries(entry.args)) {
      if (!referenceKeyed(k, cue)) continue;
      const runs = referenceRuns(v);
      if (runs !== null) keyedRuns.set(k, runs);
    }
    if (keyedRuns.size > 0) {
      judged++;
      if (
        [...keyedRuns.values()].some((runs) => runs.has(digits)) ||
        digitRuns(entry.result ?? "").has(digits) ||
        mapped(digits, keyedRuns, context)
      ) {
        agreed++;
      } else for (const k of keyedRuns.keys()) observed[k] = entry.args[k]!;
    }
  }
  if (judged === 0) return [null, {}];
  if (agreed === judged) return [true, {}];
  if (judged < present) return [null, {}];
  return [false, observed];
}

/** True when some entry's args or result hold both `digits` and a keyed value's digits. */
function mapped(digits: string, keyedRuns: Map<string, Set<string>>, context: readonly LedgerEntry[]): boolean {
  const wanted = new Set<string>();
  for (const runs of keyedRuns.values()) for (const r of runs) wanted.add(r);
  for (const other of context) {
    const runs = new Set([...digitRuns(pyDumps(other.args)), ...digitRuns(other.result ?? "")]);
    if (runs.has(digits) && [...runs].some((r) => wanted.has(r))) return true;
  }
  return false;
}

/**
 * The disagreeing arguments when every `ok` entry of the claim's candidates contradicts the
 * claim and no same-family tool's entry agrees; `null` confirms the claim (§8.2 rule 4).
 */
export function argsVerdict(
  claim: Claim,
  hitEntries: readonly LedgerEntry[],
  familyEntries: readonly LedgerEntry[],
  lexicon: Lexicon,
  options: { context?: readonly LedgerEntry[] | undefined } = {},
): Record<string, Json> | null {
  const context = options.context ?? [];
  const lits = literals(claim, lexicon);
  if ((lits[0] === null && lits[1] === null) || hitEntries.length === 0) return null;
  const amountKeys = amountKeySet(lexicon);
  const sums: Dec[] = [];
  for (const e of hitEntries) {
    const first = keyedAmounts(e, amountKeys).values().next();
    if (!first.done) sums.push(first.value);
  }
  const amountSum = sums.length >= 2 ? sums.reduce((a, b) => a.add(b)) : null;
  const verdicts = hitEntries.map((e) => entryVerdict(lits, e, lexicon, { amountSum, context }));
  if (verdicts.some(([v]) => v !== false)) return null;
  if (familyEntries.some((e) => entryVerdict(lits, e, lexicon, { context })[0] === true)) return null;
  return verdicts[0]![1];
}
