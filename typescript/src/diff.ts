/** Set-semantics diff between claims and the ledger (port of diff.py). */

import { argsVerdict } from "./args";
import { type Claim, type Reason, cloneClaim, familyRan, tokenizeToolName, verbTokens } from "./claims";
import type { Ledger, LedgerEntry } from "./ledger";
import { defaultLexicon } from "./lexicon";
import { Receipt, type Verdict, renderSummary } from "./receipt";
import type { Json } from "./types";

export interface DiffOptions {
  knownTools?: readonly string[] | undefined;
  ignoreTools?: Iterable<string> | undefined;
  llmBlocks?: boolean | undefined;
  checkArgs?: boolean | undefined;
}

let ALL_VERB_TOKENS: Set<string> | null = null;

/** Every verb token of every family in the packaged lexicon. The diff has no extractor
 * lexicon in hand, so sibling demotion always uses the packaged families. */
function allVerbTokens(): Set<string> {
  if (ALL_VERB_TOKENS === null) {
    const lex = defaultLexicon();
    const all = new Set<string>();
    for (const family of Object.keys(lex.families)) for (const t of verbTokens(family, lex)) all.add(t);
    ALL_VERB_TOKENS = all;
  }
  return ALL_VERB_TOKENS;
}

/** True when a tool that ran ok shares a packaged-lexicon verb token with `candidate`'s name
 * (`send_email` ran, the claim names `send_sms`). */
function siblingRan(candidate: string, okTools: Iterable<string>): boolean {
  const all = allVerbTokens();
  const verbs = new Set(tokenizeToolName(candidate).filter((t) => all.has(t)));
  if (verbs.size === 0) return false;
  for (const t of okTools) if (tokenizeToolName(t).some((tok) => verbs.has(tok))) return true;
  return false;
}

/** The arguments contradicting a confirmed claim, or `null` (PRD 04 §8.1). A family-only match
 * says nothing about which tool did the work, so its arguments say nothing either. */
function mismatch(claim: Claim, okEntries: readonly LedgerEntry[]): Record<string, Json> | null {
  if (claim.evidence().objectMatched === false) return null;
  const candidates = new Set(claim.candidates);
  const hitEntries = okEntries.filter((e) => candidates.has(e.tool));
  const familyEntries = okEntries.filter((e) => !candidates.has(e.tool) && familyRan(claim.verb, [e.tool]));
  return argsVerdict(claim, hitEntries, familyEntries, defaultLexicon(), { context: okEntries });
}

/**
 * Compare `claims` with what `ledger` recorded. Matching is set membership: entries are never
 * consumed. Input claims are not mutated. A claim with `source === "llm"` whose tool never ran
 * lands in `suspected` unless `llmBlocks` is set. With `checkArgs` (default), a claim whose tool
 * ran is still `args_mismatch` when the one amount or cued reference number in its clause
 * contradicts every matching call's arguments.
 */
export function diff(claims: readonly Claim[], ledger: Ledger, options: DiffOptions = {}): Receipt {
  const knownTools = options.knownTools ?? [];
  const ignored = new Set(options.ignoreTools ?? []);
  const llmBlocks = options.llmBlocks ?? false;
  const checkArgs = options.checkArgs ?? true;

  const okTools = ledger.okTools();
  const okEntries = ledger.entries.filter((e) => e.status === "ok");
  const errorTools = ledger.errorTools();

  const confirmed: Claim[] = [];
  const neverRan: Claim[] = [];
  const unmatched: Claim[] = [];
  const suspected: Claim[] = [];
  for (const claim of claims) {
    if (claim.candidates.length === 0) {
      unmatched.push(cloneClaim(claim, { candidates: [] }));
      continue;
    }
    if (claim.source === "llm" && claim.candidates.every((c) => ignored.has(c))) {
      unmatched.push(cloneClaim(claim, { candidates: [] }));
      continue;
    }
    const hit = claim.candidates.find((c) => okTools.has(c));
    if (hit !== undefined) {
      const observed = checkArgs ? mismatch(claim, okEntries) : null;
      if (observed === null) {
        confirmed.push(cloneClaim(claim, { candidates: [...claim.candidates], tool: hit }));
        continue;
      }
      const judged = cloneClaim(claim, { candidates: [...claim.candidates], tool: hit, reason: "args_mismatch", observed });
      if (claim.source === "llm" && !llmBlocks) suspected.push(judged);
      else neverRan.push(judged);
      continue;
    }
    const failed = claim.candidates.some((c) => errorTools.has(c));
    const first = claim.candidates[0]!;
    if (claim.source === "llm") {
      if (!failed && llmBlocks && siblingRan(first, okTools)) {
        unmatched.push(cloneClaim(claim, { candidates: [...claim.candidates] }));
        continue;
      }
      const llmReason: Reason = failed ? "tool_failed" : "not_called";
      const judged = cloneClaim(claim, { candidates: [...claim.candidates], tool: first, reason: llmReason });
      (llmBlocks ? neverRan : suspected).push(judged);
    } else if (failed) {
      neverRan.push(cloneClaim(claim, { candidates: [...claim.candidates], tool: first, reason: "tool_failed" }));
    } else if (claim.evidence().objectMatched === false || familyRan(claim.verb, okTools)) {
      // Not judged: matched on the verb family alone, or another tool of the same action ran.
      unmatched.push(cloneClaim(claim, { candidates: [...claim.candidates] }));
    } else {
      neverRan.push(cloneClaim(claim, { candidates: [...claim.candidates], tool: first, reason: "not_called" }));
    }
  }

  const confirmedTools = new Set(confirmed.map((c) => c.tool));
  const unreported = [...ledger].filter((e) => e.status === "ok" && !confirmedTools.has(e.tool) && !ignored.has(e.tool));

  let verdict: Verdict;
  if (neverRan.length > 0) verdict = "fabricated";
  else if (unreported.length > 0) verdict = "unreported";
  else verdict = "clean";

  return new Receipt({
    confirmed,
    claimedButNeverRan: neverRan,
    unmatched,
    ranButUnreported: unreported,
    verdict,
    summary: renderSummary(verdict, neverRan, unreported, suspected, confirmed.length),
    ledger,
    knownTools: [...new Set(knownTools)],
    suspected,
  });
}
