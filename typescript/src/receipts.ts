/** `check()` and `acheck()`: the framework-free entry points (port of receipts.py). */

import { type Claim, type ClaimExtractor, RuleExtractor, type ToolLike, ToolSpec, normalizeToolSpecs } from "./claims";
import { type DiffOptions, diff } from "./diff";
import type { Ledger } from "./ledger";
import { cpCompare } from "./pystr";
import type { Receipt } from "./receipt";

export interface CheckOptions {
  /** Extra tools to match claims against: names, `[name, description]` pairs, tool dicts or
   * objects with a `name`. Merged with the tools the ledger saw. */
  knownTools?: Iterable<ToolLike> | undefined;
  /** Tools never reported as `ranButUnreported` (read-only lookups). */
  ignoreTools?: Iterable<string> | undefined;
  /** Any `ClaimExtractor`; default `new RuleExtractor()`. */
  extractor?: ClaimExtractor | null | undefined;
  /** Let claims found only by an LLM extractor make the verdict `fabricated`; by default they
   * are reported in `receipt.suspected`. */
  llmBlocks?: boolean | undefined;
  /** Compare the one amount or cued reference number in a confirmed claim's clause with the
   * call's arguments (`args_mismatch`); `false` skips that check. */
  checkArgs?: boolean | undefined;
}

/** Union by name in first-seen order; a non-empty description is never overridden. */
function mergeTools(...groups: Iterable<ToolSpec>[]): ToolSpec[] {
  const merged = new Map<string, string>();
  for (const group of groups) {
    for (const { name, description } of group) {
      if (!merged.get(name)) merged.set(name, description);
    }
  }
  return [...merged].map(([name, description]) => new ToolSpec(name, description));
}

/** Known tools plus every tool the ledger saw, by name. */
function specs(known: readonly ToolSpec[], ledger: Ledger): ToolSpec[] {
  const seen = [...ledger.tools()].sort(cpCompare).map((name) => new ToolSpec(name, ""));
  return mergeTools(known, seen);
}

function finish(claims: Claim[], ledger: Ledger, toolSpecs: ToolSpec[], options: CheckOptions): Receipt {
  const diffOptions: DiffOptions = {
    knownTools: toolSpecs.map((s) => s.name),
    ignoreTools: options.ignoreTools,
    llmBlocks: options.llmBlocks,
    checkArgs: options.checkArgs,
  };
  return diff(claims, ledger, diffOptions);
}

/** Framework-free check of `reply` against `ledger`. */
export function check(reply: string, ledger: Ledger, options: CheckOptions = {}): Receipt {
  const toolSpecs = specs(normalizeToolSpecs(options.knownTools ?? []), ledger);
  const extractor: ClaimExtractor = options.extractor ?? new RuleExtractor();
  return finish(extractor.extract(reply, toolSpecs), ledger, toolSpecs, options);
}

/**
 * `check()` for async code: the same receipt, with the extractor's `aextract` awaited when it
 * defines one. With the rules alone there is no I/O and the result equals `check()`.
 */
export async function acheck(reply: string, ledger: Ledger, options: CheckOptions = {}): Promise<Receipt> {
  const toolSpecs = specs(normalizeToolSpecs(options.knownTools ?? []), ledger);
  const extractor: ClaimExtractor = options.extractor ?? new RuleExtractor();
  const claims =
    typeof extractor.aextract === "function"
      ? await extractor.aextract(reply, toolSpecs)
      : extractor.extract(reply, toolSpecs);
  return finish(claims, ledger, toolSpecs, options);
}
