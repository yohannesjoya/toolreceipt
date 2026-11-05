/** `check()`, `acheck()` and the `Receipts` facade (port of receipts.py). */

import { ToolCapture, wrapToolRecord } from "./capture";
import { type Claim, type ClaimExtractor, RuleExtractor, type ToolLike, ToolSpec, cloneClaim, normalizeToolSpecs } from "./claims";
import { type DiffOptions, diff } from "./diff";
import { Ledger, LedgerEntry } from "./ledger";
import { cpCompare } from "./pystr";
import type { Receipt } from "./receipt";

export interface CheckOptions {
  /** Extra tools to match claims against: names, `[name, description]` pairs, tool dicts or
   * objects with a `name`. Merged with the tools the ledger saw. */
  knownTools?: Iterable<ToolLike> | undefined;
  /** Tools the run may use where the ledger cannot see them: a provider's or the browser's. The
   * extractor sees them, so a claim about one is not pinned on another tool, and that claim stays
   * unmatched unless the ledger holds the tool or `knownTools` names it. */
  unseenTools?: Iterable<ToolLike> | undefined;
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

/** `[tools the extractor sees, tools it sees that are not judged]`. */
function visible(judged: ToolSpec[], options: CheckOptions): [ToolSpec[], Set<string>] {
  const unseen = normalizeToolSpecs(options.unseenTools ?? []);
  if (unseen.length === 0) return [judged, new Set()];
  const names = new Set(judged.map((s) => s.name));
  return [mergeTools(judged, unseen), new Set(unseen.map((s) => s.name).filter((name) => !names.has(name)))];
}

function finish(claims: Claim[], ledger: Ledger, judged: ToolSpec[], hidden: Set<string>, options: CheckOptions): Receipt {
  const diffOptions: DiffOptions = {
    knownTools: judged.map((s) => s.name),
    ignoreTools: options.ignoreTools,
    llmBlocks: options.llmBlocks,
    checkArgs: options.checkArgs,
  };
  const ok = ledger.okTools();
  const settled = hidden.size === 0 ? claims : claims.map((claim) => {
    const unjudged = claim.candidates.some((c) => hidden.has(c)) && !claim.candidates.some((c) => ok.has(c));
    return unjudged ? cloneClaim(claim, { candidates: [] }) : claim;
  });
  return diff(settled, ledger, diffOptions);
}

/** Framework-free check of `reply` against `ledger`. */
export function check(reply: string, ledger: Ledger, options: CheckOptions = {}): Receipt {
  const judged = specs(normalizeToolSpecs(options.knownTools ?? []), ledger);
  const [toolSpecs, hidden] = visible(judged, options);
  const extractor: ClaimExtractor = options.extractor ?? new RuleExtractor();
  return finish(extractor.extract(reply, toolSpecs), ledger, judged, hidden, options);
}

/**
 * `check()` for async code: the same receipt, with the extractor's `aextract` awaited when it
 * defines one. With the rules alone there is no I/O and the result equals `check()`.
 */
export async function acheck(reply: string, ledger: Ledger, options: CheckOptions = {}): Promise<Receipt> {
  const judged = specs(normalizeToolSpecs(options.knownTools ?? []), ledger);
  const [toolSpecs, hidden] = visible(judged, options);
  const extractor: ClaimExtractor = options.extractor ?? new RuleExtractor();
  const claims =
    typeof extractor.aextract === "function"
      ? await extractor.aextract(reply, toolSpecs)
      : extractor.extract(reply, toolSpecs);
  return finish(claims, ledger, judged, hidden, options);
}

export interface ReceiptsOptions extends CheckOptions {
  /** Keep tool arguments and results in the ledger; `false` also turns off the argument check. */
  captureResults?: boolean | undefined;
  /** Longest result or error text kept, in code points. */
  maxResultChars?: number | undefined;
}

/** One per agent: `wrapTools()` before the run, `check()` on the final reply. */
export class Receipts {
  private readonly options: CheckOptions;
  private readonly known: ToolSpec[];
  private readonly unseen: ToolSpec[];
  private readonly captureResults: boolean;
  private readonly maxResultChars: number;
  private readonly runs = new WeakMap<object, ToolCapture>();
  private current: ToolCapture | null = null;
  private wrapCount = 0;

  constructor(options: ReceiptsOptions = {}) {
    this.known = normalizeToolSpecs(options.knownTools ?? []);
    this.unseen = normalizeToolSpecs(options.unseenTools ?? []);
    this.options = {
      ignoreTools: new Set(options.ignoreTools ?? []),
      extractor: options.extractor,
      llmBlocks: options.llmBlocks,
      checkArgs: options.checkArgs,
    };
    this.captureResults = options.captureResults ?? true;
    this.maxResultChars = options.maxResultChars ?? 2000;
  }

  /**
   * A copy of `tools` that records each call, for one run. After a second call, `check()` needs the
   * returned record, so one request can never read another's ledger. A long-lived agent holds its
   * tools for every request: wrapping once would mix requests, so use `checkResult` there.
   */
  wrapTools<T extends object>(tools: T): T {
    const capture = new ToolCapture(this.captureResults, this.maxResultChars);
    const wrapped = wrapToolRecord(tools, capture);
    this.runs.set(wrapped, capture);
    this.current = capture;
    this.wrapCount++;
    return wrapped;
  }

  /** The newest run's ledger, or `null` before `wrapTools()`. To read a run's receipt with more than
   * one run, pass its record to `check()`. */
  get ledger(): Ledger | null {
    return this.current === null ? null : this.current.ledger;
  }

  private captureFor(source: object | null | undefined): ToolCapture {
    if (source === null || source === undefined) {
      if (this.current === null) throw new TypeError("no ledger: call wrapTools() before the run or pass a ledger");
      if (this.wrapCount > 1) throw new TypeError("wrapTools() was called more than once: pass the record it returned for this run");
      return this.current;
    }
    const capture = this.runs.get(source);
    if (capture === undefined) throw new TypeError("source is not a Ledger or a record this instance's wrapTools() returned");
    return capture;
  }

  private resolve(source: Ledger | object | null | undefined): [Ledger, CheckOptions] {
    let ledger: Ledger;
    let known = this.known;
    let unseen = this.unseen;
    if (source instanceof Ledger) {
      ledger = source;
    } else {
      const capture = this.captureFor(source);
      capture.flush();
      ledger = capture.ledger;
      known = mergeTools(known, [...capture.knownTools].map(([name, description]) => new ToolSpec(name, description)));
      unseen = mergeTools(unseen, [...capture.unseenTools].map(([name, description]) => new ToolSpec(name, description)));
    }
    if (!this.captureResults) {
      ledger = new Ledger(ledger.entries.map((e) => new LedgerEntry({ ...e, args: {}, result: null })));
    }
    return [ledger, { ...this.options, knownTools: known, unseenTools: unseen }];
  }

  /** Check `reply`. `source` is a `Ledger` or a record `wrapTools()` returned; with none, the one
   * run is read. */
  check(reply: string, source?: Ledger | object | null): Receipt {
    const [ledger, options] = this.resolve(source);
    return check(reply, ledger, options);
  }

  /** `check()` for async code; see the module-level `acheck`. */
  async acheck(reply: string, source?: Ledger | object | null): Promise<Receipt> {
    const [ledger, options] = this.resolve(source);
    return acheck(reply, ledger, options);
  }
}
