/** `check()`, `acheck()` and the `Receipts` facade (port of receipts.py). */

import { ToolCapture, wrapToolRecord } from "./capture";
import { type Claim, type ClaimExtractor, RuleExtractor, type ToolLike, ToolSpec, normalizeToolSpecs } from "./claims";
import { type DiffOptions, diff } from "./diff";
import { Ledger, LedgerEntry } from "./ledger";
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
  private readonly captureResults: boolean;
  private readonly maxResultChars: number;
  private readonly runs = new WeakMap<object, ToolCapture>();
  private current: ToolCapture | null = null;

  constructor(options: ReceiptsOptions = {}) {
    this.known = normalizeToolSpecs(options.knownTools ?? []);
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
   * A copy of `tools` that records each call, for one run. The newest run is what `check()` reads
   * by default; pass the returned record to `check()` to read a specific one, so one `Receipts`
   * can serve concurrent runs. A long-lived agent holds its tools for every request: wrapping once
   * would mix requests, so use `checkResult` there.
   */
  wrapTools<T extends object>(tools: T): T {
    const capture = new ToolCapture(this.captureResults, this.maxResultChars);
    const wrapped = wrapToolRecord(tools, capture);
    this.runs.set(wrapped, capture);
    this.current = capture;
    return wrapped;
  }

  /** The newest run's ledger, or `null` before `wrapTools()`. */
  get ledger(): Ledger | null {
    return this.current === null ? null : this.current.ledger;
  }

  private captureFor(source: object | null | undefined): ToolCapture {
    if (source === null || source === undefined) {
      if (this.current === null) throw new TypeError("no ledger: call wrapTools() before the run or pass a ledger");
      return this.current;
    }
    const capture = this.runs.get(source);
    if (capture === undefined) throw new TypeError("source is not a Ledger or a record this instance's wrapTools() returned");
    return capture;
  }

  private resolve(source: Ledger | object | null | undefined): [Ledger, CheckOptions] {
    let ledger: Ledger;
    let known = this.known;
    if (source instanceof Ledger) {
      ledger = source;
    } else {
      const capture = this.captureFor(source);
      capture.flush();
      ledger = capture.ledger;
      known = mergeTools(known, [...capture.knownTools].map(([name, description]) => new ToolSpec(name, description)));
    }
    if (!this.captureResults) {
      ledger = new Ledger(ledger.entries.map((e) => new LedgerEntry({ ...e, args: {}, result: null })));
    }
    return [ledger, { ...this.options, knownTools: known }];
  }

  /** Check `reply`. `source` is a `Ledger` or a record `wrapTools()` returned; with none, the
   * newest run is read. */
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
