/** `check()`, `acheck()` and the `Receipts` facade (port of receipts.py). */

import { ToolCapture, wrapToolRecord } from "./capture";
import { type Claim, type ClaimExtractor, RuleExtractor, type ToolLike, ToolSpec, cloneClaim, normalizeToolSpecs } from "./claims";
import { type DiffOptions, diff } from "./diff";
import { type ConfigWith, ReceiptCallbackHandler, mergeConfig } from "./langgraph";
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
  // A candidate the ledger holds decides the claim. Otherwise an unseen candidate may have done the
  // work, so the claim is not judged, even with a judged candidate that never ran.
  const held = ledger.tools();
  const settled = hidden.size === 0 ? claims : claims.map((claim) => {
    if (!claim.candidates.some((c) => hidden.has(c))) return claim;
    const candidates = claim.candidates.some((c) => held.has(c)) ? claim.candidates.filter((c) => !hidden.has(c)) : [];
    return cloneClaim(claim, { candidates });
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

/** One per agent: `wrapTools()` or `config()` before the run, `check()` on the final reply. */
export class Receipts {
  private readonly options: CheckOptions;
  private readonly known: ToolSpec[];
  private readonly unseen: ToolSpec[];
  private readonly captureResults: boolean;
  private readonly maxResultChars: number;
  private readonly runs = new WeakMap<object, ToolCapture>();
  private current: ToolCapture | null = null;
  private runCount = 0;

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
    this.runCount++;
    return wrapped;
  }

  /**
   * `base` with a new callback handler added, for one run of a LangGraph.js graph: pass it to `invoke`
   * or `stream`. Reuse it for every `invoke` of that run, a resume included: a new one on resume has
   * no record of the calls before the pause. After a second call, `check()` needs the config this
   * returned, so one request can never read another's ledger.
   */
  config<T extends object = object>(base?: T | null): ConfigWith<T> {
    const handler = new ReceiptCallbackHandler(new Ledger(), { captureResults: this.captureResults, maxResultChars: this.maxResultChars });
    this.runs.set(handler, handler.capture);
    this.current = handler.capture;
    this.runCount++;
    return mergeConfig(base, handler);
  }

  /** The newest run's ledger, or `null` before `wrapTools()` or `config()`. To read a run's receipt
   * with more than one run, pass its record or config to `check()`. */
  get ledger(): Ledger | null {
    return this.current === null ? null : this.current.ledger;
  }

  private captureFor(source: object | null | undefined): ToolCapture {
    if (source === null || source === undefined) {
      if (this.current === null) throw new TypeError("no ledger: call wrapTools() or config() before the run, or pass a ledger");
      if (this.runCount > 1) throw new TypeError("this instance has more than one run: pass the record wrapTools() or the config config() returned for this one");
      return this.current;
    }
    return this.runs.get(source) ?? this.captureFromConfig(source);
  }

  /** The run of the one receipt handler in `config.callbacks`, so a spread config still finds it. */
  private captureFromConfig(config: object): ToolCapture {
    const callbacks = (config as { callbacks?: unknown }).callbacks;
    if (callbacks === null || typeof callbacks !== "object") {
      throw new TypeError("source is not a Ledger, a record this instance's wrapTools() returned or a config its config() returned");
    }
    const manager = callbacks as { handlers?: unknown; inheritableHandlers?: unknown };
    const all: unknown[] = Array.isArray(callbacks)
      ? (callbacks as unknown[])
      : [...(Array.isArray(manager.handlers) ? (manager.handlers as unknown[]) : []), ...(Array.isArray(manager.inheritableHandlers) ? (manager.inheritableHandlers as unknown[]) : [])];
    const found = [...new Set(all)].filter((h): h is ReceiptCallbackHandler => h instanceof ReceiptCallbackHandler);
    const [handler] = found;
    if (handler === undefined) throw new TypeError("config holds no ReceiptCallbackHandler: pass the config returned by config()");
    if (found.length > 1) throw new TypeError("config holds more than one ReceiptCallbackHandler");
    const capture = this.runs.get(handler);
    if (capture === undefined) throw new TypeError("config comes from another Receipts instance");
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

  /** Check `reply`. `source` is a `Ledger`, a record `wrapTools()` returned or a config `config()`
   * returned; with none, the one run is read. */
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
