/**
 * Live ledger capture by wrapping tools' `execute`, for any record of `{ name: { execute } }`: the
 * Vercel AI SDK's `tools`, or your own. The counterpart of Python's LangGraph callback handler. No
 * framework is imported, and nothing here can throw into the agent's run.
 */

import { Ledger, LedgerEntry, NO_RESULT, asText, coerceArgs, errorText, truncate } from "./ledger";
import { typeName } from "./claims";

/** Thrown when a tools argument is not a record of tools. */
function assertRecord(tools: unknown, where: string): asserts tools is Record<string, unknown> {
  if (tools === null || typeof tools !== "object" || Array.isArray(tools) || tools instanceof Map) {
    throw new TypeError(`${where} takes a record of tools by name, got ${typeName(tools)}`);
  }
}

/** The tool's description when it has a string one, else `""`. */
function descriptionOf(tool: unknown): string {
  const description = tool !== null && typeof tool === "object" ? (tool as { description?: unknown }).description : undefined;
  return typeof description === "string" ? description : "";
}

/** True for a tool whose `execute` this package can wrap. A tool without one runs where the wrapper
 * cannot see it: a provider's or the browser's. */
function hasExecute(tool: unknown): boolean {
  return tool !== null && typeof tool === "object" && typeof (tool as { execute?: unknown }).execute === "function";
}

/** `[name, description]` of a record's tools, split into those with `execute` and those without. */
export function splitTools(tools: unknown, where: string): { runs: Array<[string, string]>; unseen: Array<[string, string]> } {
  assertRecord(tools, where);
  const runs: Array<[string, string]> = [];
  const unseen: Array<[string, string]> = [];
  for (const [name, tool] of Object.entries(tools)) (hasExecute(tool) ? runs : unseen).push([name, descriptionOf(tool)]);
  return { runs, unseen };
}

function isAsyncIterable(value: unknown): value is AsyncIterable<unknown> {
  return value !== null && typeof value === "object" && Symbol.asyncIterator in value;
}

export function isThenable(value: unknown): value is PromiseLike<unknown> {
  return value !== null && typeof value === "object" && typeof (value as { then?: unknown }).then === "function";
}

/** One run: its ledger, the tools it can see, and the calls still open. */
export class ToolCapture {
  readonly ledger = new Ledger();
  readonly knownTools = new Map<string, string>();
  readonly unseenTools = new Map<string, string>();
  private readonly captureResults: boolean;
  private readonly maxResultChars: number;
  private readonly inFlight = new Map<number, LedgerEntry>();
  private nextCall = 1;

  constructor(captureResults: boolean, maxResultChars: number) {
    this.captureResults = captureResults;
    this.maxResultChars = maxResultChars;
  }

  open(tool: string, input: unknown, options: unknown): number {
    const call = this.nextCall++;
    let callId: unknown = null;
    try {
      callId = (options as { toolCallId?: unknown } | null | undefined)?.toolCallId;
    } catch {
      // Keep the numbered id below: a call is never missing from the ledger.
    }
    this.inFlight.set(call, new LedgerEntry({
      tool,
      args: this.captureResults ? coerceArgs(input) : {},
      callId: typeof callId === "string" && callId !== "" ? callId : String(call),
      startedAt: Date.now() / 1000,
    }));
    return call;
  }

  private close(call: number): LedgerEntry | null {
    const entry = this.inFlight.get(call);
    if (entry === undefined) return null;
    this.inFlight.delete(call);
    entry.endedAt = Date.now() / 1000;
    this.ledger.append(entry);
    return entry;
  }

  finish(call: number, output: unknown): void {
    const entry = this.close(call);
    if (entry === null || !this.captureResults) return;
    entry.result = truncate(asText(output), this.maxResultChars);
  }

  fail(call: number, error: unknown): void {
    const entry = this.close(call);
    if (entry === null) return;
    entry.status = "error";
    entry.error = truncate(errorText(error), this.maxResultChars);
  }

  /** Close every open call as `status: "error", error: "no result"`. */
  flush(): void {
    for (const call of [...this.inFlight.keys()]) {
      const entry = this.close(call);
      if (entry !== null) {
        entry.status = "error";
        entry.error = NO_RESULT;
      }
    }
  }
}

async function* relay(source: AsyncIterable<unknown>, call: number, capture: ToolCapture): AsyncGenerator<unknown> {
  let last: unknown = undefined;
  try {
    for await (const value of source) {
      last = value;
      yield value;
    }
  } catch (error) {
    capture.fail(call, error);
    throw error;
  }
  capture.finish(call, last);
}

/** A plain value, not a stream or a promise, when the output cannot even be tested. */
function kindOf(output: unknown): "stream" | "promise" | "value" {
  try {
    if (isAsyncIterable(output)) return "stream";
    if (isThenable(output)) return "promise";
  } catch {
    // A Proxy can throw here; it is still the tool's output, and it goes back untouched.
  }
  return "value";
}

/** A copy of `tool`, prototype and own properties included, with `execute` replaced. A spread would
 * drop what sits on a prototype: a class-based tool's `needsApproval`, so a gated tool would run with
 * no approval. */
function withExecute(tool: object, execute: (...args: unknown[]) => unknown): object {
  const { execute: _replaced, ...own } = Object.getOwnPropertyDescriptors(tool) as Record<string, PropertyDescriptor>;
  const copy = Object.create(Object.getPrototypeOf(tool) as object | null, own) as object;
  Object.defineProperty(copy, "execute", { value: execute, enumerable: true, writable: true, configurable: true });
  return copy;
}

/** A record like `tools` whose every `execute` also reports to `capture`. A tool with no `execute`
 * passes through unchanged. Naming it as known would make an honest reply about it `fabricated`, and
 * hiding it from the extractor would pin that claim on a sibling, so it goes in `unseenTools`. */
export function wrapToolRecord<T extends object>(tools: T, capture: ToolCapture): T {
  assertRecord(tools, "wrapTools");
  const wrapped: Record<string, unknown> = {};
  for (const [name, tool] of Object.entries(tools)) {
    if (!hasExecute(tool)) {
      capture.unseenTools.set(name, descriptionOf(tool));
      wrapped[name] = tool;
      continue;
    }
    const execute = (tool as { execute: (...args: unknown[]) => unknown }).execute;
    capture.knownTools.set(name, descriptionOf(tool));
    // Not `async`: the SDK tests the raw return for an async iterable.
    wrapped[name] = withExecute(tool as object, (input: unknown, ...rest: unknown[]): unknown => {
      const call = capture.open(name, input, rest[0]);
      let output: unknown;
      try {
        output = execute.call(tool, input, ...rest);
      } catch (error) {
        capture.fail(call, error);
        throw error;
      }
      const kind = kindOf(output);
      if (kind === "stream") return relay(output as AsyncIterable<unknown>, call, capture);
      if (kind === "promise") {
        return Promise.resolve(output).then(
          (value) => {
            capture.finish(call, value);
            return value;
          },
          (error: unknown) => {
            capture.fail(call, error);
            throw error;
          },
        );
      }
      capture.finish(call, output);
      return output;
    });
  }
  return wrapped as T;
}
