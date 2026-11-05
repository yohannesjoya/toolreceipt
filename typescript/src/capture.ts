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

/** `[name, description]` of every tool in a record. */
export function toolDescriptions(tools: unknown, where: string): Array<[string, string]> {
  assertRecord(tools, where);
  return Object.entries(tools).map(([name, tool]) => [name, descriptionOf(tool)]);
}

function isAsyncIterable(value: unknown): value is AsyncIterable<unknown> {
  return value !== null && typeof value === "object" && Symbol.asyncIterator in value;
}

function isThenable(value: unknown): value is PromiseLike<unknown> {
  return value !== null && typeof value === "object" && typeof (value as { then?: unknown }).then === "function";
}

/** One run: its ledger, the tools it can see, and the calls still open. */
export class ToolCapture {
  readonly ledger = new Ledger();
  readonly knownTools = new Map<string, string>();
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
    let text: string | null = null;
    try {
      text = asText(output);
    } catch {
      // The tool returned but its output cannot be read: it ran, with no result.
    }
    entry.result = truncate(text, this.maxResultChars);
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

/** A record like `tools` whose every `execute` also reports to `capture`. A tool with no `execute`
 * runs where the wrapper cannot see it, so it passes through unchanged and stays unknown: naming it
 * would make an honest reply about it `fabricated`. */
export function wrapToolRecord<T extends object>(tools: T, capture: ToolCapture): T {
  assertRecord(tools, "wrapTools");
  const wrapped: Record<string, unknown> = {};
  for (const [name, tool] of Object.entries(tools)) {
    const execute = tool !== null && typeof tool === "object" ? (tool as { execute?: unknown }).execute : undefined;
    if (typeof execute !== "function") {
      wrapped[name] = tool;
      continue;
    }
    capture.knownTools.set(name, descriptionOf(tool));
    wrapped[name] = {
      ...(tool as object),
      // Not `async`: the SDK tests the raw return for an async iterable.
      execute(input: unknown, ...rest: unknown[]): unknown {
        const call = capture.open(name, input, rest[0]);
        let output: unknown;
        try {
          output = (execute as (...args: unknown[]) => unknown).call(tool, input, ...rest);
        } catch (error) {
          capture.fail(call, error);
          throw error;
        }
        if (isAsyncIterable(output)) return relay(output, call, capture);
        if (isThenable(output)) {
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
      },
    };
  }
  return wrapped as T;
}
