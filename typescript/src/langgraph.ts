/**
 * LangGraph.js and LangChain.js: the callback handler for live runs (port of langgraph.py). Everything
 * is duck-typed, so no framework is imported and `dependencies` stays empty. The differences from Python
 * are in the README; each comes from how LangChain.js delivers events. `graph_messages.ts` reads finished runs.
 */

import { ToolCapture } from "./capture";
import { ToolSpec, typeName } from "./claims";
import type { Ledger, LedgerEntry } from "./ledger";
import { messageText } from "./llm";

/** Thrown by `interrupt()` and a drained graph. The tool did not fail: it paused, and reruns on resume. */
const PAUSE_NAMES = new Set(["GraphInterrupt", "NodeInterrupt", "GraphDrained"]);

export function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object";
}

/** The message's `ai`, `tool` or other type, or `null`. Before @langchain/core 1.0 only `_getType()` has it. */
export function kindOf(message: unknown): string | null {
  if (!isObject(message)) return null;
  if (typeof message._getType === "function") {
    const kind: unknown = (message._getType as () => unknown).call(message);
    return typeof kind === "string" ? kind : null;
  }
  return typeof message.type === "string" ? message.type : null;
}

function isToolMessage(value: unknown): value is Record<string, unknown> {
  return kindOf(value) === "tool" && typeof (value as Record<string, unknown>).tool_call_id === "string";
}

/** Bound tools from a chat model's `invocation_params`: the OpenAI and Anthropic shapes. Anything else,
 * such as Gemini's `functionDeclarations`, is skipped. */
export function toolSpecsFromInvocationParams(params: unknown): ToolSpec[] {
  const tools = isObject(params) ? params.tools : undefined;
  if (!Array.isArray(tools)) return [];
  const specs: ToolSpec[] = [];
  for (const tool of tools as unknown[]) {
    if (!isObject(tool)) continue;
    const source = isObject(tool.function) ? tool.function : tool;
    const { name, description } = source;
    if (typeof name === "string" && name !== "") specs.push(new ToolSpec(name, typeof description === "string" ? description : ""));
  }
  return specs;
}

/** A `Command`'s messages, from a record `update` or a list of `[key, value]` pairs. */
function commandMessages(update: unknown): unknown[] {
  const found: unknown[] = Array.isArray(update)
    ? (update as unknown[]).filter((pair) => Array.isArray(pair) && pair[0] === "messages").map((pair) => (pair as unknown[])[1])
    : isObject(update) ? [update.messages] : [];
  return found.flat();
}

/** The output's `ToolMessage` when it carries one: itself, or a `Command`'s, preferring the one for
 * this call. Anything else comes back unchanged. */
function toolMessageFrom(output: unknown, callId: string | null): unknown {
  if (isToolMessage(output) || !isObject(output)) return output;
  const messages = commandMessages(output.update).filter(isToolMessage);
  return messages.find((m) => m.tool_call_id === callId) ?? messages[0] ?? output;
}

function isPause(error: unknown): boolean {
  try {
    if (!isObject(error)) return false;
    return error.is_bubble_up === true || PAUSE_NAMES.has(error.name as string);
  } catch {
    return false;
  }
}

/** The tool call's arguments: LangChain.js hands them over as a JSON string. */
function parseInput(input: unknown): unknown {
  if (typeof input !== "string") return input;
  try {
    const parsed: unknown = JSON.parse(input);
    return isObject(parsed) && !Array.isArray(parsed) ? parsed : input;
  } catch {
    return input;
  }
}

/** Chain events wrap a value that is not an object as `{ output }`. */
function unwrapOutput(outputs: unknown): unknown {
  try {
    if (isObject(outputs) && Object.keys(outputs).length === 1 && "output" in outputs) return outputs.output;
  } catch {
    // Read on as it came: `handleToolEnd` copes with what it cannot read.
  }
  return outputs;
}

export interface ReceiptCallbackHandlerOptions {
  captureResults?: boolean | undefined;
  maxResultChars?: number | undefined;
}

/**
 * Records every tool call of a run into `ledger`. Entries are appended when they close: tool end, tool
 * error or `flush`. No method throws, because LangChain.js would log and swallow it, losing the call.
 */
export class ReceiptCallbackHandler {
  readonly name = "toolreceipt";
  /** LangChain.js runs handlers in the background by default, so `invoke` can resolve before a tool's
   * end event, and an honest reply would be judged on a call still open. */
  readonly awaitHandlers = true;
  readonly ledger: Ledger;
  /** @internal */
  readonly capture: ToolCapture;
  private readonly runnableCalls = new Map<string, string>();

  constructor(ledger: Ledger, options: ReceiptCallbackHandlerOptions = {}) {
    this.ledger = ledger;
    this.capture = new ToolCapture(options.captureResults ?? true, options.maxResultChars ?? 2000, ledger);
  }

  /** Tools the run could see, by name. A non-empty description is never overridden. */
  get knownTools(): Map<string, string> {
    return this.capture.knownTools;
  }

  private addKnown(name: string, description: string): void {
    if (!this.knownTools.get(name)) this.knownTools.set(name, description);
  }

  handleChatModelStart(_llm: unknown, _messages: unknown, _runId: string, _parentRunId?: string, extraParams?: Record<string, unknown>): void {
    try {
      for (const { name, description } of toolSpecsFromInvocationParams(extraParams?.invocation_params)) this.addKnown(name, description);
    } catch {
      // A broken payload costs the tool list, never the run.
    }
  }

  handleToolStart(tool: unknown, input: unknown, runId: string, parentRunId?: string, _tags?: string[], _metadata?: Record<string, unknown>, runName?: string, toolCallId?: string): void {
    let name = "unknown";
    let description = "";
    try {
      const serialized = isObject(tool) ? tool : {};
      const fallback = serialized.name;
      if (typeof runName === "string" && runName !== "") name = runName;
      else if (typeof fallback === "string" && fallback !== "") name = fallback;
      if (typeof serialized.description === "string") description = serialized.description;
    } catch {
      // Still open an entry below: a call is never missing from the ledger.
    }
    this.addKnown(name, description);
    const callId = typeof toolCallId === "string" && toolCallId !== "" ? toolCallId : runId;
    // A real tool inside a binding fires both kinds of event: its own event records the call.
    const runnable = this.runnableCalls.get(callId);
    if (runnable !== undefined) this.capture.discard(runnable);
    this.capture.start(runId, name, parseInput(input), callId, parentRunId ?? null);
  }

  handleToolEnd(output: unknown, runId: string): void {
    let read: unknown = null;
    let failed = false;
    try {
      const entry = this.capture.openEntry(runId);
      const message = toolMessageFrom(output, entry?.callId ?? null);
      if (isToolMessage(message)) {
        failed = message.status === "error";
        if (entry !== undefined) this.correct(entry, message);
        read = messageText(message);
      } else {
        read = message;
      }
    } catch {
      // The tool returned but its output cannot be read: it ran, with no result.
      read = null;
    }
    if (failed) this.capture.report(runId, typeof read === "string" ? read : "");
    else this.capture.finish(runId, read);
  }

  /** A `ToolMessage` knows the call better than the start event: `runName` can be a caller's trace label. */
  private correct(entry: LedgerEntry, message: Record<string, unknown>): void {
    const { name, tool_call_id: callId } = message;
    if (typeof callId === "string" && callId !== "") entry.callId = callId;
    if (typeof name !== "string" || name === "" || name === entry.tool) return;
    if (this.knownTools.get(entry.tool) === "") this.knownTools.delete(entry.tool);
    entry.tool = name;
    this.addKnown(name, "");
  }

  handleToolError(error: unknown, runId: string): void {
    if (isPause(error)) this.capture.discard(runId);
    else this.capture.fail(runId, error);
  }

  /** A tool from `asTool` is a runnable, so it fires chain events and no tool events. Its run takes a tool
   * call as input and bears the tool's name; `name:parse_input` and the graph's own nodes do not. */
  handleChainStart(_chain: unknown, inputs: unknown, runId: string, parentRunId?: string, _tags?: string[], _metadata?: Record<string, unknown>, _runType?: string, runName?: string): void {
    try {
      if (!isObject(inputs) || inputs.type !== "tool_call" || typeof inputs.name !== "string" || inputs.name !== runName) return;
      const callId = typeof inputs.id === "string" && inputs.id !== "" ? inputs.id : runId;
      this.addKnown(inputs.name, "");
      this.capture.start(runId, inputs.name, inputs.args, callId, parentRunId ?? null);
      this.runnableCalls.set(callId, runId);
    } catch {
      // A broken payload is not a tool call this handler can name.
    }
  }

  handleChainEnd(outputs: unknown, runId: string): void {
    if (this.capture.openEntry(runId) === undefined) return;
    this.handleToolEnd(unwrapOutput(outputs), runId);
  }

  handleChainError(error: unknown, runId: string): void {
    if (this.capture.openEntry(runId) !== undefined) this.handleToolError(error, runId);
  }

  /** Close every open call as `status: "error", error: "no result"`. */
  flush(): void {
    this.capture.flush();
  }
}

type Callbacks<T> = T extends { callbacks: readonly (infer H)[] } ? Array<H | ReceiptCallbackHandler>
  : T extends { callbacks: infer M extends object } ? M
  : T extends { callbacks?: infer C } ? (unknown extends C ? ReceiptCallbackHandler[] : Exclude<C, undefined> | ReceiptCallbackHandler[])
  : ReceiptCallbackHandler[];

/** `T` with a callback handler added: a manager stays a manager, and anything else is a list. An
 * optional `callbacks`, as on `RunnableConfig`, may be either. */
export type ConfigWith<T> = Omit<T, "callbacks"> & { callbacks: Callbacks<T> };

/** A shallow copy of `base` with `handler` added. A manager is copied, so the caller's is not changed. */
export function mergeConfig<T extends object = object>(base: T | null | undefined, handler: ReceiptCallbackHandler): ConfigWith<T> {
  const merged: Record<string, unknown> = { ...base };
  const callbacks = merged.callbacks;
  const manager = callbacks as { copy?: unknown; addHandler?: unknown } | null | undefined;
  if (callbacks === null || callbacks === undefined) {
    merged.callbacks = [handler];
  } else if (Array.isArray(callbacks)) {
    merged.callbacks = [...(callbacks as unknown[]), handler];
  } else if (typeof manager?.copy === "function" && typeof manager.addHandler === "function") {
    const copy = (manager.copy as () => { addHandler: (h: unknown, inherit: boolean) => void }).call(callbacks);
    copy.addHandler(handler, true);
    merged.callbacks = copy;
  } else {
    throw new TypeError(`config.callbacks must be a list of handlers or a CallbackManager, got ${typeName(callbacks)}`);
  }
  return merged as ConfigWith<T>;
}
