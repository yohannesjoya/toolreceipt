/**
 * LangGraph.js and LangChain.js: a callback handler for live runs, and a reader for finished ones
 * (port of langgraph.py). Everything is duck-typed, so no framework is imported and `dependencies` stays
 * empty. The differences from Python are in the README; each comes from how LangChain.js delivers events.
 */

import { ToolCapture } from "./capture";
import { ToolSpec, typeName } from "./claims";
import { Ledger, LedgerEntry, NO_RESULT, coerceArgs } from "./ledger";
import { messageText } from "./llm";
import { SERIALIZED_LANGCHAIN, isSerializedLangChain } from "./messages";
import { pyStrip } from "./pystr";

const MESSAGE_TYPES = new Set(["ai", "human", "tool", "system", "remove", "generic", "developer", "function"]);
/** Thrown by `interrupt()` and a drained graph. The tool did not fail: it paused, and reruns on resume. */
const PAUSE_NAMES = new Set(["GraphInterrupt", "NodeInterrupt", "GraphDrained"]);

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object";
}

/** The message's `ai`, `tool` or other type, or `null`. Before @langchain/core 1.0 only `_getType()` has it. */
function kindOf(message: unknown): string | null {
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
    this.capture.start(runId, name, parseInput(input), typeof toolCallId === "string" && toolCallId !== "" ? toolCallId : runId, parentRunId ?? null);
  }

  handleToolEnd(output: unknown, runId: string): void {
    let read: unknown = null;
    let failed = false;
    let callId: string | null = null;
    try {
      const message = toolMessageFrom(output, this.capture.callId(runId));
      if (isToolMessage(message)) {
        failed = message.status === "error";
        callId = message.tool_call_id as string;
        read = messageText(message);
      } else {
        read = message;
      }
    } catch {
      // The tool returned but its output cannot be read: it ran, with no result.
      read = null;
    }
    if (failed) this.capture.report(runId, typeof read === "string" ? read : "", callId);
    else this.capture.finish(runId, read, callId);
  }

  handleToolError(error: unknown, runId: string): void {
    if (isPause(error)) this.capture.discard(runId);
    else this.capture.fail(runId, error);
  }

  /** Close every open call as `status: "error", error: "no result"`. */
  flush(): void {
    this.capture.flush();
  }
}

/** `T` with a callback handler added: a manager stays a manager, and anything else is a list. */
export type ConfigWith<T> = Omit<T, "callbacks"> & {
  callbacks: T extends { callbacks: readonly (infer H)[] } ? Array<H | ReceiptCallbackHandler>
    : T extends { callbacks: infer M extends object } ? M
    : ReceiptCallbackHandler[];
};

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

/** The messages of this turn: those after the last human message. A checkpointed result holds the whole
 * thread, and an earlier turn's call would back a claim this turn never acted on. Anything unreadable
 * throws: an empty ledger would misjudge the reply. */
function turnOf(source: unknown): unknown[] {
  const messages: unknown = isObject(source) && !Array.isArray(source) ? source.messages : source;
  if (!Array.isArray(messages)) {
    throw new TypeError(`expected a list of LangChain messages or a result with a messages list, got ${typeName(source)}`);
  }
  let start = 0;
  (messages as unknown[]).forEach((message, i) => {
    if (isSerializedLangChain(message)) throw new TypeError(SERIALIZED_LANGCHAIN);
    const kind = kindOf(message);
    const readable = isObject(message) && (typeof message._getType === "function" || (MESSAGE_TYPES.has(kind as string) && "content" in message));
    if (!readable) throw new TypeError(`expected a LangChain message, got ${typeName(message)}; for OpenAI, Anthropic and AI SDK messages use checkMessages`);
    if (kind === "human") start = i + 1;
  });
  return (messages as unknown[]).slice(start);
}

function toolCallsOf(message: unknown): Array<Record<string, unknown>> {
  const calls = (message as { tool_calls?: unknown }).tool_calls;
  return Array.isArray(calls) ? (calls as unknown[]).filter(isObject) : [];
}

/**
 * A ledger from a finished run's messages: each tool call paired with its `ToolMessage`. A call with no
 * result is an error, `"no result"`. Takes a graph result or its `messages`, and reads this turn only.
 */
export function ledgerFromGraph(source: unknown): Ledger {
  const entries: LedgerEntry[] = [];
  const open = new Map<string, LedgerEntry>();
  for (const message of turnOf(source)) {
    const kind = kindOf(message);
    if (kind === "ai") {
      for (const call of toolCallsOf(message)) {
        const callId = typeof call.id === "string" && call.id !== "" ? call.id : null;
        const entry = new LedgerEntry({ tool: typeof call.name === "string" ? call.name : "unknown", status: "error", args: coerceArgs(call.args), error: NO_RESULT, callId });
        entries.push(entry);
        if (callId !== null) open.set(callId, entry);
      }
    } else if (kind === "tool") {
      const callId = (message as { tool_call_id?: unknown }).tool_call_id as string;
      const matched = open.get(callId);
      if (matched === undefined) continue;
      open.delete(callId);
      const text = messageText(message);
      if ((message as { status?: unknown }).status === "error") {
        matched.error = text;
      } else {
        matched.status = "ok";
        matched.error = null;
        matched.result = text;
      }
    }
  }
  return new Ledger(entries);
}

/**
 * The text of this turn's last AI message with text, or `""` when the run ends on a tool call: that
 * message carries `tool_calls`, or a call or a `ToolMessage` follows it. Takes a graph result or its
 * `messages`.
 */
export function finalReplyFromGraph(source: unknown): string {
  for (const message of turnOf(source).reverse()) {
    const kind = kindOf(message);
    if (kind === "tool") return "";
    if (kind === "ai") {
      if (toolCallsOf(message).length > 0) return "";
      const text = messageText(message);
      if (pyStrip(text) !== "") return text;
    }
  }
  return "";
}
