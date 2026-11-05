/**
 * Framework-free message input (port of messages.py): a ledger and a final reply from a message
 * list. Reads OpenAI Chat Completions messages, OpenAI Responses items and Anthropic messages as
 * plain objects, Maps or SDK class instances, by key, with no SDK imported. Only the last turn is
 * read: the messages after the last `user` message. A plain transcript shape is read too:
 * role/content lines with settled ledger entries at the position where the tool settled.
 */

import { typeName } from "./claims";
import { Ledger, LedgerEntry, NO_RESULT, coerceArgs } from "./ledger";
import { cpCompare, isMapping, pyDumps, pyQuote, pyStr, pyStrip, pyTruthy } from "./pystr";
import type { Receipt } from "./receipt";
import { type CheckOptions, acheck, check } from "./receipts";
import type { Json } from "./types";

export type MessageOptions = CheckOptions;

const SUPPORTED = "OpenAI Chat Completions messages, OpenAI Responses items or Anthropic messages";
const TEXT_TYPES = new Set(["text", "output_text", "input_text"]);
const LANGCHAIN_TYPES = new Set(["ai", "human", "tool", "system"]);
// Tool-call shapes this reader knows about but does not read yet: dropping them would empty the
// ledger and make every honest claim `not_called`.
const UNREAD_ITEM_TYPES = new Set([
  "custom_tool_call", "custom_tool_call_output", "mcp_call", "computer_call", "local_shell_call", "shell_call",
]);

/** A field by key (Maps) or property (plain objects, SDK instances); `null` when missing. */
export function get(obj: unknown, key: string): unknown {
  if (obj === null || typeof obj !== "object") return null;
  const value: unknown = obj instanceof Map ? obj.get(key) : (obj as Record<string, unknown>)[key];
  return value === undefined ? null : value;
}

function blocks(content: unknown): unknown[] | null {
  return Array.isArray(content) ? (content as unknown[]) : null;
}

/** A string as is; a block list as its text blocks joined by newlines; else `""`. */
export function text(content: unknown): string {
  if (typeof content === "string") return content;
  const list = blocks(content);
  if (list === null) return "";
  const parts: string[] = [];
  for (const block of list) {
    const t = get(block, "text");
    if (TEXT_TYPES.has(get(block, "type") as string) && typeof t === "string") parts.push(t);
  }
  return parts.join("\n");
}

/** A user message that starts a turn: one with text and no `tool_result` block. Anthropic returns
 * tool results inside user messages, and a tool's screenshot comes back as an image-only user
 * message; both continue the run. */
function isUser(message: unknown): boolean {
  if (get(message, "role") !== "user") return false;
  const content = get(message, "content");
  if (typeof content === "string") return pyStrip(content) !== "";
  const list = blocks(content);
  if (list === null) return true;
  if (list.some((b) => get(b, "type") === "tool_result")) return false;
  return pyStrip(text(content)) !== "";
}

function rejectUnsupported(message: unknown): void {
  const role = get(message, "role");
  if (role === "function" || get(message, "function_call") !== null) {
    throw new TypeError(`legacy OpenAI function_call messages are not supported; pass ${SUPPORTED}`);
  }
  const itemType = get(message, "type");
  // @langchain/core before 1.0 has `_getType()` and no `type` property.
  if ((role === null && LANGCHAIN_TYPES.has(itemType as string)) || typeof get(message, "_getType") === "function") {
    throw new TypeError(`LangChain messages are not read here; toolreceipt reads ${SUPPORTED}`);
  }
  if (UNREAD_ITEM_TYPES.has(itemType as string)) {
    throw new TypeError(`${itemType as string} items are not supported yet; only function tool calls are read`);
  }
  // Their tool calls sit in blocks this reader does not know, so skipping them would empty the
  // ledger and turn an honest reply into `fabricated`.
  for (const block of blocks(get(message, "content")) ?? []) {
    const blockType = get(block, "type");
    if (blockType === "tool-call" || blockType === "tool-result") {
      throw new TypeError(`Vercel AI SDK messages are not supported yet (a ${blockType} block); pass ${SUPPORTED}`);
    }
  }
  for (const call of blocks(get(message, "tool_calls")) ?? []) {
    const callType = get(call, "type");
    if (callType !== null && callType !== "function") {
      throw new TypeError(`tool calls of type ${pyQuote(callType)} are not supported yet; only function tool calls are read`);
    }
  }
}

/** OpenAI `arguments` is a JSON string; unparsable text is kept as `{ input: ... }`. */
function parseArgs(args: unknown): Record<string, Json> {
  if (typeof args === "string") {
    let parsed: unknown;
    try {
      parsed = JSON.parse(args);
    } catch {
      return coerceArgs(args);
    }
    return coerceArgs(parsed);
  }
  return coerceArgs(args);
}

function idOf(value: unknown): string {
  return pyTruthy(value) ? pyStr(value) : "";
}

/** `[call id, tool name, args]` for every tool call the message carries: Chat Completions
 * `tool_calls`, a Responses `function_call` item, Anthropic `tool_use` blocks. A Responses item's
 * `id` is the item id, never the call id. */
function calls(message: unknown): Array<[string, string, Record<string, Json>]> {
  const out: Array<[string, string, Record<string, Json>]> = [];
  for (const call of blocks(get(message, "tool_calls")) ?? []) {
    const fn = get(call, "function");
    const name = get(fn, "name");
    if (typeof name === "string" && name !== "") out.push([idOf(get(call, "id")), name, parseArgs(get(fn, "arguments"))]);
  }
  if (get(message, "type") === "function_call") {
    const name = get(message, "name");
    if (typeof name === "string" && name !== "") out.push([idOf(get(message, "call_id")), name, parseArgs(get(message, "arguments"))]);
  }
  for (const block of blocks(get(message, "content")) ?? []) {
    const name = get(block, "name");
    if (get(block, "type") === "tool_use" && typeof name === "string" && name !== "") {
      out.push([idOf(get(block, "id")), name, coerceArgs(get(block, "input"))]);
    }
  }
  return out;
}

/** `[call id, text, isError]` for every tool result the message carries. Anthropic's `is_error`
 * is the only explicit failure flag. */
function results(message: unknown): Array<[string, string, boolean]> {
  const out: Array<[string, string, boolean]> = [];
  if (get(message, "role") === "tool") out.push([idOf(get(message, "tool_call_id")), text(get(message, "content")), false]);
  if (get(message, "type") === "function_call_output") out.push([idOf(get(message, "call_id")), text(get(message, "output")), false]);
  for (const block of blocks(get(message, "content")) ?? []) {
    if (get(block, "type") === "tool_result") {
      out.push([idOf(get(block, "tool_use_id")), text(get(block, "content")), pyTruthy(get(block, "is_error"))]);
    }
  }
  return out;
}

/** An assistant message with non-empty text. A tool-call-only assistant message is not a turn. */
export function isTurn(message: unknown): boolean {
  return get(message, "role") === "assistant" && pyStrip(text(get(message, "content"))) !== "";
}

/** `dict(keys: ...)` or the type name: enough to find the item, no values (README "Privacy"). */
function shape(message: unknown): string {
  if (isMapping(message)) {
    const keys = message instanceof Map ? [...message.keys()].map(String) : Object.keys(message);
    return `dict(keys: ${keys.sort(cpCompare).join(", ")})`;
  }
  return typeName(message);
}

/** A result or error as the ledger stores it: a string or `null` as is, anything else as JSON. */
function asText(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "string") return value;
  try {
    return pyDumps(value);
  } catch {
    // eslint-disable-next-line @typescript-eslint/no-base-to-string
    return String(value);
  }
}

/** A settled entry's timestamp: a number or `null`; anything else throws, because
 * `Ledger.entries` sorts by it. */
function timestamp(message: unknown, key: string): number | null {
  const value = get(message, key);
  if (value === null) return null;
  if (typeof value !== "number") {
    throw new TypeError(`a settled transcript entry's ${key} must be a number or null, got ${typeName(value)} in ${shape(message)}`);
  }
  return value;
}

/**
 * The `LedgerEntry` a settled transcript entry records, else `null`. A settled entry is an item
 * whose `tool` is present and whose `role` and `type` are absent. A `status` other than `ok` or
 * `error` throws, an `error` entry with no text gets `NO_RESULT`, and an empty or non-string
 * `tool` throws. Error messages name the shape, never the payload.
 */
export function settled(item: unknown): LedgerEntry | null {
  const message: unknown = item instanceof LedgerEntry ? item.toDict() : item;
  const tool = get(message, "tool");
  if (tool === null || get(message, "role") !== null || get(message, "type") !== null) return null;
  if (typeof tool !== "string" || tool === "") {
    throw new TypeError(`a settled transcript entry needs a non-empty string tool, got ${typeName(tool)} in ${shape(message)}`);
  }
  const status = get(message, "status") ?? "ok";
  if (status !== "ok" && status !== "error") {
    const shown = typeof status === "string" ? pyQuote(status) : typeName(status);
    throw new TypeError(`a settled transcript entry's status must be 'ok' or 'error', got ${shown} in ${shape(message)}`);
  }
  let error = asText(get(message, "error"));
  if (status === "error" && !error) error = NO_RESULT;
  const callId = get(message, "call_id");
  return new LedgerEntry({
    tool,
    status,
    args: coerceArgs(get(message, "args")),
    result: asText(get(message, "result")),
    error,
    callId: callId === null ? null : pyStr(callId),
    parentRunId: get(message, "parent_run_id") as string | null,
    startedAt: timestamp(message, "started_at"),
    endedAt: timestamp(message, "ended_at"),
  });
}

/**
 * One entry per tool call of `messages`, in call order, paired with its result by id. A call with
 * no result is `status: "error", error: "no result"`; a result with no call is ignored; an empty
 * or repeated id pairs first-in, first-out. Status is `ok` for every result except an Anthropic
 * `tool_result` with `is_error`. A settled entry is appended at its position and never pairs.
 * Throws `TypeError` on an unsupported shape.
 */
export function pair(messages: Iterable<unknown>): LedgerEntry[] {
  const entries: LedgerEntry[] = [];
  const open = new Map<string, LedgerEntry[]>();
  for (const message of messages) {
    rejectUnsupported(message);
    const done = settled(message);
    if (done !== null) {
      entries.push(done);
      continue;
    }
    for (const [callId, name, args] of calls(message)) {
      const entry = new LedgerEntry({ tool: name, status: "error", args, error: NO_RESULT, callId });
      entries.push(entry);
      const queue = open.get(callId);
      if (queue) queue.push(entry);
      else open.set(callId, [entry]);
    }
    for (const [callId, body, isError] of results(message)) {
      const entry = open.get(callId)?.shift();
      if (entry === undefined) continue;
      if (isError) {
        entry.error = body;
      } else {
        entry.status = "ok";
        entry.error = null;
        entry.result = body;
      }
    }
  }
  return entries;
}

/** The last turn: the messages after the last user message that carries no tool result; all
 * when there is none. */
export function runMessages(messages: readonly unknown[]): unknown[] {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (isUser(messages[i])) return messages.slice(i + 1);
  }
  return [...messages];
}

/** One entry per tool call of the last turn, paired with its result by id; see `pair`. */
export function ledgerFromMessages(messages: readonly unknown[]): Ledger {
  return new Ledger(pair(runMessages(messages)));
}

/** Text of the last turn's last assistant message when no tool call or settled entry follows it
 * or sits in it, else `""` (a run that ends on a tool call has no final reply). */
export function finalReply(messages: readonly unknown[]): string {
  for (const message of runMessages(messages).reverse()) {
    if (calls(message).length > 0 || settled(message) !== null) return "";
    if (get(message, "role") === "assistant") return text(get(message, "content"));
  }
  return "";
}

/**
 * `check(finalReply(messages), ledgerFromMessages(messages), options)` with no framework.
 * `knownTools` is the `tools` list sent to the API; without it a tool that was never called
 * cannot be reported as `not_called`.
 */
export function checkMessages(messages: readonly unknown[], options: MessageOptions = {}): Receipt {
  return check(finalReply(messages), ledgerFromMessages(messages), options);
}

/** `checkMessages` for async code; see `acheck`. */
export async function acheckMessages(messages: readonly unknown[], options: MessageOptions = {}): Promise<Receipt> {
  return acheck(finalReply(messages), ledgerFromMessages(messages), options);
}
