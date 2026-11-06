/**
 * Framework-free message input (port of messages.py): a ledger and a final reply from a message
 * list. Reads OpenAI Chat Completions messages, OpenAI Responses items, Anthropic messages and, in
 * this port only, Vercel AI SDK model messages (5 and newer), as plain objects, Maps or SDK class
 * instances, by key, with no SDK imported. Only the last turn is
 * read: the messages after the last `user` message. A plain transcript shape is read too:
 * role/content lines with settled ledger entries at the position where the tool settled.
 */

import { typeName } from "./claims";
import { Ledger, LedgerEntry, NO_RESULT, asText, coerceArgs } from "./ledger";
import { cpCompare, isMapping, pyQuote, pyStr, pyStrip, pyTruthy } from "./pystr";
import type { Receipt } from "./receipt";
import { type CheckOptions, acheck, check } from "./receipts";
import type { Json } from "./types";

export type MessageOptions = CheckOptions;

const SUPPORTED = "OpenAI Chat Completions messages, OpenAI Responses items, Anthropic messages or Vercel AI SDK model messages";
const TEXT_TYPES = new Set(["text", "output_text", "input_text"]);
const LANGCHAIN_TYPES = new Set(["ai", "human", "tool", "system"]);
const STORED_TYPES = new Set([...LANGCHAIN_TYPES, "function", "generic", "developer", "chat"]);
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

/** Python's `str.isspace` for one character; `false` for an empty string. */
function isSpace(ch: string): boolean {
  return ch !== "" && pyStrip(ch) === "";
}

/** Whether the first non-space character of `text` is a lowercase letter, as Python's `str.islower`. */
function opensLowercase(text: string): boolean {
  const first = [...pyStrip(text)][0];
  return first !== undefined && first === first.toLowerCase() && first !== first.toUpperCase();
}

/** A string as is; a block list as its text blocks joined; else `""`. Blocks join with a newline,
 * unless one side already has whitespace at the join and the next block opens with a lowercase letter:
 * that is one sentence split across blocks, and a newline would cut a negation off from its verb.
 * Anything else may open a new sentence, whose claim an earlier negation must not swallow. */
export function text(content: unknown): string {
  if (typeof content === "string") return content;
  const list = blocks(content);
  if (list === null) return "";
  let joined: string | null = null;
  for (const block of list) {
    const t = get(block, "text");
    if (!TEXT_TYPES.has(get(block, "type") as string) || typeof t !== "string") continue;
    if (joined === null) joined = t;
    else if ((isSpace(joined.slice(-1)) || isSpace(t.slice(0, 1))) && opensLowercase(t)) joined += t;
    else joined += "\n" + t;
  }
  return joined ?? "";
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

/** True for what a LangChain message serializes to: `JSON.stringify` gives `lc` and `kwargs`, and
 * `mapChatMessagesToStoredMessages` gives `type` and `data`. Neither holds its fields at the top level,
 * so reading one would give an empty ledger, and a false claim would read `clean`. */
export function isSerializedLangChain(message: unknown): boolean {
  if (get(message, "lc") !== null && get(message, "kwargs") !== null) return true;
  return STORED_TYPES.has(get(message, "type") as string) && get(message, "data") !== null;
}

export const SERIALIZED_LANGCHAIN = "serialized LangChain messages are not read; revive them with mapStoredMessagesToChatMessages";

function rejectUnsupported(message: unknown): void {
  if (isSerializedLangChain(message)) throw new TypeError(SERIALIZED_LANGCHAIN);
  const role = get(message, "role");
  if (role === "function" || get(message, "function_call") !== null) {
    throw new TypeError(`legacy OpenAI function_call messages are not supported; pass ${SUPPORTED}`);
  }
  const itemType = get(message, "type");
  // @langchain/core before 1.0 has `_getType()` and no `type` property.
  if ((role === null && LANGCHAIN_TYPES.has(itemType as string)) || typeof get(message, "_getType") === "function") {
    throw new TypeError(`LangChain messages are not read here; use ledgerFromGraph and finalReplyFromGraph, or pass ${SUPPORTED}`);
  }
  if (UNREAD_ITEM_TYPES.has(itemType as string)) {
    throw new TypeError(`${itemType as string} items are not supported yet; only function tool calls are read`);
  }
  // AI SDK 4 keeps `args` and `result` where 5 has `input` and `output`: reading on would lose the
  // arguments and the result, so an honest reply could be judged on an empty call.
  for (const block of blocks(get(message, "content")) ?? []) {
    const blockType = get(block, "type");
    const oldCall = blockType === "tool-call" && get(block, "input") === null && get(block, "args") !== null;
    const oldResult = blockType === "tool-result" && get(block, "output") === null && get(block, "result") !== null;
    if (oldCall || oldResult) {
      throw new TypeError(`AI SDK 4 messages are not supported (a ${blockType} block with ${oldCall ? "args" : "result"}); AI SDK 5 and newer are read`);
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
 * `tool_calls`, a Responses `function_call` item, Anthropic `tool_use` blocks, AI SDK `tool-call`
 * parts. A Responses item's `id` is the item id, never the call id. */
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
    const toolName = get(block, "toolName");
    if (get(block, "type") === "tool-call" && typeof toolName === "string" && toolName !== "") {
      out.push([idOf(get(block, "toolCallId")), toolName, coerceArgs(get(block, "input"))]);
    }
  }
  return out;
}

/** A call the run has to settle itself: not one an AI SDK provider ran inside the same message. */
function hasLocalCall(message: unknown): boolean {
  const provider = (blocks(get(message, "content")) ?? []).filter(
    (b) => get(b, "type") === "tool-call" && pyTruthy(get(b, "providerExecuted")),
  );
  return calls(message).length > provider.length;
}

/** `[text, isError]` of an AI SDK `tool-result` part's `output`. */
function outputOf(output: unknown): [string, boolean] {
  const type = get(output, "type");
  const value = get(output, "value");
  switch (type) {
    case "text":
    case "json":
      return [asText(value) ?? "", false];
    case "error-text":
    case "error-json":
      return [asText(value) ?? "", true];
    case "execution-denied":
      return [asText(get(output, "reason")) ?? "execution denied", true];
    case "content":
      return [text(value), false];
    default:
      return [asText(output) ?? "", false];
  }
}

/** `[call id, text, isError, tool name]` for every tool result the message carries. Anthropic's
 * `is_error` and an AI SDK `output` of an error type are the only explicit failure flags. Only an AI
 * SDK part names its tool. */
function results(message: unknown): Array<[string, string, boolean, string | null]> {
  const out: Array<[string, string, boolean, string | null]> = [];
  // An AI SDK `tool` message holds parts with their own ids, read below.
  const parts = blocks(get(message, "content")) ?? [];
  const aiSdkParts = parts.some((b) => {
    const type = get(b, "type");
    return typeof type === "string" && type.startsWith("tool-");
  });
  if (get(message, "role") === "tool" && !aiSdkParts) out.push([idOf(get(message, "tool_call_id")), text(get(message, "content")), false, null]);
  if (get(message, "type") === "function_call_output") out.push([idOf(get(message, "call_id")), text(get(message, "output")), false, null]);
  for (const block of parts) {
    if (get(block, "type") === "tool_result") {
      out.push([idOf(get(block, "tool_use_id")), text(get(block, "content")), pyTruthy(get(block, "is_error")), null]);
    }
    if (get(block, "type") === "tool-result") {
      const [body, isError] = outputOf(get(block, "output"));
      const name = get(block, "toolName");
      out.push([idOf(get(block, "toolCallId")), body, isError, typeof name === "string" && name !== "" ? name : null]);
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
 * no result is `status: "error", error: "no result"`; a result with no call is ignored, except an AI
 * SDK part, which names its tool and proves the tool ran: an approved tool runs before the request's
 * first step, its call in an earlier request. An empty or repeated id pairs first-in, first-out. Status
 * is `ok` for every result except an Anthropic `tool_result` with `is_error` and an AI SDK error
 * output. A settled entry is appended at its position and never pairs.
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
    for (const [callId, body, isError, name] of results(message)) {
      const entry = open.get(callId)?.shift();
      if (entry === undefined) {
        if (name !== null) {
          entries.push(new LedgerEntry({ tool: name, status: isError ? "error" : "ok", result: isError ? null : body, error: isError ? body : null, callId }));
        }
        continue;
      }
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
    if (hasLocalCall(message) || settled(message) !== null) return "";
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
