/** Reading a finished LangGraph.js or LangChain.js run from its messages (port of langgraph.py). */

import { typeName } from "./claims";
import { isObject, kindOf } from "./langgraph";
import { Ledger, LedgerEntry, NO_RESULT, coerceArgs } from "./ledger";
import { messageText } from "./llm";
import { SERIALIZED_LANGCHAIN, isSerializedLangChain, text } from "./messages";
import { pyStrip } from "./pystr";

/** Name of the user message `receiptGuard` adds. It continues the run. */
export const NOTE_NAME = "toolreceipt";

const MESSAGE_TYPES = new Set(["ai", "human", "tool", "system", "remove", "generic", "developer", "function"]);

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
    if (kind === "human" && (message as { name?: unknown }).name !== NOTE_NAME) start = i + 1;
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

/** A reply's text, its blocks joined as `checkMessages` joins them. `AIMessage.text` joins them with
 * nothing, which lets a negation in one block swallow a claim in the next. */
function replyText(message: unknown): string {
  const content = (message as { content?: unknown }).content;
  if (typeof content === "string") return content;
  // LangChain allows plain strings beside blocks, and its own `.text` drops them: read them as text blocks.
  if (Array.isArray(content)) return text((content as unknown[]).map((b) => (typeof b === "string" ? { type: "text", text: b } : b)));
  return messageText(message);
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
      const reply = replyText(message);
      if (pyStrip(reply) !== "") return reply;
    }
  }
  return "";
}
