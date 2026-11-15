/**
 * `receiptGuard`: a `langchain` agent middleware that checks each final reply before the agent returns it,
 * and sends the agent back once when the reply claims an action its tool calls do not back (port of
 * python/toolreceipt/langchain.py). A plain object, so nothing is imported and `dependencies` stays empty.
 */

import type { Claim, ClaimExtractor, ToolLike } from "./claims";
import { cloneClaim, normalizeToolSpecs } from "./claims";
import { finalReplyFromGraph, ledgerFromGraph } from "./graph_messages";
import { isObject, kindOf } from "./langgraph";
import { NOTE_NAME } from "./messages";
import { type Receipt, formatObserved, renderSummary } from "./receipt";
import type { Json } from "./types";
import { acheck } from "./receipts";

export interface ReceiptGuardOptions {
  /** Notes per user turn before the reply goes through as it is; default 1. */
  maxRetries?: number | undefined;
  extractor?: ClaimExtractor | null | undefined;
  ignoreTools?: Iterable<string> | undefined;
  llmBlocks?: boolean | undefined;
  checkArgs?: boolean | undefined;
}

/** What the judged reply carries in `response_metadata.toolreceipt`. */
export interface GuardOutcome {
  outcome: "passed" | "fixed" | "unresolved";
  retries: number;
  receipt: Record<string, Json>;
}

type State = { messages: unknown[] };
type Update = { messages: unknown[]; jumpTo?: "model" };

/** The shape `createAgent` reads from a middleware. */
export interface AgentMiddlewareLike {
  readonly name: "ReceiptGuard";
  readonly afterModel: { canJumpTo: ["model"]; hook: (state: State) => Promise<Update | undefined> };
}

const LINES: Record<string, (claim: Claim) => string> = {
  not_called: (c) => `\`${c.tool ?? ""}\` was not called.`,
  tool_failed: (c) => `\`${c.tool ?? ""}\` failed.`,
  args_mismatch: (c) => `\`${c.tool ?? ""}\` ran with ${c.observed ? formatObserved(c.observed) : "other arguments"}, not what the reply says.`,
};

/** One line per false claim, worded by reason. It never asks for a tool call, so a call that ran is not
 * repeated, and it carries no tool result or error text. */
function noteText(receipt: Receipt): string {
  const lines = receipt.claimedButNeverRan.map((c) => `- "${c.text}": ${(LINES[c.reason ?? ""] ?? LINES.not_called!)(c)}`);
  return [
    "Your last reply says things the tool calls in this turn do not show:",
    ...lines,
    "",
    "Rewrite the reply so it says only what the tool calls show. Do not call tools to fix this.",
  ].join("\n");
}

/** The receipt with no tool arguments or results: no ledger, unreported calls cut to their tool, no
 * `observed` values, and a summary without the `ledger k=v` part. The outcome rides on a message UIs
 * render and checkpointers keep; the messages hold the rest. */
function storedReceipt(receipt: Receipt): Record<string, Json> {
  const { ledger: _ledger, ...stored } = receipt.toDict();
  stored.ran_but_unreported = receipt.ranButUnreported.map((e) => ({ tool: e.tool, status: e.status, call_id: e.callId }));
  for (const key of ["confirmed", "claimed_but_never_ran", "unmatched", "suspected"]) {
    for (const claim of stored[key] as Array<Record<string, Json>>) claim.observed = null;
  }
  const neverRan = receipt.claimedButNeverRan.map((c) => cloneClaim(c, { observed: null }));
  stored.summary = renderSummary(receipt.verdict, neverRan, receipt.ranButUnreported, receipt.suspected, receipt.confirmed.length);
  return stored;
}

function isNote(message: unknown): boolean {
  return kindOf(message) === "human" && isObject(message) && message.name === NOTE_NAME;
}

/** Notes since the last human message that is not a note. */
function notesThisTurn(messages: unknown[]): number {
  let count = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (kindOf(messages[i]) !== "human") continue;
    if (!isNote(messages[i])) break;
    count++;
  }
  return count;
}

/** A copy of an AI message with more `response_metadata`, keeping its class, id and fields, so the
 * reducer replaces the reply in place. */
function withMetadata(message: Record<string, unknown>, extra: Record<string, unknown>): unknown {
  const copy = Object.create(Object.getPrototypeOf(message) as object | null, Object.getOwnPropertyDescriptors(message)) as Record<string, unknown>;
  copy.response_metadata = { ...(isObject(message.response_metadata) ? message.response_metadata : {}), ...extra };
  return copy;
}

/**
 * Checks each final reply against this turn's tool calls. When the receipt is not ok, it adds a user
 * message named `toolreceipt` and sends the agent back, at most `maxRetries` times per user turn. The
 * judged reply carries a `GuardOutcome` in `response_metadata.toolreceipt`. It never blocks or rewrites.
 */
export function receiptGuard(tools: Iterable<ToolLike>, options: ReceiptGuardOptions = {}): AgentMiddlewareLike {
  const maxRetries = options.maxRetries ?? 1;
  if (!Number.isInteger(maxRetries) || maxRetries < 0) throw new RangeError(`maxRetries must be a whole number, 0 or more, got ${String(maxRetries)}`);
  const knownTools = normalizeToolSpecs(tools);
  // Copied once: a generator would be used up by the first check.
  const ignoreTools = [...(options.ignoreTools ?? [])];
  const checkOptions = { knownTools, extractor: options.extractor, ignoreTools, llmBlocks: options.llmBlocks, checkArgs: options.checkArgs };

  async function hook(state: State): Promise<Update | undefined> {
    const messages = state.messages;
    const last = messages.at(-1);
    if (kindOf(last) !== "ai" || !isObject(last)) return undefined;
    if (Array.isArray(last.tool_calls) && last.tool_calls.length > 0) return undefined;
    const receipt = await acheck(finalReplyFromGraph(messages), ledgerFromGraph(messages), checkOptions);
    const retries = notesThisTurn(messages);
    if (!receipt.ok && retries < maxRetries) {
      return { messages: [{ role: "user", name: NOTE_NAME, content: noteText(receipt) }], jumpTo: "model" };
    }
    const stored = storedReceipt(receipt);
    const outcome: GuardOutcome = { outcome: receipt.ok ? (retries > 0 ? "fixed" : "passed") : "unresolved", retries, receipt: stored };
    return { messages: [withMetadata(last, { toolreceipt: outcome })] };
  }

  const guard = { name: "ReceiptGuard" as const, afterModel: { canJumpTo: ["model"] as ["model"], hook } };
  // The brand `createMiddleware` sets; `createAgent` accepts the object without it today.
  Object.defineProperty(guard, Symbol.for("AgentMiddleware"), { value: true, enumerable: false });
  return guard;
}
