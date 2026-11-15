/**
 * Compile-time check, run by `npm run check` after the build: the published types against the real
 * LangChain.js types, under the strictest resolution a user can pick. `toolreceipt` resolves to `dist/`.
 */
import { CallbackManager } from "@langchain/core/callbacks/manager";
import type { RunnableConfig } from "@langchain/core/runnables";
import { END, MessagesAnnotation, START, StateGraph } from "@langchain/langgraph";
import { ReceiptCallbackHandler, Ledger, Receipts, finalReplyFromGraph, ledgerFromGraph, mergeConfig } from "toolreceipt";

const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", () => ({ messages: [] }))
  .addEdge(START, "agent")
  .addEdge("agent", END)
  .compile();

const receipts = new Receipts();
const config = receipts.config();
export const assignable: RunnableConfig = config;
const threaded = receipts.config({ configurable: { thread_id: "t" }, tags: ["x"] });
export const tags: string[] = threaded.tags;
export const managed: CallbackManager = receipts.config({ callbacks: new CallbackManager() }).callbacks;
export const direct: RunnableConfig = { callbacks: [new ReceiptCallbackHandler(new Ledger())] };
export const merged: RunnableConfig = mergeConfig({ tags: ["x"] }, new ReceiptCallbackHandler(new Ledger()));

export function forwarded(base: RunnableConfig): RunnableConfig {
  const merged = receipts.config(base);
  // @ts-expect-error A forwarded config's `callbacks` may be a manager, so it cannot be indexed unchecked.
  void merged.callbacks[0];
  return merged;
}

export async function run(): Promise<string> {
  const result = await graph.invoke({ messages: [] }, config);
  await graph.invoke({ messages: [] }, threaded);
  await graph.stream({ messages: [] }, config);
  ledgerFromGraph(result);
  return receipts.check(finalReplyFromGraph(result), config).verdict;
}

// The guard is accepted by createAgent's middleware list, with real tool objects as its tools.
import { createAgent } from "langchain";
import { tool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { receiptGuard, type GuardOutcome } from "toolreceipt";
const sendEmail = tool(() => "sent", { name: "send_email", description: "Send an email", schema: z.object({}) });
export const guarded = createAgent({ model: new ChatOpenAI({ model: "gpt-4o-mini" }), tools: [sendEmail], middleware: [receiptGuard([sendEmail])] });
export function readOutcome(message: { response_metadata: Record<string, unknown> }): GuardOutcome | undefined {
  return message.response_metadata.toolreceipt as GuardOutcome | undefined;
}
