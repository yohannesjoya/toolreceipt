/**
 * `receiptGuard` against the real `langchain` agent over a stubbed `fetch`: no network, no key. The same
 * cases as python/tests/test_guard.py, with the same outcomes.
 */
/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-return, @typescript-eslint/no-unsafe-argument */
import { beforeAll, describe, expect, it } from "vitest";

import { check, checkMessages, finalReplyFromGraph, ledgerFromGraph, receiptGuard } from "../src/index";

const nodeMajor = Number(process.versions.node.split(".")[0]);

interface Call { name: string; id: string; args?: unknown }
type Step = string | Call[];

describe.skipIf(nodeMajor < 20)("receiptGuard on real LangChain.js", () => {
  let core: any, langchain: any, openai: any, graphModule: any, zod: any;
  let tools: any[];
  const refunds: number[] = [];

  beforeAll(async () => {
    core = await import("@langchain/core/messages");
    const { tool } = await import("@langchain/core/tools");
    langchain = await import("langchain");
    openai = await import("@langchain/openai");
    graphModule = await import("@langchain/langgraph");
    zod = await import("zod");
    const { z } = zod;
    tools = [
      tool(({ amount }: { amount: number }) => {
        refunds.push(amount);
        return `refunded ${amount} (secret-result-text)`;
      }, { name: "issue_refund", description: "Issue a refund to the customer.", schema: z.object({ amount: z.number() }) }),
      tool(() => {
        throw new Error("SMTP gateway timeout (secret-error-text)");
      }, { name: "send_email", description: "Send an email to the customer.", schema: z.object({ subject: z.string() }) }),
      tool(() => "TCK-1", { name: "create_ticket", description: "Create a support ticket.", schema: z.object({ title: z.string() }) }),
    ];
  });

  /** OpenAI chat responses, one per step; `calls` counts model requests. Each response has its own id:
   * `messagesStateReducer` replaces a message with the same id. */
  function scripted(steps: Step[]) {
    const state = { calls: 0 };
    const fetch = (): Promise<Response> => {
      const step = steps[state.calls] ?? "";
      state.calls++;
      const message = typeof step === "string"
        ? { role: "assistant", content: step }
        : { role: "assistant", content: null, tool_calls: step.map((c) => ({ id: c.id, type: "function", function: { name: c.name, arguments: JSON.stringify(c.args ?? {}) } })) };
      const body = { id: `chatcmpl-${state.calls}`, object: "chat.completion", created: 1, model: "m", choices: [{ index: 0, message, finish_reason: typeof step === "string" ? "stop" : "tool_calls" }], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } };
      return Promise.resolve(new Response(JSON.stringify(body), { headers: { "content-type": "application/json" } }));
    };
    return { state, model: new openai.ChatOpenAI({ apiKey: "test", model: "gpt-4o-mini", configuration: { fetch } }) };
  }

  const call = (name: string, id: string, args: unknown = {}): Call[] => [{ name, id, args }];
  const input = (text = "Refund $32 and email me.") => ({ messages: [new core.HumanMessage(text)] });

  async function run(steps: Step[], options: Record<string, unknown> = {}) {
    refunds.length = 0;
    const { state, model } = scripted(steps);
    const agent = langchain.createAgent({ model, tools, middleware: [receiptGuard(tools, options)] });
    return { result: await agent.invoke(input()), calls: state.calls };
  }

  const outcome = (result: any) => result.messages.at(-1).response_metadata.toolreceipt;
  const notes = (result: any): any[] => result.messages.filter((m: any) => m._getType() === "human" && m.name === "toolreceipt");

  it("passes an honest reply untouched, with no extra call", async () => {
    const { result, calls } = await run([call("issue_refund", "c1", { amount: 32 }), "I've refunded $32."]);
    expect(calls).toBe(2);
    expect(result.messages.at(-1).content).toBe("I've refunded $32.");
    expect(notes(result)).toEqual([]);
    const out = outcome(result);
    expect([out.outcome, out.retries, out.receipt.verdict]).toEqual(["passed", 0, "clean"]);
  });

  it("fixes a lie on retry", async () => {
    const { result, calls } = await run([call("issue_refund", "c1", { amount: 32 }), "I've refunded $32 and created a ticket for you.", "I've refunded $32. I have not created a ticket."]);
    expect(calls).toBe(3);
    const [note] = notes(result);
    expect(note.content).toContain("create_ticket");
    expect(note.content).toContain("was not called");
    expect(note.content).toContain("Do not call tools");
    const out = outcome(result);
    expect([out.outcome, out.retries, out.receipt.verdict]).toEqual(["fixed", 1, "clean"]);
  });

  it("stops after one note when the agent lies twice", async () => {
    const lie = "I've refunded $32 and created a ticket for you.";
    const { result, calls } = await run([call("issue_refund", "c1", { amount: 32 }), lie, lie]);
    expect(calls).toBe(3);
    expect(notes(result)).toHaveLength(1);
    const out = outcome(result);
    expect([out.outcome, out.retries, out.receipt.verdict]).toEqual(["unresolved", 1, "fabricated"]);
  });

  it("lists every false claim", async () => {
    const { result } = await run([call("issue_refund", "c1", { amount: 32 }), "I've refunded $32, created a ticket and emailed you the confirmation.", "I've refunded $32."]);
    const [note] = notes(result);
    expect(note.content).toContain("create_ticket");
    expect(note.content).toContain("send_email");
  });

  it("names a failed tool as failed, without its error text", async () => {
    const { result } = await run([call("issue_refund", "c1", { amount: 32 }), call("send_email", "c2", { subject: "Refund" }), "I've refunded $32 and emailed you the confirmation.", "I've refunded $32, but the email could not be sent."]);
    const [note] = notes(result);
    expect(note.content).toContain("send_email");
    expect(note.content).toContain("failed");
    expect(note.content).not.toContain("secret-error-text");
    expect(note.content).not.toContain("secret-result-text");
    expect(outcome(result).outcome).toBe("fixed");
  });

  it("names the observed amount, and the run makes no second refund", async () => {
    const { result } = await run([call("issue_refund", "c1", { amount: 32 }), "I've refunded $64.", "I've refunded $32."]);
    const [note] = notes(result);
    expect(note.content).toContain("issue_refund");
    expect(note.content).toContain("32");
    expect(note.content).not.toContain("was not called");
    expect(note.content).not.toContain("secret-result-text");
    expect(refunds).toEqual([32]);
    expect(outcome(result).outcome).toBe("fixed");
  });

  it("resets retries for each user turn", async () => {
    refunds.length = 0;
    const { model } = scripted([call("issue_refund", "c1", { amount: 32 }), "I've refunded $32 and created a ticket.", "I've refunded $32.", "I've created a ticket for you.", "I have not created a ticket."]);
    const agent = langchain.createAgent({ model, tools, middleware: [receiptGuard(tools)], checkpointer: new graphModule.MemorySaver() });
    const config = { configurable: { thread_id: "t" } };
    const first = await agent.invoke(input("Refund $32."), config);
    const second = await agent.invoke(input("Open a ticket too."), config);
    expect(outcome(first).outcome).toBe("fixed");
    expect([outcome(second).outcome, outcome(second).retries]).toEqual(["fixed", 1]);
  });

  it("only records with maxRetries 0", async () => {
    const { result, calls } = await run([call("issue_refund", "c1", { amount: 32 }), "I've refunded $32 and created a ticket."], { maxRetries: 0 });
    expect(calls).toBe(2);
    expect(notes(result)).toEqual([]);
    expect(outcome(result).outcome).toBe("unresolved");
  });

  it("draws no note for a claim that matches no tool", async () => {
    const { result, calls } = await run([call("issue_refund", "c1", { amount: 32 }), "I've refunded $32 and noted your preference."]);
    expect(calls).toBe(2);
    expect(notes(result)).toEqual([]);
  });

  it("stores an outcome with no tool results", async () => {
    const big = (await import("@langchain/core/tools")).tool(() => "x".repeat(50_000) + " private-result-text", { name: "lookup_order", description: "Look up an order by id.", schema: zod.z.object({ order_id: zod.z.string() }) });
    const { model } = scripted([call("lookup_order", "c1", { order_id: "4821" }), "Your order is on its way."]);
    const agent = langchain.createAgent({ model, tools: [big], middleware: [receiptGuard([big])] });
    const out = outcome(await agent.invoke(input()));
    expect("ledger" in out.receipt).toBe(false);
    expect(JSON.stringify(out)).not.toContain("private-result-text");
    expect(JSON.stringify(out).length).toBeLessThan(2000);
  });

  it("leaves a history that reads clean after the fact", async () => {
    const { result } = await run([call("issue_refund", "c1", { amount: 32 }), "I've refunded $32 and created a ticket for you.", "I've refunded $32. I have not created a ticket."]);
    expect(check(finalReplyFromGraph(result), ledgerFromGraph(result), { knownTools: tools }).verdict).toBe("clean");
    // The OpenAI request shape, as `ChatOpenAI` sends it: the note keeps its `name`.
    const openaiShaped = result.messages.map((m: any) => {
      const kind = m._getType();
      if (kind === "human") return { role: "user", content: m.content, ...(m.name ? { name: m.name } : {}) };
      if (kind === "tool") return { role: "tool", tool_call_id: m.tool_call_id, content: m.content };
      return { role: "assistant", content: m.content || null, ...(m.tool_calls?.length ? { tool_calls: m.tool_calls.map((c: any) => ({ id: c.id, type: "function", function: { name: c.name, arguments: JSON.stringify(c.args) } })) } : {}) };
    });
    expect(checkMessages(openaiShaped, { knownTools: tools }).verdict).toBe("clean");
  });
});

describe("receiptGuard", () => {
  it("is a branded middleware object that imports nothing", () => {
    const guard = receiptGuard(["send_email"]);
    expect(guard.name).toBe("ReceiptGuard");
    expect((guard as unknown as Record<symbol, unknown>)[Symbol.for("AgentMiddleware")]).toBe(true);
    expect(guard.afterModel.canJumpTo).toEqual(["model"]);
    expect("tools" in guard).toBe(false);
  });

  it("rejects negative retries", () => {
    expect(() => receiptGuard([], { maxRetries: -1 })).toThrow(RangeError);
  });

  it("treats a note as part of the run in checkMessages", () => {
    const messages = [
      { role: "user", content: "Refund $32." },
      { role: "assistant", content: null, tool_calls: [{ id: "c1", type: "function", function: { name: "issue_refund", arguments: "{\"amount\":32}" } }] },
      { role: "tool", tool_call_id: "c1", content: "ok" },
      { role: "assistant", content: "I've refunded $32 and created a ticket." },
      { role: "user", name: "toolreceipt", content: "Your last reply says things the tool calls in this turn do not show." },
      { role: "assistant", content: "I've refunded $32." },
    ];
    expect(checkMessages(messages, { knownTools: ["issue_refund", "create_ticket"] }).verdict).toBe("clean");
  });
});
