/**
 * The LangGraph.js adapter with fake events and plain message objects, so these run on every Node the
 * package supports. Ports python/tests/test_langgraph_adapter.py; the real packages are in
 * langgraph_real.test.ts.
 */
import { describe, expect, it } from "vitest";

import {
  Ledger,
  ReceiptCallbackHandler,
  Receipts,
  ToolSpec,
  check,
  checkMessages,
  checkTurns,
  finalReplyFromGraph,
  ledgerFromGraph,
  mergeConfig,
  toolSpecsFromInvocationParams,
} from "../src/index";

const human = (content: string) => ({ type: "human", content });
const ai = (content: string, tool_calls: unknown[] = []) => ({ type: "ai", content, tool_calls });
const toolMessage = (content: unknown, tool_call_id: string, status = "success") => ({ type: "tool", content, tool_call_id, status });

const CALLS = ai("", [
  { name: "issue_refund", args: { order_id: "4821" }, id: "c1" },
  { name: "send_email", args: { to: "a@b.c" }, id: "c2" },
]);
const FINAL = ai("Refund issued and I have emailed you the confirmation.");

function start(handler: ReceiptCallbackHandler, runId: string, runName: string, input = "{}", toolCallId?: string): void {
  handler.handleToolStart({}, input, runId, undefined, undefined, undefined, runName, toolCallId);
}

describe("toolSpecsFromInvocationParams", () => {
  it("reads the OpenAI and Anthropic shapes", () => {
    const params = { tools: [
      { type: "function", function: { name: "a", description: "A" } },
      { name: "b", description: "B", input_schema: {} },
    ] };
    expect(toolSpecsFromInvocationParams(params)).toEqual([new ToolSpec("a", "A"), new ToolSpec("b", "B")]);
  });

  it("skips what has no string name, and a tools value that is not a list", () => {
    expect(toolSpecsFromInvocationParams({ tools: [null, 3, { name: 4 }, { function: {} }, { functionDeclarations: [] }] })).toEqual([]);
    expect(toolSpecsFromInvocationParams({ tools: "x" })).toEqual([]);
    expect(toolSpecsFromInvocationParams({})).toEqual([]);
    expect(toolSpecsFromInvocationParams({ tools: [{ name: "a", description: 3 }] })).toEqual([new ToolSpec("a", "")]);
  });
});

describe("ReceiptCallbackHandler", () => {
  it("waits for its handlers, so a run's calls are closed when invoke resolves", () => {
    const handler = new ReceiptCallbackHandler(new Ledger());
    expect(handler.awaitHandlers).toBe(true);
    expect(typeof handler.name).toBe("string");
  });

  it("takes the tool from runName, then serialized.name, then unknown", () => {
    const handler = new ReceiptCallbackHandler(new Ledger());
    handler.handleToolStart({ name: "from_serialized" }, "{}", "r1", undefined, undefined, undefined, "from_run_name");
    handler.handleToolEnd("x", "r1");
    handler.handleToolStart({ name: "from_serialized" }, "{}", "r2");
    handler.handleToolEnd("x", "r2");
    handler.handleToolStart({}, "{}", "r3");
    handler.handleToolEnd("x", "r3");
    expect(handler.ledger.entries.map((e) => e.tool)).toEqual(["from_run_name", "from_serialized", "unknown"]);
  });

  it("parses a JSON object input, and reads any other string as coerceArgs does", () => {
    const handler = new ReceiptCallbackHandler(new Ledger());
    start(handler, "r1", "a", '{"amount":32}');
    start(handler, "r2", "b", "plain text");
    start(handler, "r3", "c", "[1,2]");
    for (const run of ["r1", "r2", "r3"]) handler.handleToolEnd("x", run);
    expect(handler.ledger.entries.map((e) => e.args)).toEqual([{ amount: 32 }, { input: "plain text" }, { input: "[1,2]" }]);
  });

  it("uses toolCallId, falls back to runId, and keeps the parent run", () => {
    const handler = new ReceiptCallbackHandler(new Ledger());
    handler.handleToolStart({}, "{}", "r1", "parent", undefined, undefined, "a", "c1");
    handler.handleToolStart({}, "{}", "r2", undefined, undefined, undefined, "b");
    handler.handleToolEnd("x", "r1");
    handler.handleToolEnd("x", "r2");
    const [first, second] = handler.ledger.entries;
    expect([first?.callId, first?.parentRunId]).toEqual(["c1", "parent"]);
    expect([second?.callId, second?.parentRunId]).toEqual(["r2", null]);
  });

  it("reads a raw value, a ToolMessage and a failed ToolMessage", () => {
    const handler = new ReceiptCallbackHandler(new Ledger());
    start(handler, "r1", "a");
    handler.handleToolEnd("done", "r1");
    start(handler, "r2", "b");
    handler.handleToolEnd(toolMessage("sent", "c2"), "r2");
    start(handler, "r3", "c");
    handler.handleToolEnd(toolMessage("denied", "c3", "error"), "r3");
    const entries = handler.ledger.entries;
    expect(entries.map((e) => [e.status, e.result, e.error, e.callId])).toEqual([
      ["ok", "done", null, "r1"],
      ["ok", "sent", null, "c2"],
      ["error", null, "denied", "c3"],
    ]);
  });

  it("picks a Command's matching ToolMessage, from a record and from a tuple list", () => {
    const other = toolMessage("fine", "c9");
    const failed = toolMessage("SMTP gateway timeout", "c2", "error");
    for (const update of [{ messages: [other, failed] }, [["messages", [other, failed]]]]) {
      const handler = new ReceiptCallbackHandler(new Ledger());
      start(handler, "r1", "send_email", "{}", "c2");
      handler.handleToolEnd({ update, lg_name: "Command" }, "r1");
      const [entry] = handler.ledger.entries;
      expect([entry?.status, entry?.error, entry?.callId]).toEqual(["error", "SMTP gateway timeout", "c2"]);
    }
  });

  it("falls back to a Command's first ToolMessage, and takes one message that is not in a list", () => {
    const handler = new ReceiptCallbackHandler(new Ledger());
    start(handler, "r1", "a", "{}", "c1");
    handler.handleToolEnd({ update: { messages: [toolMessage("first", "x1"), toolMessage("second", "x2")] } }, "r1");
    start(handler, "r2", "b", "{}", "c2");
    handler.handleToolEnd({ update: { messages: toolMessage("no", "c2", "error") } }, "r2");
    expect(handler.ledger.entries.map((e) => [e.status, e.callId])).toEqual([["ok", "x1"], ["error", "c2"]]);
  });

  it("records a thrown error, and removes a paused call", () => {
    const handler = new ReceiptCallbackHandler(new Ledger());
    start(handler, "r1", "a");
    handler.handleToolError(new Error("boom"), "r1");
    expect(handler.ledger.entries[0]?.error).toBe("Error: boom");

    const interrupts = [
      Object.assign(new Error("x"), { name: "GraphInterrupt" }),
      Object.assign(new Error("x"), { name: "NodeInterrupt" }),
      Object.assign(new Error("x"), { name: "GraphDrained" }),
      Object.assign(new Error("x"), { name: "SomeFutureBubble", is_bubble_up: true }),
    ];
    for (const [i, error] of interrupts.entries()) {
      start(handler, `p${i}`, "paused");
      handler.handleToolError(error, `p${i}`);
    }
    handler.flush();
    expect(handler.ledger.entries.map((e) => e.tool)).toEqual(["a"]);
  });

  it("keeps an abort as a tool error", () => {
    const handler = new ReceiptCallbackHandler(new Ledger());
    start(handler, "r1", "slow");
    handler.handleToolError(Object.assign(new Error("aborted"), { name: "AbortError" }), "r1");
    expect(handler.ledger.entries[0]?.status).toBe("error");
  });

  it("with results off keeps args, results and a ToolMessage's failure text out", () => {
    const handler = new ReceiptCallbackHandler(new Ledger(), { captureResults: false });
    start(handler, "r1", "a", '{"secret":1}');
    handler.handleToolEnd("private", "r1");
    start(handler, "r2", "b", '{"secret":1}');
    handler.handleToolEnd(toolMessage("private failure", "c2", "error"), "r2");
    const [ok, failed] = handler.ledger.entries;
    expect([ok?.args, ok?.result, ok?.status]).toEqual([{}, null, "ok"]);
    expect([failed?.args, failed?.result, failed?.status, failed?.error]).toEqual([{}, null, "error", "tool reported an error"]);
  });

  it("truncates to maxResultChars", () => {
    const handler = new ReceiptCallbackHandler(new Ledger(), { maxResultChars: 3 });
    start(handler, "r1", "a");
    handler.handleToolEnd("abcdef", "r1");
    expect(handler.ledger.entries[0]?.result).toBe("ab…");
  });

  it("collects known tools; a non-empty description is never overridden", () => {
    const handler = new ReceiptCallbackHandler(new Ledger());
    handler.handleChatModelStart({}, [], "m1", undefined, { invocation_params: { tools: [{ name: "a", description: "A" }] } });
    handler.handleChatModelStart({}, [], "m2", undefined, { invocation_params: { tools: [{ name: "a", description: "" }, { name: "b" }] } });
    start(handler, "r1", "a");
    start(handler, "r2", "c");
    expect([...handler.knownTools]).toEqual([["a", "A"], ["b", ""], ["c", ""]]);
  });

  it("keeps 50 interleaved calls apart, and flush closes the open half", () => {
    const ledger = new Ledger();
    const handler = new ReceiptCallbackHandler(ledger);
    for (let i = 0; i < 50; i++) start(handler, `r${i}`, `t${i % 3}`, JSON.stringify({ i }), `c${i}`);
    for (let i = 1; i < 50; i += 2) handler.handleToolEnd(`ok${i}`, `r${i}`);
    handler.flush();
    expect(ledger.length).toBe(50);
    expect(ledger.entries.filter((e) => e.status === "error" && e.error === "no result")).toHaveLength(25);
  });

  it("opens an entry for input JSON cannot write", () => {
    const handler = new ReceiptCallbackHandler(new Ledger());
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    handler.handleToolStart({}, circular, "r1", undefined, undefined, undefined, "create_ticket");
    handler.handleToolEnd("TCK-1", "r1");
    const [entry] = handler.ledger.entries;
    expect([entry?.tool, entry?.status, entry?.result]).toEqual(["create_ticket", "ok", "TCK-1"]);
  });

  it("never throws, and still records", () => {
    const handler = new ReceiptCallbackHandler(new Ledger());
    const broken = new Proxy({}, { get() { throw new Error("broken mapping"); }, has() { throw new Error("broken"); } });
    handler.handleChatModelStart({}, [], "m", undefined, broken);
    handler.handleChatModelStart(broken, [], "m2", undefined, { invocation_params: broken });
    handler.handleToolStart(broken, "", "bad_start");
    handler.handleToolEnd("done", "bad_start");
    start(handler, "bad_end", "send_email");
    handler.handleToolEnd(broken, "bad_end");
    start(handler, "bad_error", "issue_refund");
    handler.handleToolError(broken, "bad_error");
    handler.handleToolEnd("late", "never_opened");
    handler.handleToolError(new Error("late"), "never_opened");
    const entries = new Map(handler.ledger.entries.map((e) => [e.tool, e]));
    expect([...entries.keys()]).toEqual(["unknown", "send_email", "issue_refund"]);
    expect(entries.get("unknown")?.status).toBe("ok");
    expect(entries.get("send_email")?.status).toBe("ok");
    expect(entries.get("issue_refund")?.status).toBe("error");
  });
});

describe("ReceiptCallbackHandler chain events", () => {
  const call = { name: "send_email", args: { to: "a" }, type: "tool_call", id: "c2" };
  const chainStart = (h: ReceiptCallbackHandler, run: string, inputs: unknown, runName?: string) => {
    h.handleChainStart({}, inputs, run, "parent", undefined, undefined, undefined, runName);
  };

  it("opens a call only for a tool call whose run bears its name", () => {
    const handler = new ReceiptCallbackHandler(new Ledger());
    chainStart(handler, "other", { messages: [] }, "agent");
    chainStart(handler, "parse", call, "send_email:parse_input");
    chainStart(handler, "named", { ...call, id: "c3" });
    handler.flush();
    expect(handler.ledger.length).toBe(0);
    chainStart(handler, "run", call, "send_email");
    handler.handleChainEnd({ output: "sent" }, "run");
    const [entry] = handler.ledger.entries;
    expect([entry?.tool, entry?.args, entry?.callId, entry?.result, entry?.parentRunId]).toEqual(["send_email", { to: "a" }, "c2", "sent", "parent"]);
  });

  it("ignores the end and error of a chain it did not open", () => {
    const handler = new ReceiptCallbackHandler(new Ledger());
    handler.handleChainEnd({ output: "x" }, "unknown");
    handler.handleChainError(new Error("x"), "unknown");
    expect(handler.ledger.length).toBe(0);
  });

  it("leaves a real tool's own event to record the call", () => {
    const handler = new ReceiptCallbackHandler(new Ledger());
    chainStart(handler, "binding", call, "send_email");
    handler.handleToolStart({}, '{"to":"a"}', "tool", "binding", undefined, undefined, "send_email", "c2");
    handler.handleToolEnd("sent", "tool");
    handler.handleChainEnd({ output: "sent" }, "binding");
    expect(handler.ledger.entries.map((e) => e.callId)).toEqual(["c2"]);
  });

  it("reads a failure and a ToolMessage, and removes a pause", () => {
    const handler = new ReceiptCallbackHandler(new Ledger());
    chainStart(handler, "a", { ...call, id: "a" }, "send_email");
    handler.handleChainError(new Error("boom"), "a");
    chainStart(handler, "b", { ...call, id: "b" }, "send_email");
    handler.handleChainEnd(toolMessage("denied", "b", "error"), "b");
    chainStart(handler, "c", { ...call, id: "c" }, "send_email");
    handler.handleChainError(Object.assign(new Error("x"), { name: "GraphInterrupt" }), "c");
    handler.flush();
    expect(handler.ledger.entries.map((e) => [e.callId, e.status, e.error])).toEqual([["a", "error", "Error: boom"], ["b", "error", "denied"]]);
  });

  it("never throws on a broken payload", () => {
    const handler = new ReceiptCallbackHandler(new Ledger());
    const broken = new Proxy({}, { get() { throw new Error("broken"); } });
    chainStart(handler, "x", broken, "send_email");
    chainStart(handler, "run", call, "send_email");
    handler.handleChainEnd(broken, "run");
    expect(handler.ledger.entries.map((e) => [e.status, e.result])).toEqual([["ok", null]]);
  });
});

describe("ReceiptCallbackHandler tool names", () => {
  it("prefers the ToolMessage's name, and drops a run label from the known tools", () => {
    const handler = new ReceiptCallbackHandler(new Ledger());
    start(handler, "r1", "tool-step", "{}", "c1");
    handler.handleToolEnd({ ...toolMessage("sent", "c1"), name: "send_email" }, "r1");
    expect(handler.ledger.entries[0]?.tool).toBe("send_email");
    expect([...handler.knownTools.keys()]).toEqual(["send_email"]);
  });

  it("keeps a known tool's description, and the call id when the message's is empty", () => {
    const handler = new ReceiptCallbackHandler(new Ledger());
    handler.handleChatModelStart({}, [], "m", undefined, { invocation_params: { tools: [{ name: "send_email", description: "Send" }] } });
    start(handler, "r1", "send_email", "{}", "c1");
    handler.handleToolEnd({ ...toolMessage("sent", ""), name: "send_email" }, "r1");
    expect(handler.ledger.entries[0]?.callId).toBe("c1");
    expect([...handler.knownTools]).toEqual([["send_email", "Send"]]);
  });
});

describe("mergeConfig", () => {
  class Manager {
    handlers: unknown[] = [];
    inherited: unknown[] = [];
    copied = false;
    copy(): Manager {
      const next = new Manager();
      next.handlers = [...this.handlers];
      next.copied = true;
      return next;
    }
    addHandler(handler: unknown, inherit: boolean): void {
      this.handlers.push(handler);
      if (inherit) this.inherited.push(handler);
    }
  }

  it("handles no callbacks, a list and a manager, and keeps other keys", () => {
    const h = new ReceiptCallbackHandler(new Ledger());
    const other = new ReceiptCallbackHandler(new Ledger());
    expect(mergeConfig(null, h).callbacks).toEqual([h]);
    expect(mergeConfig(undefined, h).callbacks).toEqual([h]);

    const list = [other];
    const merged = mergeConfig({ callbacks: list, configurable: { thread_id: "t1" } }, h);
    expect(merged.callbacks).toEqual([other, h]);
    expect(merged.configurable).toEqual({ thread_id: "t1" });
    expect(list).toEqual([other]);

    const manager = new Manager();
    const managed = mergeConfig({ callbacks: manager, configurable: { thread_id: "t1" } }, h);
    expect((managed.callbacks).copied).toBe(true);
    expect((managed.callbacks).inherited).toEqual([h]);
    expect(manager.handlers).toEqual([]);
    expect(managed.configurable).toEqual({ thread_id: "t1" });
  });

  it("rejects anything else", () => {
    const h = new ReceiptCallbackHandler(new Ledger());
    expect(() => mergeConfig({ callbacks: "nope" }, h)).toThrow(TypeError);
  });
});

describe("Receipts.config", () => {
  const events = (config: { callbacks: ReceiptCallbackHandler[] }, reply: { tool: string; toolCallId: string }) => {
    const handler = config.callbacks.at(-1) as ReceiptCallbackHandler;
    start(handler, `run-${reply.toolCallId}`, reply.tool, "{}", reply.toolCallId);
    handler.handleToolEnd("done", `run-${reply.toolCallId}`);
  };

  it("reads the newest run with one config, and a given config with more", () => {
    const receipts = new Receipts();
    const first = receipts.config();
    events(first, { tool: "issue_refund", toolCallId: "c1" });
    expect(receipts.ledger?.entries.map((e) => e.tool)).toEqual(["issue_refund"]);
    expect(receipts.check("I've refunded you.").verdict).toBe("clean");

    const second = receipts.config();
    events(second, { tool: "send_email", toolCallId: "c2" });
    expect(() => receipts.check("I've refunded you.")).toThrow(/more than one run/);
    expect(receipts.check("I've refunded you.", first).verdict).toBe("clean");
    expect(receipts.check("I've refunded you.", second).verdict).toBe("unreported");
  });

  it("finds the handler in a spread config", () => {
    const receipts = new Receipts();
    const config = receipts.config();
    receipts.config();
    events(config, { tool: "issue_refund", toolCallId: "c1" });
    expect(receipts.check("I've refunded you.", { ...config, configurable: { thread_id: "t" } }).verdict).toBe("clean");
  });

  it("rejects a config with no handler, more than one, or another instance's", () => {
    const receipts = new Receipts();
    const mine = receipts.config();
    const theirs = new Receipts().config();
    expect(() => receipts.check("x", { callbacks: [] })).toThrow(/no ReceiptCallbackHandler/);
    expect(() => receipts.check("x", { callbacks: [...mine.callbacks, ...theirs.callbacks] })).toThrow(/more than one/);
    expect(() => receipts.check("x", theirs)).toThrow(/another Receipts/);
    expect(() => receipts.check("x", {})).toThrow(TypeError);
  });

  it("finds the handler in a manager", () => {
    const receipts = new Receipts();
    const config = receipts.config();
    const handler = config.callbacks[0];
    expect(receipts.check("Hello", { callbacks: { handlers: [handler], inheritableHandlers: [handler] } }).verdict).toBe("clean");
  });

  it("flushes an open call, and hides args with results off", () => {
    const receipts = new Receipts({ captureResults: false });
    const config = receipts.config();
    start(config.callbacks[0] as ReceiptCallbackHandler, "r1", "issue_refund", '{"a":1}', "c1");
    const receipt = receipts.check("I've refunded you.", config);
    expect(receipt.ledger.entries[0]?.args).toEqual({});
    expect(receipt.ledger.entries[0]?.error).toBe("no result");
    expect(receipt.verdict).toBe("fabricated");
  });

  it("names discovered tools with their descriptions, and accepts acheck", async () => {
    const receipts = new Receipts();
    const config = receipts.config();
    (config.callbacks[0] as ReceiptCallbackHandler).handleChatModelStart({}, [], "m", undefined, {
      invocation_params: { tools: [{ type: "function", function: { name: "create_ticket", description: "Create a ticket" } }] },
    });
    const receipt = await receipts.acheck("I created a ticket for you.", config);
    expect(receipt.verdict).toBe("fabricated");
    expect(receipt.claimedButNeverRan[0]?.reason).toBe("not_called");
    expect(receipt.knownTools).toContain("create_ticket");
  });

  it("counts wrapTools and config as runs together", () => {
    const receipts = new Receipts();
    receipts.config();
    receipts.wrapTools({});
    expect(() => receipts.check("x")).toThrow(/more than one run/);
  });

  it("keeps a wrapTools record working as a source", () => {
    const receipts = new Receipts();
    const tools = receipts.wrapTools({ callbacks: { execute: () => "x" } });
    expect(receipts.check("Hello", tools).verdict).toBe("clean");
  });
});

describe("ledgerFromGraph", () => {
  it("pairs calls with results, and an unpaired call has no result", () => {
    const ledger = ledgerFromGraph([human("x"), CALLS, toolMessage("refunded", "c1"), FINAL]);
    const byTool = new Map(ledger.entries.map((e) => [e.tool, e]));
    expect(byTool.get("issue_refund")?.status).toBe("ok");
    expect(byTool.get("issue_refund")?.callId).toBe("c1");
    expect(byTool.get("issue_refund")?.args).toEqual({ order_id: "4821" });
    expect(byTool.get("issue_refund")?.startedAt).toBeNull();
    expect([byTool.get("send_email")?.status, byTool.get("send_email")?.error]).toEqual(["error", "no result"]);
  });

  it("flattens list content, and takes a result with messages", () => {
    const messages = [CALLS, toolMessage([{ type: "text", text: "refunded" }], "c1")];
    expect(ledgerFromGraph(messages).entries[0]?.result).toBe("refunded");
    expect(ledgerFromGraph({ messages }).entries[0]?.result).toBe("refunded");
  });

  it("records a failed result", () => {
    const [entry] = ledgerFromGraph([ai("", [{ name: "a", args: {}, id: "c1" }]), toolMessage("denied", "c1", "error")]).entries;
    expect([entry?.status, entry?.error, entry?.result]).toEqual(["error", "denied", null]);
  });

  it("reads only the turn after the last human message", () => {
    const thread = [
      human("refund me"), ai("", [{ name: "issue_refund", args: {}, id: "c1" }]), toolMessage("ok", "c1"), ai("Done."),
      human("again"), ai("I refunded you again."),
    ];
    expect(ledgerFromGraph(thread).length).toBe(0);
    expect(finalReplyFromGraph(thread)).toBe("I refunded you again.");
  });

  it("reads `_getType` messages", () => {
    const message = { _getType: () => "ai", content: "", tool_calls: [{ name: "a", args: {}, id: "c1" }] };
    expect(ledgerFromGraph([message]).entries[0]?.tool).toBe("a");
  });

  it("throws on what it cannot read, rather than give an empty ledger", () => {
    for (const read of [ledgerFromGraph, finalReplyFromGraph]) {
      expect(() => read([{ role: "assistant", content: "x" }])).toThrow(/checkMessages/);
      expect(() => read([{ lc: 1, type: "constructor", id: ["langchain_core", "messages", "AIMessage"], kwargs: { content: "x" } }])).toThrow(/mapStoredMessagesToChatMessages/);
      expect(() => read([{ type: "ai", data: { content: "x", tool_calls: [] } }])).toThrow(/mapStoredMessagesToChatMessages/);
      expect(() => read({ state: [] })).toThrow(/messages/);
      expect(() => read("nope")).toThrow(TypeError);
      expect(() => read(null)).toThrow(TypeError);
    }
  });
});

describe("finalReplyFromGraph", () => {
  it("handles the Python edge cases", () => {
    expect(finalReplyFromGraph({ messages: [human("x"), CALLS, FINAL] })).toBe(FINAL.content);
    expect(finalReplyFromGraph([human("x"), CALLS])).toBe("");
    expect(finalReplyFromGraph({ messages: [] })).toBe("");
    expect(finalReplyFromGraph([human("x"), ai("I've issued your refund.", [{ name: "issue_refund", args: {}, id: "c9" }])])).toBe("");
    const result = toolMessage("ok", "c1");
    expect(finalReplyFromGraph([human("x"), FINAL, CALLS, result])).toBe("");
    expect(finalReplyFromGraph([human("x"), CALLS, result, FINAL])).toBe(FINAL.content);
  });

  it("joins text blocks as checkMessages does", () => {
    const blocks = [{ type: "text", text: "I have not looked yet " }, { type: "text", text: "\u2705 I've refunded your order." }];
    expect(finalReplyFromGraph([{ type: "ai", content: blocks, text: blocks.map((b) => b.text).join("") }])).toBe("I have not looked yet \n\u2705 I've refunded your order.");
  });

  it("reads bare strings in a content list, so no claim or negation is lost", () => {
    expect(finalReplyFromGraph([human("x"), { type: "ai", content: ["I refunded you."] }])).toBe("I refunded you.");
    const mixed = { type: "ai", content: ["I have not ", { type: "text", text: "refunded you yet." }] };
    expect(finalReplyFromGraph([human("x"), mixed])).toBe("I have not refunded you yet.");
    const lone = finalReplyFromGraph([human("x"), { type: "ai", content: ["I refunded you."] }]);
    expect(check(lone, new Ledger(), { knownTools: ["issue_refund"] }).verdict).toBe("fabricated");
    expect(check(finalReplyFromGraph([human("x"), mixed]), new Ledger(), { knownTools: ["issue_refund"] }).verdict).toBe("clean");
  });

  it("skips an empty AI message", () => {
    expect(finalReplyFromGraph([human("x"), FINAL, ai("  ")])).toBe(FINAL.content);
  });
});

describe("checkMessages on LangChain shapes", () => {
  const serialized = [
    { lc: 1, type: "constructor", id: ["langchain_core", "messages", "AIMessage"], kwargs: { content: "I refunded you." } },
    { type: "ai", data: { content: "I refunded you." } },
  ];

  it("throws on serialized messages, naming how to revive them", () => {
    for (const message of serialized) {
      expect(() => checkMessages([message], { knownTools: ["issue_refund"] })).toThrow(/mapStoredMessagesToChatMessages/);
      expect(() => checkTurns([message], { knownTools: ["issue_refund"] })).toThrow(/mapStoredMessagesToChatMessages/);
    }
  });

  it("names the adapter in the LangChain error", () => {
    expect(() => checkMessages([ai("x")], {})).toThrow(/ledgerFromGraph/);
  });
});
