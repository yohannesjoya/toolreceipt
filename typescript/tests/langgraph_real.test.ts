/**
 * The real LangChain.js packages over a stubbed `fetch`: no network, no key. These pin the event and
 * message shapes the adapter depends on, so a new release that moves them fails here. Each case differs
 * from the Python handler, or is a place the live and post-hoc paths part (PRD 11 §5.2).
 */
/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-return */
import { beforeAll, describe, expect, it } from "vitest";

import { Ledger, ReceiptCallbackHandler, Receipts, check, checkMessages, finalReplyFromGraph, ledgerFromGraph } from "../src/index";

const nodeMajor = Number(process.versions.node.split(".")[0]);

interface Call { name: string; id: string; args?: unknown }
type Step = string | Call[];

describe.skipIf(nodeMajor < 20)("real LangChain.js", () => {
  let core: any, toolsModule: any, manager: any, graphModule: any, prebuilt: any, langchain: any, openai: any, anthropic: any, zod: any;
  let refund: any, email: any, ticket: any;

  beforeAll(async () => {
    core = await import("@langchain/core/messages");
    toolsModule = await import("@langchain/core/tools");
    manager = await import("@langchain/core/callbacks/manager");
    graphModule = await import("@langchain/langgraph");
    prebuilt = await import("@langchain/langgraph/prebuilt");
    langchain = await import("langchain");
    openai = await import("@langchain/openai");
    anthropic = await import("@langchain/anthropic");
    zod = await import("zod");
    const { tool } = toolsModule;
    const { z } = zod;
    refund = tool(({ amount }: { amount: number }) => `refunded ${amount}`, { name: "issue_refund", description: "Issue a refund", schema: z.object({ amount: z.number() }) });
    email = tool(() => "sent", { name: "send_email", description: "Send an email", schema: z.object({}) });
    ticket = tool(() => "TCK-1", { name: "create_ticket", description: "Create a support ticket", schema: z.object({}) });
  });

  /** A `fetch` that plays OpenAI chat responses. Each has its own id: `messagesStateReducer` replaces a
   * message with the same id, so a shared one drops the tool call from the state. */
  function chatFetch(steps: Step[]) {
    let i = 0;
    return (): Promise<Response> => {
      const step = steps[i] ?? "";
      i++;
      const message = typeof step === "string"
        ? { role: "assistant", content: step }
        : { role: "assistant", content: null, tool_calls: step.map((c) => ({ id: c.id, type: "function", function: { name: c.name, arguments: JSON.stringify(c.args ?? {}) } })) };
      const body = { id: `chatcmpl-${i}`, object: "chat.completion", created: 1, model: "m", choices: [{ index: 0, message, finish_reason: typeof step === "string" ? "stop" : "tool_calls" }], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } };
      return Promise.resolve(new Response(JSON.stringify(body), { headers: { "content-type": "application/json" } }));
    };
  }

  const model = (steps: Step[]) => new openai.ChatOpenAI({ apiKey: "test", model: "gpt-4o-mini", configuration: { fetch: chatFetch(steps) } });
  const agent = (steps: Step[], tools: unknown[], extra: object = {}) => langchain.createAgent({ model: model(steps), tools, ...extra });
  const input = (text = "hi") => ({ messages: [new core.HumanMessage(text)] });

  /** `StateGraph` with `ToolNode`, for what `createAgent` does not expose. */
  function graph(steps: Step[], tools: unknown[], nodeOptions?: object, compile: object = {}) {
    const bound = model(steps).bindTools(tools);
    const { StateGraph, MessagesAnnotation, START, END } = graphModule;
    return new StateGraph(MessagesAnnotation)
      .addNode("agent", async (state: any, config: any) => ({ messages: [await bound.invoke(state.messages, config)] }))
      .addNode("tools", new prebuilt.ToolNode(tools, nodeOptions))
      .addEdge(START, "agent")
      .addConditionalEdges("agent", (state: any) => (state.messages.at(-1).tool_calls?.length ? "tools" : END))
      .addEdge("tools", "agent")
      .compile(compile);
  }

  const refundCall: Call = { name: "issue_refund", id: "c1", args: { amount: 32 } };

  describe("createAgent", () => {
    it("reads an honest reply clean on both paths", async () => {
      const receipts = new Receipts();
      const config = receipts.config();
      const result = await agent([[refundCall], "I've refunded $32."], [refund, email]).invoke(input(), config);
      const reply = finalReplyFromGraph(result);
      expect(reply).toBe("I've refunded $32.");

      const live = receipts.check(reply, config);
      expect(live.verdict).toBe("clean");
      const [entry] = live.ledger.entries;
      expect([entry?.tool, entry?.callId, entry?.status, entry?.result]).toEqual(["issue_refund", "c1", "ok", "refunded 32"]);
      expect(entry?.args).toEqual({ amount: 32 });
      expect(entry?.parentRunId).toEqual(expect.any(String));

      const posthoc = check(reply, ledgerFromGraph(result), { knownTools: ["issue_refund", "send_email"] });
      expect(posthoc.verdict).toBe("clean");
      expect(posthoc.ledger.entries[0]?.args).toEqual({ amount: 32 });
    });

    it("reads a false reply fabricated on both paths, with bound tools discovered live", async () => {
      const receipts = new Receipts();
      const config = receipts.config();
      const result = await agent([[refundCall], "I've refunded $32 and emailed you the confirmation."], [refund, email]).invoke(input(), config);
      const reply = finalReplyFromGraph(result);

      const live = receipts.check(reply, config);
      expect(live.verdict).toBe("fabricated");
      expect(live.claimedButNeverRan[0]?.tool).toBe("send_email");
      expect(live.claimedButNeverRan[0]?.reason).toBe("not_called");
      expect(live.knownTools).toEqual(expect.arrayContaining(["issue_refund", "send_email"]));

      const posthoc = check(reply, ledgerFromGraph(result), { knownTools: ["issue_refund", "send_email"] });
      expect(posthoc.verdict).toBe("fabricated");
      expect(posthoc.summary).toBe(live.summary);
    });

    it("parses the input, so the argument check sees the amount", async () => {
      const run = async (reply: string) => {
        const receipts = new Receipts();
        const config = receipts.config();
        const result = await agent([[refundCall], reply], [refund]).invoke(input(), config);
        return receipts.check(finalReplyFromGraph(result), config);
      };
      expect((await run("I've refunded $32.")).verdict).toBe("clean");
      const wrong = await run("I've refunded $64.");
      expect(wrong.verdict).toBe("fabricated");
      expect(wrong.claimedButNeverRan[0]?.reason).toBe("args_mismatch");
    });

    it("waits for the handler: a slow end event is closed when invoke resolves", async () => {
      class Slow extends ReceiptCallbackHandler {
        override handleToolEnd(output: unknown, runId: string): void {
          void output;
          void runId;
          return new Promise((resolve) => setTimeout(resolve, 30)).then(() => {
            super.handleToolEnd(output, runId);
          }) as unknown as void;
        }
      }
      const handler = new Slow(new Ledger());
      await agent([[refundCall], "Done."], [refund]).invoke(input(), { callbacks: [handler] });
      expect(handler.ledger.entries.map((e) => e.status)).toEqual(["ok"]);
    });

    it("records parallel calls, and stream", async () => {
      const calls: Step[] = [[{ name: "issue_refund", id: "p1", args: { amount: 1 } }, { name: "issue_refund", id: "p2", args: { amount: 2 } }], "Done."];
      const receipts = new Receipts();
      const config = receipts.config();
      await agent(calls, [refund]).invoke(input(), config);
      expect(receipts.ledger?.entries.map((e) => e.callId).sort()).toEqual(["p1", "p2"]);
      expect(receipts.ledger?.entries.find((e) => e.callId === "p2")?.args).toEqual({ amount: 2 });

      const streaming = new Receipts();
      for await (const _ of await agent([[refundCall], "Done."], [refund]).stream(input(), streaming.config())) void _;
      expect(streaming.ledger?.entries.map((e) => e.callId)).toEqual(["c1"]);
    });

    it("works with a caller's CallbackManager, which stays unchanged", async () => {
      const mine = new manager.CallbackManager();
      const receipts = new Receipts();
      const config = receipts.config({ callbacks: mine, tags: ["t"] });
      expect(mine.handlers).toHaveLength(0);
      await agent([[refundCall], "Done."], [refund]).invoke(input(), config);
      expect(receipts.check("Done.", config).ledger.entries[0]?.callId).toBe("c1");
      expect(config.tags).toEqual(["t"]);
    });

    it("keeps concurrent runs apart on one instance, and a spread config finds its run", async () => {
      const receipts = new Receipts();
      const first = receipts.config();
      const second = receipts.config();
      await Promise.all([
        agent([[{ name: "issue_refund", id: "a", args: { amount: 1 } }], "Done."], [refund]).invoke(input(), first),
        agent([[{ name: "send_email", id: "b" }], "Done."], [email]).invoke(input(), { ...second, configurable: { thread_id: "t" } }),
      ]);
      expect(receipts.check("Done.", first).ledger.entries.map((e) => e.callId)).toEqual(["a"]);
      expect(receipts.check("Done.", { ...second, configurable: { thread_id: "t" } }).ledger.entries.map((e) => e.callId)).toEqual(["b"]);
    });

    it("reads a stubbed ChatAnthropic run's tools", async () => {
      const fetchStub = () => {
        let n = 0;
        return () => {
          n++;
          const content = n === 1 ? [{ type: "tool_use", id: "a1", name: "issue_refund", input: { amount: 32 } }] : [{ type: "text", text: "I've refunded $32." }];
          return Promise.resolve(new Response(JSON.stringify({ id: `msg_${n}`, type: "message", role: "assistant", model: "m", content, stop_reason: n === 1 ? "tool_use" : "end_turn", stop_sequence: null, usage: { input_tokens: 1, output_tokens: 1 } }), { headers: { "content-type": "application/json" } }));
        };
      };
      const receipts = new Receipts();
      const config = receipts.config();
      const chat = new anthropic.ChatAnthropic({ apiKey: "test", model: "claude-x", streaming: false, clientOptions: { fetch: fetchStub() } });
      const result = await langchain.createAgent({ model: chat, tools: [refund, email] }).invoke(input(), config);
      const receipt = receipts.check(finalReplyFromGraph(result), config);
      expect(receipt.verdict).toBe("clean");
      expect(receipt.knownTools).toEqual(expect.arrayContaining(["issue_refund", "send_email"]));
      expect(receipt.ledger.entries[0]?.args).toEqual({ amount: 32 });
    });

    it("does not read structured output, and post-hoc shows the synthetic tool", async () => {
      const { toolStrategy } = langchain;
      const responseFormat = toolStrategy(zod.z.object({ answer: zod.z.string() }));
      const name = responseFormat[0]?.name ?? responseFormat.name;
      const receipts = new Receipts();
      const config = receipts.config();
      const result = await agent([[{ name, id: "s1", args: { answer: "x" } }]], [refund], { responseFormat }).invoke(input(), config);
      expect(result.structuredResponse).toEqual({ answer: "x" });
      expect(receipts.ledger?.length).toBe(0);
      expect(ledgerFromGraph(result).entries.map((e) => e.tool)).toEqual([name]);
    });

    it("parts the paths when middleware answers instead of the tool", async () => {
      const { createMiddleware } = langchain;
      const block = createMiddleware({
        name: "block",
        wrapToolCall: (request: any) => new core.ToolMessage({ content: "ok", tool_call_id: request.toolCall.id }),
      });
      const receipts = new Receipts();
      const config = receipts.config();
      const result = await agent([[refundCall], "I refunded you."], [refund], { middleware: [block] }).invoke(input(), config);
      const reply = finalReplyFromGraph(result);
      expect(receipts.check(reply, config).verdict).toBe("fabricated");
      expect(check(reply, ledgerFromGraph(result), { knownTools: ["issue_refund"] }).verdict).toBe("clean");
    });
  });

  describe("StateGraph", () => {
    const boom = () => toolsModule.tool(() => { throw new Error("boom"); }, { name: "send_email", description: "Send an email", schema: zod.z.object({}) });

    it("records a tool error when ToolNode catches it, the default", async () => {
      const receipts = new Receipts();
      const config = receipts.config();
      const result = await graph([[{ name: "send_email", id: "c2" }], "I've emailed you."], [boom()]).invoke(input(), config);
      const live = receipts.check(finalReplyFromGraph(result), config);
      expect(live.verdict).toBe("fabricated");
      expect(live.claimedButNeverRan[0]?.reason).toBe("tool_failed");
      expect(live.ledger.entries[0]?.error).toContain("boom");
      expect(check(finalReplyFromGraph(result), ledgerFromGraph(result)).verdict).toBe("fabricated");
    });

    it("records a tool error when ToolNode lets it through", async () => {
      const receipts = new Receipts();
      const config = receipts.config();
      await expect(graph([[{ name: "send_email", id: "c2" }]], [boom()], { handleToolErrors: false }).invoke(input(), config)).rejects.toThrow("boom");
      expect(receipts.ledger?.entries[0]?.status).toBe("error");
    });

    it("reads a returned failure, a Command and content_and_artifact", async () => {
      const { tool } = toolsModule;
      const { z } = zod;
      const denied = tool((_: unknown, cfg: any) => new core.ToolMessage({ content: "denied", status: "error", tool_call_id: cfg.toolCall.id }), { name: "denied", description: "x", schema: z.object({}) });
      const command = tool((_: unknown, cfg: any) => new graphModule.Command({ update: { messages: [new core.ToolMessage({ content: "via command", tool_call_id: cfg.toolCall.id })] } }), { name: "command", description: "x", schema: z.object({}) });
      const tuples = tool((_: unknown, cfg: any) => new graphModule.Command({ update: [["messages", [new core.ToolMessage({ content: "bad", status: "error", tool_call_id: cfg.toolCall.id })]]] }), { name: "tuples", description: "x", schema: z.object({}) });
      const artifact = tool(() => ["text", { a: 1 }], { name: "artifact", description: "x", schema: z.object({}), responseFormat: "content_and_artifact" });
      const steps: Step[] = [[{ name: "denied", id: "d" }], [{ name: "command", id: "c" }], [{ name: "tuples", id: "t" }], [{ name: "artifact", id: "a" }], "Done."];
      const receipts = new Receipts();
      await graph(steps, [denied, command, tuples, artifact]).invoke(input(), receipts.config());
      const seen = new Map(receipts.ledger?.entries.map((e) => [e.tool, [e.status, e.result ?? e.error, e.callId]]));
      expect(seen.get("denied")).toEqual(["error", "denied", "d"]);
      expect(seen.get("command")).toEqual(["ok", "via command", "c"]);
      expect(seen.get("tuples")).toEqual(["error", "bad", "t"]);
      expect(seen.get("artifact")).toEqual(["ok", "text", "a"]);
    });

    it("records a subgraph's calls", async () => {
      const { StateGraph, MessagesAnnotation, START, END } = graphModule;
      const outer = new StateGraph(MessagesAnnotation).addNode("sub", graph([[refundCall], "Done."], [refund])).addEdge(START, "sub").addEdge("sub", END).compile();
      const receipts = new Receipts();
      await outer.invoke(input(), receipts.config());
      expect(receipts.ledger?.entries.map((e) => e.callId)).toEqual(["c1"]);
    });

    it("keeps an abort as a tool error", async () => {
      const slow = toolsModule.tool((_: unknown, cfg: any) => new Promise((_resolve, reject) => {
        cfg.signal?.addEventListener("abort", () => { reject(new Error("aborted")); });
      }), { name: "slow", description: "x", schema: zod.z.object({}) });
      const controller = new AbortController();
      setTimeout(() => { controller.abort(); }, 30);
      const receipts = new Receipts();
      const config = receipts.config({ signal: controller.signal });
      await expect(graph([[{ name: "slow", id: "s" }]], [slow], { handleToolErrors: false }).invoke(input(), config)).rejects.toThrow();
      expect(receipts.check("Hello", config).ledger.entries[0]?.status).toBe("error");
    });

    function interrupting() {
      const { tool } = toolsModule;
      const sendEmail = tool(() => { const answer = graphModule.interrupt("ok?"); return `sent ${answer}`; }, { name: "send_email", description: "Send an email", schema: zod.z.object({}) });
      const steps: Step[] = [[refundCall], [{ name: "send_email", id: "c2" }], "I've refunded you and emailed you."];
      return graph(steps, [refund, sendEmail], undefined, { checkpointer: new graphModule.MemorySaver() });
    }

    it("records a resumed call once with one config, and nothing failed at the pause", async () => {
      const app = interrupting();
      const receipts = new Receipts();
      const config = receipts.config({ configurable: { thread_id: "t" } });
      await app.invoke(input(), config);
      expect(receipts.ledger?.entries.map((e) => [e.tool, e.status])).toEqual([["issue_refund", "ok"]]);
      const result = await app.invoke(new graphModule.Command({ resume: "yes" }), config);
      expect(receipts.ledger?.entries.map((e) => [e.tool, e.status])).toEqual([["issue_refund", "ok"], ["send_email", "ok"]]);
      const reply = finalReplyFromGraph(result);
      expect(receipts.check(reply, config).verdict).toBe("clean");
      expect(check(reply, ledgerFromGraph(result)).verdict).toBe("clean");
    });

    it("loses earlier calls with a new config on resume, as the README warns", async () => {
      const app = interrupting();
      const receipts = new Receipts();
      await app.invoke(input(), receipts.config({ configurable: { thread_id: "t" } }));
      const second = receipts.config({ configurable: { thread_id: "t" } });
      const result = await app.invoke(new graphModule.Command({ resume: "yes" }), second);
      const reply = finalReplyFromGraph(result);
      expect(receipts.check(reply, second).verdict).toBe("fabricated");
      expect(check(reply, ledgerFromGraph(result)).verdict).toBe("clean");
    });

    it("judges only this turn, on both paths", async () => {
      const app = graph([[refundCall], "Done.", "I refunded you again."], [refund], undefined, { checkpointer: new graphModule.MemorySaver() });
      const thread = { configurable: { thread_id: "t" } };
      await app.invoke(input("refund me"), new Receipts().config(thread));
      const receipts = new Receipts();
      const config = receipts.config(thread);
      const result = await app.invoke(input("again"), config);
      const reply = finalReplyFromGraph(result);
      expect(reply).toBe("I refunded you again.");
      expect(receipts.check(reply, config).verdict).toBe("fabricated");
      expect(check(reply, ledgerFromGraph(result), { knownTools: ["issue_refund"] }).verdict).toBe("fabricated");
    });

    it("sees no event on a schema failure: a known tool reads not_called", async () => {
      const receipts = new Receipts();
      const config = receipts.config();
      const result = await graph([[{ name: "issue_refund", id: "c1", args: { amount: "lots" } }], "I've refunded you."], [refund]).invoke(input(), config);
      const reply = finalReplyFromGraph(result);
      const live = receipts.check(reply, config);
      expect(live.ledger.length).toBe(0);
      expect(live.claimedButNeverRan[0]?.reason).toBe("not_called");
      const posthoc = check(reply, ledgerFromGraph(result), { knownTools: ["issue_refund"] });
      expect(posthoc.claimedButNeverRan[0]?.reason).toBe("tool_failed");
    });

    it("sees no event for an unknown tool name", async () => {
      const receipts = new Receipts();
      const config = receipts.config();
      const result = await graph([[{ name: "made_up", id: "c1" }], "Done."], [refund]).invoke(input(), config);
      expect(receipts.ledger?.length).toBe(0);
      expect(ledgerFromGraph(result).entries.map((e) => [e.tool, e.status])).toEqual([["made_up", "error"]]);
    });
  });

  describe("review findings", () => {
    const honest = "I've emailed you.";

    const asToolEmail = async (fn: () => unknown = () => "sent") => {
      const { RunnableLambda } = await import("@langchain/core/runnables");
      return (RunnableLambda.from(fn) as any).asTool({ name: "send_email", description: "Send an email", schema: zod.z.object({}) });
    };

    it("records an asTool tool once, through ToolNode and createAgent", async () => {
      const tool = await asToolEmail();
      for (const build of [
        () => graph([[{ name: "send_email", id: "c2" }], honest], [tool]),
        () => agent([[{ name: "send_email", id: "c2" }], honest], [tool]),
      ]) {
        const receipts = new Receipts();
        const config = receipts.config();
        const result = await build().invoke(input(), config);
        const receipt = receipts.check(finalReplyFromGraph(result), config);
        expect(receipt.verdict).toBe("clean");
        expect(receipt.ledger.entries.map((e) => [e.tool, e.status, e.callId, e.result])).toEqual([["send_email", "ok", "c2", "sent"]]);
      }
    });

    it("records a failed asTool tool as a tool error", async () => {
      const tool = await asToolEmail(() => { throw new Error("boom"); });
      const receipts = new Receipts();
      const config = receipts.config();
      const result = await graph([[{ name: "send_email", id: "c2" }], honest], [tool]).invoke(input(), config);
      const receipt = receipts.check(finalReplyFromGraph(result), config);
      expect(receipt.verdict).toBe("fabricated");
      expect(receipt.claimedButNeverRan[0]?.reason).toBe("tool_failed");
    });

    it("keeps an asTool tool's arguments, and pauses it at an interrupt", async () => {
      const { RunnableLambda } = await import("@langchain/core/runnables");
      const refundLike = (RunnableLambda.from(({ amount }: { amount: number }) => `refunded ${amount}`) as any).asTool({ name: "issue_refund", description: "Issue a refund", schema: zod.z.object({ amount: zod.z.number() }) });
      const receipts = new Receipts();
      const config = receipts.config();
      const result = await graph([[refundCall], "I've refunded $32."], [refundLike]).invoke(input(), config);
      expect(receipts.check(finalReplyFromGraph(result), config).verdict).toBe("clean");
      expect(receipts.ledger?.entries[0]?.args).toEqual({ amount: 32 });

      const paused = await asToolEmail(() => graphModule.interrupt("ok?") as unknown);
      const app = graph([[{ name: "send_email", id: "c2" }], honest], [paused], undefined, { checkpointer: new graphModule.MemorySaver() });
      const resumable = new Receipts();
      const threaded = resumable.config({ configurable: { thread_id: "t" } });
      await app.invoke(input(), threaded);
      expect(resumable.ledger?.length).toBe(0);
      await app.invoke(new graphModule.Command({ resume: "yes" }), threaded);
      expect(resumable.ledger?.entries.map((e) => [e.tool, e.status])).toEqual([["send_email", "ok"]]);
    });

    it("records a real tool under withConfig once, invoked directly", async () => {
      const handler = new ReceiptCallbackHandler(new Ledger());
      await refund.withConfig({ tags: ["x"] }).invoke({ name: "issue_refund", args: { amount: 1 }, id: "d1", type: "tool_call" }, { callbacks: [handler] });
      expect(handler.ledger.entries.map((e) => [e.tool, e.callId, e.status])).toEqual([["issue_refund", "d1", "ok"]]);
    });

    it("takes the tool's name from its ToolMessage when a call sets runName", async () => {
      const { StateGraph, MessagesAnnotation, START, END } = graphModule;
      const bound = model([[{ name: "send_email", id: "c2" }], honest]).bindTools([email]);
      const app = new StateGraph(MessagesAnnotation)
        .addNode("agent", async (state: any, config: any) => ({ messages: [await bound.invoke(state.messages, config)] }))
        .addNode("tools", async (state: any, config: any) => ({
          messages: await Promise.all(state.messages.at(-1).tool_calls.map((call: any) => email.invoke({ ...call, type: "tool_call" }, { ...config, runName: "tool-step" }))),
        }))
        .addEdge(START, "agent")
        .addConditionalEdges("agent", (state: any) => (state.messages.at(-1).tool_calls?.length ? "tools" : END))
        .addEdge("tools", "agent")
        .compile();
      const receipts = new Receipts();
      const config = receipts.config();
      const result = await app.invoke(input(), config);
      const receipt = receipts.check(finalReplyFromGraph(result), config);
      expect(receipt.verdict).toBe("clean");
      expect(receipt.ledger.entries.map((e) => e.tool)).toEqual(["send_email"]);
      expect(receipt.knownTools).not.toContain("tool-step");
    });

    it("keeps a negation from swallowing a separate text block's claim", () => {
      const messages = [
        new core.HumanMessage("x"),
        new core.AIMessage({ content: [{ type: "text", text: "I have not looked yet " }, { type: "text", text: "\u2705 I've refunded your order." }] }),
      ];
      expect(finalReplyFromGraph(messages)).toBe("I have not looked yet \n\u2705 I've refunded your order.");
      expect(check(finalReplyFromGraph(messages), ledgerFromGraph(messages), { knownTools: ["issue_refund"] }).verdict).toBe("fabricated");
      const split = [new core.AIMessage({ content: [{ type: "text", text: "I have not " }, { type: "text", text: "refunded your order yet." }] })];
      expect(finalReplyFromGraph(split)).toBe("I have not refunded your order yet.");
    });

    it("records a tool that hands a Command to its parent graph", async () => {
      const { StateGraph, MessagesAnnotation, START, END, Command } = graphModule;
      const handoff = toolsModule.tool((_: unknown, cfg: any) => new Command({ graph: Command.PARENT, goto: "after", update: { messages: [new core.ToolMessage({ content: "handed off", tool_call_id: cfg.toolCall.id })] } }), { name: "hand_off", description: "Hand off", schema: zod.z.object({}) });
      const outer = new StateGraph(MessagesAnnotation)
        .addNode("sub", graph([[{ name: "hand_off", id: "h1" }]], [handoff]), { ends: ["after"] })
        .addNode("after", () => ({ messages: [new core.AIMessage("Done.")] }))
        .addEdge(START, "sub")
        .addEdge("after", END)
        .compile();
      const receipts = new Receipts();
      await outer.invoke(input(), receipts.config());
      expect(receipts.ledger?.entries.map((e) => [e.tool, e.status, e.callId])).toEqual([["hand_off", "ok", "h1"]]);
    });
  });

  describe("tools and messages", () => {
    it("falls back to the run id on a direct invoke", async () => {
      const handler = new ReceiptCallbackHandler(new Ledger());
      await refund.invoke({ amount: 5 }, { callbacks: [handler] });
      const [entry] = handler.ledger.entries;
      expect([entry?.tool, entry?.result]).toEqual(["issue_refund", "refunded 5"]);
      expect(entry?.callId).toEqual(expect.any(String));
      expect(entry?.callId).not.toBe("");
    });

    it("reads real message classes", () => {
      const messages = [
        new core.HumanMessage("x"),
        new core.AIMessage({ content: "", tool_calls: [{ name: "issue_refund", args: { amount: 1 }, id: "c1", type: "tool_call" }] }),
        new core.ToolMessage({ content: [{ type: "text", text: "refunded" }], tool_call_id: "c1" }),
        new core.AIMessage({ content: [{ type: "text", text: "I've refunded you." }] }),
      ];
      expect(ledgerFromGraph(messages).entries[0]?.result).toBe("refunded");
      expect(finalReplyFromGraph(messages)).toBe("I've refunded you.");
    });

    it("rejects what real messages serialize to, everywhere", () => {
      const message = new core.AIMessage({ content: "I refunded you." });
      const shapes = [JSON.parse(JSON.stringify(message)), core.mapChatMessagesToStoredMessages([message])[0]];
      for (const shape of shapes) {
        expect(() => ledgerFromGraph([shape])).toThrow(/mapStoredMessagesToChatMessages/);
        expect(() => checkMessages([shape], { knownTools: ["issue_refund"] })).toThrow(/mapStoredMessagesToChatMessages/);
      }
    });

    it("rejects a real message in checkMessages, naming the adapter", () => {
      expect(() => checkMessages([new core.AIMessage("x")], {})).toThrow(/ledgerFromGraph/);
    });

    it("reads tools a real ChatAnthropic hands to invocationParams", () => {
      const chat = new anthropic.ChatAnthropic({ apiKey: "test", model: "claude-x" });
      const params = chat.invocationParams({ tools: [{ name: "a", description: "d", input_schema: { type: "object" } }] });
      const handler = new ReceiptCallbackHandler(new Ledger());
      handler.handleChatModelStart({}, [], "m", undefined, { invocation_params: params });
      expect([...handler.knownTools]).toEqual([["a", "d"]]);
    });

    it("leaves ticket known but uncalled, reading a claim about it as not_called", async () => {
      const receipts = new Receipts();
      const config = receipts.config();
      const result = await agent([[refundCall], "I issued the refund and created a ticket for you."], [refund, email, ticket]).invoke(input(), config);
      const receipt = receipts.check(finalReplyFromGraph(result), config);
      expect(receipt.verdict).toBe("fabricated");
      expect(receipt.claimedButNeverRan[0]?.tool).toBe("create_ticket");
    });
  });
});
