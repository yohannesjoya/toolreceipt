/**
 * The real `ai` package, versions 6 and 7, with its mock model: no network. These pin the shapes the
 * readers depend on, so a new AI SDK major that moves them fails here.
 */
/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-return, @typescript-eslint/no-unsafe-argument */
import { describe, expect, it } from "vitest";

import { Receipts, checkMessages, checkResult, knownToolsFrom, ledgerFromMessages, ledgerFromResult } from "../src/index";

const usage = { inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 }, outputTokens: { total: 1, text: 1, reasoning: 0 } };
const finish = (unified: string) => ({ unified, raw: unified });

const call = (toolCallId: string, toolName: string, input: unknown) => ({ type: "tool-call", toolCallId, toolName, input: JSON.stringify(input) });
const callStep = (...content: unknown[]) => ({ content, finishReason: finish("tool-calls"), usage, warnings: [] });
const textStep = (text: string) => ({ content: [{ type: "text", text }], finishReason: finish("stop"), usage, warnings: [] });

const nodeMajor = Number(process.versions.node.split(".")[0]);
const VERSIONS = [
  {
    name: "ai 6",
    skip: false,
    load: async (): Promise<[any, any]> => [await import("ai6"), await import("ai6/test")],
    allMessages: (r: any) => r.response.messages,
  },
  {
    name: "ai 7",
    skip: nodeMajor < 22, // `ai@7` declares node >=22.
    load: async (): Promise<[any, any]> => [await import("ai"), await import("ai/test")],
    allMessages: (r: any) => r.responseMessages,
  },
];

for (const version of VERSIONS) {
  describe.skipIf(version.skip)(version.name, () => {
    function modelOf(test: any, steps: unknown[]): unknown {
      let i = 0;
      return new test.MockLanguageModelV3({ doGenerate: () => Promise.resolve(steps[i++]) });
    }

    function toolsOf(ai: any, overrides: Record<string, unknown> = {}): Record<string, any> {
      const inputSchema = ai.jsonSchema({ type: "object" });
      return {
        issue_refund: ai.tool({ description: "Issue a refund to the customer", inputSchema, execute: () => Promise.resolve({ refunded: true }) }),
        send_email: ai.tool({ description: "Send an email", inputSchema, execute: () => Promise.resolve("sent") }),
        ...overrides,
      };
    }

    function generate(ai: any, test: any, tools: unknown, steps: unknown[]): Promise<any> {
      const stopWhen = ai.isStepCount ? ai.isStepCount(5) : ai.stepCountIs(5);
      return ai.generateText({ model: modelOf(test, steps), tools, prompt: "refund me", stopWhen });
    }

    const REPLY = "I've refunded $32 and emailed you the confirmation.";

    it("gives clean on an honest two-step run through all three entry points", async () => {
      const [ai, test] = await version.load();
      const tools = toolsOf(ai);
      const receipts = new Receipts();
      const wrapped = receipts.wrapTools(tools);
      const steps = [callStep(call("c1", "issue_refund", { amount: 32 }), call("c2", "send_email", {})), textStep(REPLY)];
      const result = await generate(ai, test, wrapped, steps);
      const knownTools = knownToolsFrom(tools);

      const live = receipts.check(result.text);
      expect(live.verdict).toBe("clean");
      expect(live.confirmed.map((c) => c.tool)).toEqual(["issue_refund", "send_email"]);
      expect(checkResult(result, { knownTools }).verdict).toBe("clean");
      expect(checkMessages(version.allMessages(result), { knownTools }).verdict).toBe("clean");

      const entries = receipts.ledger!.entries;
      expect(entries.map((e) => [e.tool, e.status, e.callId])).toEqual([["issue_refund", "ok", "c1"], ["send_email", "ok", "c2"]]);
      expect(entries[0]!.args).toEqual({ amount: 32 });
      expect(entries[0]!.result).toBe('{"refunded": true}');
      expect(entries[1]!.result).toBe("sent");
    });

    it("gives fabricated when the model only said it acted", async () => {
      const [ai, test] = await version.load();
      const tools = toolsOf(ai);
      const receipts = new Receipts();
      const result = await generate(ai, test, receipts.wrapTools(tools), [textStep(REPLY)]);
      const knownTools = knownToolsFrom(tools);
      for (const receipt of [receipts.check(result.text), checkResult(result, { knownTools }), checkMessages(version.allMessages(result), { knownTools })]) {
        expect(receipt.verdict).toBe("fabricated");
        expect(receipt.claimedButNeverRan.map((c) => [c.tool, c.reason])).toEqual([["issue_refund", "not_called"], ["send_email", "not_called"]]);
      }
    });

    it("reports a tool that threw as tool_failed, and the run goes on", async () => {
      const [ai, test] = await version.load();
      const tools = toolsOf(ai, {
        send_email: ai.tool({ description: "Send an email", inputSchema: ai.jsonSchema({ type: "object" }), execute: () => Promise.reject(new Error("smtp down")) }),
      });
      const receipts = new Receipts();
      const steps = [callStep(call("c1", "issue_refund", { amount: 32 }), call("c2", "send_email", {})), textStep(REPLY)];
      const result = await generate(ai, test, receipts.wrapTools(tools), steps);
      const knownTools = knownToolsFrom(tools);
      for (const receipt of [receipts.check(result.text), checkResult(result, { knownTools }), checkMessages(version.allMessages(result), { knownTools })]) {
        expect(receipt.verdict).toBe("fabricated");
        expect(receipt.claimedButNeverRan.map((c) => [c.tool, c.reason])).toEqual([["send_email", "tool_failed"]]);
      }
      expect(receipts.ledger!.errorTools()).toEqual(new Set(["send_email"]));
      expect(receipts.ledger!.entries.find((e) => e.tool === "send_email")!.error).toBe("Error: smtp down");
    });

    it("passes a streaming tool's output through and records its last yield", async () => {
      const [ai, test] = await version.load();
      const tools = toolsOf(ai, {
        send_email: ai.tool({
          description: "Send an email",
          inputSchema: ai.jsonSchema({ type: "object" }),
          // eslint-disable-next-line @typescript-eslint/require-await
          execute: async function* () { yield "queued"; yield "sent"; },
        }),
      });
      const receipts = new Receipts();
      const result = await generate(ai, test, receipts.wrapTools(tools), [callStep(call("c1", "send_email", {})), textStep("I emailed you.")]);
      expect(receipts.ledger!.entries[0]!.result).toBe("sent");
      expect(ledgerFromResult(result).entries[0]!.result).toBe("sent");
      expect(receipts.check(result.text).verdict).toBe("clean");
    });

    it("works under streamText", async () => {
      const [ai, test] = await version.load();
      const tools = toolsOf(ai);
      const receipts = new Receipts();
      const chunks = [
        [{ type: "stream-start", warnings: [] }, call("c1", "issue_refund", { amount: 32 }), call("c2", "send_email", {}), { type: "finish", finishReason: finish("tool-calls"), usage }],
        [
          { type: "stream-start", warnings: [] },
          { type: "text-start", id: "t" },
          { type: "text-delta", id: "t", delta: REPLY },
          { type: "text-end", id: "t" },
          { type: "finish", finishReason: finish("stop"), usage },
        ],
      ];
      let i = 0;
      const model = new test.MockLanguageModelV3({
        doStream: () => Promise.resolve({ stream: test.convertArrayToReadableStream(chunks[i++]) }),
      });
      const stopWhen = ai.isStepCount ? ai.isStepCount(5) : ai.stepCountIs(5);
      const result = ai.streamText({ model, tools: receipts.wrapTools(tools), prompt: "refund me", stopWhen });
      const text = await result.text;
      expect(text).toBe(REPLY);
      expect(receipts.check(text).verdict).toBe("clean");
      expect(() => ledgerFromResult(result)).toThrow(/await result\.steps/);
      expect(checkResult({ steps: await result.steps }, { knownTools: knownToolsFrom(tools) }).verdict).toBe("clean");
    });

    it("leaves a provider tool out of live capture and reads it from the steps", async () => {
      const [ai, test] = await version.load();
      const tools = toolsOf(ai, { send_fax: { type: "provider", id: "x.send_fax", name: "send_fax", args: {}, inputSchema: ai.jsonSchema({ type: "object" }) } });
      const reply = "I've faxed the form.";
      const steps = [{
        content: [
          { type: "tool-call", toolCallId: "p1", toolName: "send_fax", input: "{}", providerExecuted: true },
          { type: "tool-result", toolCallId: "p1", toolName: "send_fax", result: { ok: true } },
          { type: "text", text: reply },
        ],
        finishReason: finish("stop"),
        usage,
        warnings: [],
      }];
      const receipts = new Receipts();
      const wrapped = receipts.wrapTools(tools);
      expect(wrapped["send_fax"]).toBe(tools["send_fax"]);
      const result = await generate(ai, test, wrapped, steps);

      const live = receipts.check(result.text);
      expect(live.ledger.length).toBe(0);
      expect(live.ok).toBe(true); // never accused: the wrapper cannot see this tool

      const knownTools = knownToolsFrom(tools);
      const fromSteps = checkResult(result, { knownTools });
      expect(fromSteps.ledger.entries.map((e) => [e.tool, e.status])).toEqual([["send_fax", "ok"]]);
      expect(fromSteps.ok).toBe(true);
      const fromMessages = checkMessages(version.allMessages(result), { knownTools });
      expect(fromMessages.ledger.entries.map((e) => [e.tool, e.status])).toEqual([["send_fax", "ok"]]);
      expect(fromMessages.ok).toBe(true);
      expect(checkResult(result).confirmed.length).toBe(checkMessages(version.allMessages(result)).confirmed.length);
    });

    it("hands the tool's own output to the model unchanged", async () => {
      const [ai, test] = await version.load();
      const output = { refunded: true, nested: { id: 7 } };
      const tools = toolsOf(ai, {
        issue_refund: ai.tool({ description: "Refund", inputSchema: ai.jsonSchema({ type: "object" }), execute: () => Promise.resolve(output) }),
      });
      const result = await generate(ai, test, new Receipts().wrapTools(tools), [callStep(call("c1", "issue_refund", {})), textStep("Done.")]);
      const part = result.steps[0].content.find((p: any) => p.type === "tool-result");
      expect(part.output).toBe(output);
    });
  });
}

describe.skipIf(nodeMajor < 22)("ai 7 response.messages", () => {
  it("holds the last step only, which is why the README sends callers to responseMessages", async () => {
    const ai: any = await import("ai");
    const test: any = await import("ai/test");
    let i = 0;
    const steps = [callStep(call("c1", "send_email", {})), textStep("I emailed you.")];
    const result = await ai.generateText({
      model: new test.MockLanguageModelV3({ doGenerate: () => Promise.resolve(steps[i++]) }),
      tools: { send_email: ai.tool({ description: "Send an email", inputSchema: ai.jsonSchema({ type: "object" }), execute: () => Promise.resolve("sent") }) },
      prompt: "email me",
      stopWhen: ai.isStepCount(5),
    });
    expect(ledgerFromMessages(result.response.messages).length).toBe(0);
    expect(ledgerFromMessages(result.responseMessages).length).toBe(1);
  });
});
