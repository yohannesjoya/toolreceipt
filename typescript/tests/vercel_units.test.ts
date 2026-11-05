/** Result and model-message readers for the Vercel AI SDK, on shapes the real SDK does not produce on demand. */
import { describe, expect, it } from "vitest";

import { acheckResult, checkMessages, checkResult, finalReply, finalReplyFromResult, knownToolsFrom, ledgerFromMessages, ledgerFromResult } from "../src/index";
import { checkTurns } from "../src/turns";

const callPart = (toolCallId: string, toolName: string, input: unknown = {}) => ({ type: "tool-call", toolCallId, toolName, input });
const resultPart = (toolCallId: string, toolName: string, output: unknown) => ({ type: "tool-result", toolCallId, toolName, output });

describe("ledgerFromResult", () => {
  it("pairs across steps by toolCallId", () => {
    const steps = [
      { content: [callPart("a", "one", { x: 1 }), callPart("b", "two")] },
      { content: [{ type: "tool-result", toolCallId: "b", toolName: "two", input: {}, output: "b done" }, { type: "tool-result", toolCallId: "a", toolName: "one", input: { x: 1 }, output: { n: 1 } }] },
    ];
    const entries = ledgerFromResult({ steps }).entries;
    expect(entries.map((e) => [e.tool, e.status, e.result, e.callId])).toEqual([["one", "ok", '{"n": 1}', "a"], ["two", "ok", "b done", "b"]]);
    expect(entries[0]!.args).toEqual({ x: 1 });
  });

  it("marks a call with no result as no result", () => {
    const [entry] = ledgerFromResult({ steps: [{ content: [callPart("a", "one")] }] }).entries;
    expect([entry!.status, entry!.error]).toEqual(["error", "no result"]);
  });

  it("marks tool-error as an error with its text", () => {
    const part = { type: "tool-error", toolCallId: "a", toolName: "one", input: {}, error: new Error("boom") };
    const [entry] = ledgerFromResult({ steps: [{ content: [callPart("a", "one"), part] }] }).entries;
    expect([entry!.status, entry!.error]).toEqual(["error", "Error: boom"]);
  });

  it("skips preliminary results", () => {
    const content: unknown[] = [callPart("a", "one"), { type: "tool-result", toolCallId: "a", toolName: "one", output: "partial", preliminary: true }];
    expect(ledgerFromResult({ steps: [{ content }] }).entries[0]!.status).toBe("error");
    content.push({ type: "tool-result", toolCallId: "a", toolName: "one", output: "final" });
    expect(ledgerFromResult({ steps: [{ content }] }).entries[0]!.result).toBe("final");
  });

  it("names streamText's promise, and rejects what it cannot read", () => {
    expect(() => ledgerFromResult({ steps: Promise.resolve([]) })).toThrow(/await result\.steps/);
    expect(() => ledgerFromResult({})).toThrow(TypeError);
    expect(() => ledgerFromResult({ steps: [{ text: "AI SDK 4 step" }] })).toThrow(/AI SDK 5/);
    expect(() => ledgerFromResult(null)).toThrow(TypeError);
  });

  it("never reads result.toolCalls, whose meaning changed in AI SDK 7", () => {
    expect(ledgerFromResult({ steps: [], toolCalls: [callPart("a", "one")] }).length).toBe(0);
  });
});

describe("finalReplyFromResult", () => {
  const text = (t: string) => ({ type: "text", text: t });

  it("is the last step's text, joined as result.text joins it", () => {
    const steps = [{ content: [text("thinking")] }, { content: [text("a"), text("b")] }];
    expect(finalReplyFromResult({ steps })).toBe("ab");
  });

  it("is empty when the last step ends on a tool call", () => {
    expect(finalReplyFromResult({ steps: [{ content: [text("hi"), callPart("a", "one")] }] })).toBe("");
    expect(finalReplyFromResult({ steps: [] })).toBe("");
  });

  it("keeps the text beside a provider-executed call", () => {
    const steps = [{ content: [{ ...callPart("a", "one"), providerExecuted: true }, resultPart("a", "one", 1), text("done")] }];
    expect(finalReplyFromResult({ steps })).toBe("done");
  });
});

describe("checkResult", () => {
  const steps = [{ content: [{ type: "text", text: "I've refunded $64." }] }];

  it("checks the reply against the steps", () => {
    const receipt = checkResult({ steps }, { knownTools: ["issue_refund"] });
    expect(receipt.verdict).toBe("fabricated");
  });

  it("has an async form", async () => {
    expect((await acheckResult({ steps }, { knownTools: ["issue_refund"] })).verdict).toBe("fabricated");
  });
});

describe("knownToolsFrom", () => {
  it("lists names and descriptions, including tools with no execute", () => {
    const specs = knownToolsFrom({ a: { description: "Do a" }, b: {}, c: { description: 7 } });
    expect(specs.map((s) => [s.name, s.description])).toEqual([["a", "Do a"], ["b", ""], ["c", ""]]);
  });

  it("rejects what is not a record", () => {
    expect(() => knownToolsFrom([])).toThrow(TypeError);
    expect(() => knownToolsFrom(null)).toThrow(TypeError);
  });
});

describe("AI SDK model messages", () => {
  const assistant = (...content: unknown[]) => ({ role: "assistant", content });
  const tool = (...content: unknown[]) => ({ role: "tool", content });
  const reply = assistant({ type: "text", text: "Done." });

  it.each([
    [{ type: "text", value: "fine" }, "ok", "fine", null],
    [{ type: "json", value: { a: 1 } }, "ok", '{"a": 1}', null],
    [{ type: "error-text", value: "boom" }, "error", null, "boom"],
    [{ type: "error-json", value: { code: 5 } }, "error", null, '{"code": 5}'],
    [{ type: "execution-denied", reason: "user said no" }, "error", null, "user said no"],
    [{ type: "execution-denied" }, "error", null, "execution denied"],
    [{ type: "content", value: [{ type: "text", text: "one" }, { type: "image-url", url: "u" }, { type: "text", text: "two" }] }, "ok", "one\ntwo", null],
  ])("reads output %s", (output, status, result, error) => {
    const [entry] = ledgerFromMessages([assistant(callPart("a", "t", { x: 1 })), tool(resultPart("a", "t", output)), reply]).entries;
    expect([entry!.tool, entry!.status, entry!.result, entry!.error, entry!.callId]).toEqual(["t", status, result, error, "a"]);
    expect(entry!.args).toEqual({ x: 1 });
  });

  it("marks a call with no result as no result", () => {
    const [entry] = ledgerFromMessages([assistant(callPart("a", "t"))]).entries;
    expect([entry!.status, entry!.error]).toEqual(["error", "no result"]);
  });

  it("does not let a tool message of parts read as an OpenAI tool message", () => {
    const list = [assistant(callPart("", "t")), tool(resultPart("other", "t", { type: "text", value: "x" }))];
    expect(ledgerFromMessages(list).entries[0]!.status).toBe("error");
  });

  it("reads a provider tool result held in the assistant message", () => {
    const message = assistant({ ...callPart("p", "search"), providerExecuted: true }, resultPart("p", "search", { type: "json", value: 1 }), { type: "text", text: "Found it." });
    expect(ledgerFromMessages([message]).entries[0]!.status).toBe("ok");
    expect(finalReply([message])).toBe("Found it.");
  });

  it("keeps a run that ends on a local tool call without a final reply", () => {
    expect(finalReply([assistant({ type: "text", text: "Working." }, callPart("a", "t"))])).toBe("");
  });

  it("starts the last turn at the last user message", () => {
    const list = [
      assistant(callPart("old", "before")),
      { role: "user", content: [{ type: "text", text: "again" }] },
      assistant(callPart("a", "t")),
      tool(resultPart("a", "t", { type: "text", value: "x" })),
      reply,
    ];
    expect([...ledgerFromMessages(list).tools()]).toEqual(["t"]);
  });

  it("gives per-turn receipts", () => {
    const list = [
      { role: "user", content: "refund me" },
      assistant(callPart("a", "issue_refund", { amount: 64 })),
      tool(resultPart("a", "issue_refund", { type: "text", value: "x" })),
      assistant({ type: "text", text: "I've refunded $64." }),
    ];
    expect(checkTurns(list, { knownTools: ["issue_refund"] }).map((t) => t.verdict)).toEqual(["clean"]);
  });

  it("rejects AI SDK 4 parts rather than reading an empty ledger", () => {
    expect(() => checkMessages([assistant({ type: "tool-call", toolCallId: "a", toolName: "t", args: {} })])).toThrow(/AI SDK 4/);
    expect(() => checkMessages([tool({ type: "tool-result", toolCallId: "a", toolName: "t", result: 1 })])).toThrow(/AI SDK 4/);
  });
});
