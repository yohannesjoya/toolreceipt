/** Ports python/tests/test_llm_extractor.py, with the JavaScript rules of PRD 09 §4. */
import { describe, expect, it, vi } from "vitest";

import { RuleExtractor, normalizeToolSpecs } from "../src/claims";
import { LLMExtractor as RootLLM, Ledger, LedgerEntry, acheck, acheckTurns, check } from "../src/index";
import { LLMExtractor, PROMPT_V2, PROMPT_VERSION, buildPrompt, messageText, parseOutput } from "../src/llm";

const TOOLS = normalizeToolSpecs([["send_email", "Send an email"], ["issue_refund", "Refund an {order}"]]);
const REPLY = "I sent the email.";
const OUT = '{"claims": [{"text": "check your inbox", "tool": "send_email"}]}';
const INBOX = "Refund done, check your inbox.";
const quiet = { warn: () => undefined };
const rules = (text = REPLY) => new RuleExtractor().extract(text, TOOLS).map((c) => c.toDict());
const dicts = (claims: Array<{ toDict(): unknown }>) => claims.map((c) => c.toDict());

describe("prompt", () => {
  it("version and placeholders", () => {
    expect(PROMPT_VERSION).toBe(2);
    expect(PROMPT_V2).toContain("{tools}");
    expect(PROMPT_V2).toContain("{reply}");
  });

  it("holds tools in order and the reply last", () => {
    const p = buildPrompt("Done {reply}", TOOLS);
    expect(p.indexOf("- send_email: Send an email")).toBeLessThan(p.indexOf("- issue_refund: Refund an {order}"));
    expect(p.trimEnd().endsWith("Done {reply}\n>>>")).toBe(true);
    expect(p.split("Done {reply}").length - 1).toBe(1);
  });

  it("does not expand placeholders in descriptions", () => {
    const p = buildPrompt("MY REPLY", normalizeToolSpecs([["odd_tool", "Issue {reply} refund }{ and {tools}"]]));
    expect(p.split("MY REPLY").length - 1).toBe(1);
    expect(p).toContain("- odd_tool: Issue {reply} refund }{ and {tools}");
    expect(p.trimEnd().endsWith("MY REPLY\n>>>")).toBe(true);
  });

  it("does not treat $ in a reply as a replacement pattern", () => {
    expect(buildPrompt("refunded $& and $1", [])).toContain("refunded $& and $1\n>>>");
  });
});

describe("parseOutput", () => {
  it("tolerates fences and prose", () => {
    expect(parseOutput('Sure:\n```json\n{"claims": [{"text": "I sent it", "tool": "send_email"}]}\n```')).toEqual([["I sent it", "send_email"]]);
    expect(parseOutput('{"claims": []}')).toEqual([]);
    expect(parseOutput('{"claims": [{"text": "all done", "tool": null}]}')).toEqual([["all done", null]]);
    expect(parseOutput('{"claims": [{"text": "a } b {", "tool": null}]} and {more} prose')).toEqual([["a } b {", null]]);
    expect(parseOutput('{"claims": [{"text": "quote \\" } here"}]}')).toEqual([['quote " } here', null]]);
  });

  it.each(["", "no json here", '{"claims": "x"}', '{"claims": [{"text": 5}]}', "[1, 2]", '{"claims": [{"tool": "send_email"}]}', '{"claims": [{"text": "x", "tool": 3}]}', '{"claims": [', "{not json}", '{"claims": [null]}'])("rejects %j", (bad) => {
    expect(() => parseOutput(bad)).toThrow(TypeError);
  });
});

describe("errors do not leak model output", () => {
  it.each(['{"claims":[{"text":my card 4242-4242 refund}]}', '{"text": jane@example.com refund }'])("%s", (bad) => {
    let message = "";
    try {
      parseOutput(bad);
    } catch (exc) {
      message = (exc as Error).message;
    }
    expect(message).toBe("unparsable JSON in model output");
  });
});

describe("extract", () => {
  it("adds LLM claims with source", () => {
    const ex = new LLMExtractor({ complete: () => OUT });
    const llm = ex.extract(INBOX, TOOLS).filter((c) => c.source === "llm");
    expect(llm.map((c) => [c.text, c.candidates, c.verb, c.confidence, c.object, c.count])).toEqual([["check your inbox", ["send_email"], "other", "medium", null, null]]);
    expect(llm[0]!.evidence().objectMatched).toBe(true);
    expect(ex.lastRaw).toBe(OUT);
    expect(ex.lastError).toBeNull();
  });

  it("passes the prompt with reply and tools", () => {
    const seen: string[] = [];
    new LLMExtractor({ complete: (p) => { seen.push(p); return '{"claims": []}'; } }).extract(REPLY, TOOLS);
    expect(seen.length).toBe(1);
    expect(seen[0]).toContain(REPLY);
    expect(seen[0]).toContain("- issue_refund:");
  });

  it("falls back to rule claims, records the error, warns without the reply", () => {
    const warn = vi.fn();
    const ex = new LLMExtractor({ logger: { warn }, complete: () => { throw new RangeError("rate limited"); } });
    expect(dicts(ex.extract(REPLY, TOOLS))).toEqual(rules());
    expect(ex.lastError).toBeInstanceOf(RangeError);
    expect(warn).toHaveBeenCalledOnce();
    expect(warn.mock.calls[0]![0]).toContain("RangeError");
    expect(warn.mock.calls[0]![0]).not.toContain(REPLY);
  });

  it.each(["", "nope", '{"claims": "x"}'])("falls back on unparsable output %j", (bad) => {
    const ex = new LLMExtractor({ logger: quiet, complete: () => bad });
    expect(dicts(ex.extract(REPLY, TOOLS))).toEqual(rules());
    expect(ex.lastError).toBeInstanceOf(TypeError);
    expect(ex.lastRaw).toBe(bad);
  });

  it("onError raise propagates", () => {
    expect(() => new LLMExtractor({ onError: "raise", complete: () => { throw new RangeError("down"); } }).extract("x", TOOLS)).toThrow(RangeError);
    expect(() => new LLMExtractor({ onError: "raise", complete: () => "nope" }).extract("x", TOOLS)).toThrow(TypeError);
  });

  it("a null completion is an error", () => {
    const ex = new LLMExtractor({ logger: quiet, complete: () => null as unknown as string });
    expect(dicts(ex.extract(REPLY, TOOLS))).toEqual(rules());
    expect(ex.lastError).toBeInstanceOf(TypeError);
    expect(() => new LLMExtractor({ onError: "raise", complete: () => null as unknown as string }).extract("x", TOOLS)).toThrow(TypeError);
  });

  it("a promise from complete is a wiring mistake, whatever onError says", () => {
    const asyncComplete = (() => Promise.reject(new Error("down"))) as unknown as (p: string) => string;
    expect(() => new LLMExtractor({ complete: asyncComplete }).extract(REPLY, TOOLS)).toThrow(/use acheck/);
  });

  it("a span across clauses judges only the first clause's literals", () => {
    const reply = "Your refund for order 4821 is on its way, and $5 will be credited as a goodwill gesture.";
    const span = "refund for order 4821 is on its way, and $5 will be credited";
    const out = JSON.stringify({ claims: [{ text: span, tool: "issue_refund" }] });
    const ledger = new Ledger([new LedgerEntry({ tool: "issue_refund", args: { amount: 64, order_id: "4821" } })]);
    const r = check(reply, ledger, { knownTools: ["issue_refund"], extractor: new LLMExtractor({ complete: () => out }) });
    expect(r.suspected).toEqual([]);
    expect(r.claimedButNeverRan).toEqual([]);
    expect(r.confirmed.map((c) => c.tool)).toEqual(["issue_refund"]);
  });

  it("is exported from the root", () => {
    expect(RootLLM).toBe(LLMExtractor);
  });
});

describe("messageText", () => {
  it("reads property, method, content and blocks", () => {
    expect(messageText("s")).toBe("s");
    expect(messageText({ text: "p" })).toBe("p");
    expect(messageText({ text: () => "a" })).toBe("a");
    expect(messageText({ content: "b" })).toBe("b");
    expect(messageText({ content: [{ type: "thinking", thinking: "x" }, { type: "text", text: "c" }] })).toBe("c");
    expect(() => messageText({})).toThrow(TypeError);
    expect(() => messageText(null)).toThrow(TypeError);
  });
});

describe("async", () => {
  it("aextract awaits complete and equals extract", async () => {
    const ex = new LLMExtractor({ complete: () => OUT });
    expect(dicts(await ex.aextract(INBOX, TOOLS))).toEqual(dicts(ex.extract(INBOX, TOOLS)));
    expect(ex.lastRaw).toBe(OUT);
    expect(ex.lastError).toBeNull();
  });

  it("aextract accepts a complete that returns a promise", async () => {
    const asyncComplete = (() => Promise.resolve(OUT)) as unknown as (p: string) => string;
    const got = await new LLMExtractor({ complete: asyncComplete }).aextract(INBOX, TOOLS);
    expect(got.filter((c) => c.source === "llm").length).toBe(1);
  });

  it("aextract falls back and records the error", async () => {
    const ex = new LLMExtractor({ logger: quiet, complete: () => { throw new RangeError("x"); } });
    expect(dicts(await ex.aextract(REPLY, TOOLS))).toEqual(rules());
    expect(ex.lastError).toBeInstanceOf(RangeError);
    expect(ex.lastRaw).toBeNull();
  });

  it("aextract onError raise rejects", async () => {
    await expect(new LLMExtractor({ onError: "raise", acomplete: () => Promise.reject(new RangeError("x")) }).aextract("x", TOOLS)).rejects.toThrow(RangeError);
  });

  it("acomplete backs acheck", async () => {
    const ex = new LLMExtractor({ acomplete: () => Promise.resolve(OUT) });
    const r = await acheck(INBOX, new Ledger(), { knownTools: TOOLS, extractor: ex });
    expect(r.suspected.map((c) => c.tool)).toEqual(["send_email"]);
    expect(ex.lastRaw).toBe(OUT);
  });

  it("an async-only extractor throws under extract and check, whatever onError says", () => {
    const ex = new LLMExtractor({ acomplete: () => Promise.resolve(OUT), onError: "fallback" });
    expect(() => ex.extract(REPLY, TOOLS)).toThrow(/async-only/);
    expect(() => check(REPLY, new Ledger(), { extractor: ex })).toThrow(/async-only/);
  });

  it("the constructor needs one callable and a valid onError", () => {
    expect(() => new LLMExtractor({})).toThrow(/complete or acomplete/);
    expect(() => new (LLMExtractor as unknown as new () => LLMExtractor)()).toThrow(/complete or acomplete/);
    expect(() => new LLMExtractor({ complete: () => "", onError: "nope" as never })).toThrow(/onError/);
  });

  it("an AbortError propagates under fallback and leaves lastError alone", async () => {
    const abort = Object.assign(new Error("aborted"), { name: "AbortError" });
    const ex = new LLMExtractor({ acomplete: () => Promise.reject(abort) });
    await expect(ex.aextract(REPLY, TOOLS)).rejects.toBe(abort);
    expect(ex.lastError).toBeNull();
  });

  // openai 7.51 and @anthropic-ai/sdk 1.5 throw this class on abort, with name "Error".
  it("an SDK's APIUserAbortError propagates under fallback", async () => {
    class APIUserAbortError extends Error {}
    const abort = new APIUserAbortError("Request was aborted.");
    expect(abort.name).toBe("Error");
    const ex = new LLMExtractor({ acomplete: () => Promise.reject(abort) });
    await expect(ex.aextract(REPLY, TOOLS)).rejects.toBe(abort);
    expect(ex.lastError).toBeNull();
  });

  it("works per turn through acheckTurns", async () => {
    const msgs = [{ role: "user", content: "a" }, { role: "assistant", content: INBOX }];
    const [r] = await acheckTurns(msgs, { knownTools: ["send_email"], extractor: new LLMExtractor({ acomplete: () => Promise.resolve(OUT) }) });
    expect(r!.suspected.map((c) => c.tool)).toEqual(["send_email"]);
  });
});

describe("fromChatModel", () => {
  const msg = { text: OUT };

  it("prefers ainvoke and is async-only", async () => {
    const calls: string[] = [];
    const ex = LLMExtractor.fromChatModel({
      invoke() { calls.push("invoke"); return Promise.resolve(msg); },
      ainvoke() { calls.push("ainvoke"); return Promise.resolve(msg); },
    });
    expect((await ex.aextract(INBOX, TOOLS)).length).toBeGreaterThan(0);
    expect(calls).toEqual(["ainvoke"]);
    expect(() => ex.extract(REPLY, TOOLS)).toThrow(/async-only/);
  });

  it("uses invoke with its this, and forwards options", async () => {
    class Model {
      out = msg;
      invoke() { return Promise.resolve(this.out); }
    }
    const ex = LLMExtractor.fromChatModel(new Model());
    await ex.aextract(INBOX, TOOLS);
    expect(ex.lastRaw).toBe(OUT);
    const down = LLMExtractor.fromChatModel({ invoke: () => Promise.reject(new RangeError("down")) }, { onError: "raise" });
    await expect(down.aextract("x", TOOLS)).rejects.toThrow(RangeError);
  });

  it("needs a callable", () => {
    expect(() => LLMExtractor.fromChatModel({})).toThrow(/invoke/);
  });
});
