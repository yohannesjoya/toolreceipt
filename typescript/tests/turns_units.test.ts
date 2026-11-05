/** Ports python/tests/test_turns_units.py. */
import { describe, expect, it } from "vitest";

import { Claim, type ClaimExtractor } from "../src/claims";
import { Receipt } from "../src/receipt";
import { TurnReceipt, acheckTurns, checkMessages, checkTurns } from "../src/index";
import { windows } from "../src/turns";
import { loadCases } from "./helpers";

const U = (text: string) => ({ role: "user", content: text });
const A = (text: string) => ({ role: "assistant", content: text });
const CALL = (cid: string) => ({ id: cid, type: "function", function: { name: "issue_refund", arguments: JSON.stringify({ order_id: "4821", amount: 64 }) } });
const TOOL = (cid: string) => ({ role: "tool", tool_call_id: cid, content: "Refund issued" });
const CALL_ONLY = { role: "assistant", content: null, tool_calls: [CALL("c1")] };
const TOOL_RESULT = TOOL("c1");

describe("turns units", () => {
  it("windows partition the list", () => {
    const msgs = [U("a"), CALL_ONLY, TOOL_RESULT, A("t1"), CALL_ONLY, TOOL_RESULT, A("t2"), U("b"), A("t3"), U("c")];
    expect(windows(msgs)).toEqual([[3, 6], [6, 8], [8, 10]]);
    expect(windows([])).toEqual([]);
    expect(windows([U("a"), CALL_ONLY])).toEqual([]);
  });

  it("tool-only and whitespace messages are not turns", () => {
    const msgs = [U("a"), { role: "assistant", content: "  \n", tool_calls: [CALL("c1")] }, TOOL("c1"), A("Done, refunded.")];
    const [r] = checkTurns(msgs, { knownTools: ["issue_refund"] });
    expect([r!.turn, r!.messageIndex, r!.verdict]).toEqual([1, 3, "clean"]);
  });

  it("leading entries before any turn are in window one", () => {
    const msgs = [{ tool: "lookup_order" }, CALL_ONLY, TOOL_RESULT, A("Anything else?")];
    const [r] = checkTurns(msgs, { knownTools: ["lookup_order", "issue_refund"] });
    expect(r!.ranButUnreported.map((e) => e.tool)).toEqual(["lookup_order", "issue_refund"]);
  });

  it("consecutive turns: the second has an empty window", () => {
    const msgs = [U("a"), A("Let me check."), CALL_ONLY, TOOL_RESULT, A("Still checking.")];
    const [r1, r2] = checkTurns(msgs, { knownTools: ["issue_refund"] });
    expect(r1!.ranButUnreported.map((e) => e.tool)).toEqual(["issue_refund"]);
    expect(r2!.ranButUnreported).toEqual([]);
    expect(r2!.verdict).toBe("clean");
    expect([...r2!.ledger].map((e) => e.tool)).toEqual(["issue_refund"]);
  });

  it("empty and no-text lists", () => {
    expect(checkTurns([])).toEqual([]);
    expect(checkTurns([U("a"), CALL_ONLY, TOOL_RESULT])).toEqual([]);
  });

  it("a LangChain object in an earlier run raises; a trailing user is harmless", () => {
    expect(() => checkTurns([{ type: "ai", content: "x" }, U("b"), A("Done.")])).toThrow(TypeError);
    expect(checkTurns([U("a"), A("Hi."), U("b")]).length).toBe(1);
  });

  it("checkMessages is clean where checkTurns flags", () => {
    const c = new Map(loadCases<{ messages: unknown[]; known_tools: never }>("turns")).get("openai_claim_then_pending_call")!;
    expect(checkMessages(c.messages, { knownTools: c.known_tools }).verdict).toBe("clean");
    expect(checkTurns(c.messages, { knownTools: c.known_tools })[0]!.verdict).toBe("fabricated");
  });

  it("ignoreTools and tool dicts", () => {
    const tools = [{ type: "function", function: { name: "lookup_order", description: "Look up an order." } }];
    const [r] = checkTurns([U("a"), { tool: "lookup_order" }, A("Anything else?")], { knownTools: tools, ignoreTools: ["lookup_order"] });
    expect(r!.verdict).toBe("clean");
    expect(r!.knownTools).toEqual(["lookup_order"]);
  });

  it("TurnReceipt toDict key order", () => {
    const [r] = checkTurns([U("a"), A("Hi.")]);
    expect(r).toBeInstanceOf(TurnReceipt);
    expect(r).toBeInstanceOf(Receipt);
    const keys = Object.keys(r!.toDict());
    expect(keys.slice(0, 3)).toEqual(["turn", "message_index", "text"]);
    expect(keys.slice(3)).toEqual(Object.keys(Receipt.prototype.toDict.call(r)));
  });

  it("acheckTurns matches and awaits per turn", async () => {
    const msgs = [U("a"), A("I've issued your refund."), CALL_ONLY, TOOL_RESULT, A("Done.")];
    const sync = checkTurns(msgs, { knownTools: ["issue_refund"] }).map((r) => r.toDict());
    expect((await acheckTurns(msgs, { knownTools: ["issue_refund"] })).map((r) => r.toDict())).toEqual(sync);
    const calls: string[] = [];
    const stub: ClaimExtractor = {
      extract: () => [],
      aextract: (text) => {
        calls.push(text);
        return Promise.resolve([]);
      },
    };
    await acheckTurns(msgs, { extractor: stub });
    expect(calls).toEqual(["I've issued your refund.", "Done."]);
  });

  it("an LLM claim lands in its turn only", () => {
    const llm: ClaimExtractor = {
      extract: (text) => (text === "Done." ? [new Claim({ text, verb: "other", candidates: ["send_email"], source: "llm" })] : []),
    };
    const [r1, r2] = checkTurns([U("a"), A("Hi."), A("Done.")], { knownTools: ["send_email"], extractor: llm });
    expect(r1!.suspected).toEqual([]);
    expect(r2!.suspected.map((c) => c.tool)).toEqual(["send_email"]);
    expect(r2!.ok).toBe(true);
  });

  it("ignoreTools across windows", () => {
    const msgs = [U("a"), { tool: "lookup_order" }, A("Let me check."), { tool: "lookup_order" }, A("Anything else?")];
    const [r1, r2] = checkTurns(msgs, { knownTools: ["lookup_order"], ignoreTools: ["lookup_order"] });
    expect([r1!.verdict, r2!.verdict]).toEqual(["clean", "clean"]);
    expect(r1!.ranButUnreported).toEqual([]);
    expect(r2!.ranButUnreported).toEqual([]);
  });

  it("knownTools forms match checkMessages, and a one-shot iterable lasts across turns", () => {
    const tools = [{ type: "function", name: "issue_refund", description: "Issue a refund." }, { type: "web_search" }];
    const msgs = [U("a"), A("I've issued your refund.")];
    const [r] = checkTurns(msgs, { knownTools: tools });
    expect(r!.verdict).toBe("fabricated");
    expect(r!.knownTools).toEqual(["issue_refund"]);
    const { turn: _t, message_index: _m, text: _x, ...got } = r!.toDict();
    expect(got).toEqual(checkMessages(msgs, { knownTools: tools }).toDict());
    function* once() {
      yield "issue_refund";
    }
    const two = checkTurns([U("a"), A("I've issued your refund."), A("I've issued your refund.")], { knownTools: once() });
    expect(two.map((x) => x.verdict)).toEqual(["fabricated", "fabricated"]);
    function* ignored() {
      yield "lookup_order";
    }
    const lookups = [U("a"), { tool: "lookup_order" }, A("One."), { tool: "lookup_order" }, A("Two."), { tool: "lookup_order" }, A("Three.")];
    const three = checkTurns(lookups, { knownTools: ["lookup_order"], ignoreTools: ignored() });
    expect(three.map((x) => x.verdict)).toEqual(["clean", "clean", "clean"]);
  });
});
