/** Ports python/tests/test_messages_units.py. test_top_level_exports is replaced by the root import below. */
import { describe, expect, it } from "vitest";

import { acheckMessages, checkMessages, finalReply, ledgerFromMessages, normalizeToolSpecs } from "../src/index";
import { type ToolLike } from "../src/claims";
import { LedgerEntry, Ledger, NO_RESULT } from "../src/ledger";
import { isTurn, pair, runMessages, settled } from "../src/messages";
import { loadCases } from "./helpers";

const TOOLS = [{ type: "function", function: { name: "issue_refund", description: "Issue a refund." } }];

function turn(text: string, cid: string): Array<Record<string, unknown>> {
  return [
    { role: "user", content: text },
    { role: "assistant", content: null, tool_calls: [{ id: cid, type: "function", function: { name: "issue_refund", arguments: '{"order_id": "4821", "amount": 64}' } }] },
    { role: "tool", tool_call_id: cid, content: "Refund issued" },
    { role: "assistant", content: "I've refunded $64 for order 4821." },
  ];
}

class Obj {
  constructor(fields: Record<string, unknown>) {
    Object.assign(this, fields);
  }
}
function objects(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(objects);
  if (value !== null && typeof value === "object") {
    return new Obj(Object.fromEntries(Object.entries(value).map(([k, v]) => [k, objects(v)])));
  }
  return value;
}

describe("messages units", () => {
  it("skips system and developer messages", () => {
    const messages = [{ role: "system", content: "Be brief." }, { role: "developer", content: "x" }, ...turn("Refund order 4821.", "call_1")];
    expect(runMessages(messages)).toEqual(messages.slice(3));
    expect(checkMessages(messages, { knownTools: TOOLS }).verdict).toBe("clean");
  });

  it("scopes to the last user turn", () => {
    const first = turn("Refund order 4821.", "call_1");
    const second = turn("And again.", "call_2");
    expect(runMessages([...first, ...second])).toEqual(second.slice(1));
    expect(runMessages([])).toEqual([]);
    expect(runMessages(first.slice(1))).toEqual(first.slice(1));
  });

  it("acheckMessages equals checkMessages", async () => {
    const messages = turn("Refund order 4821.", "call_1");
    const sync = checkMessages(messages, { knownTools: TOOLS }).toDict();
    expect((await acheckMessages(messages, { knownTools: TOOLS })).toDict()).toEqual(sync);
    expect(checkMessages(messages, { knownTools: TOOLS, checkArgs: false }).verdict).toBe("clean");
  });

  it("reads SDK class instances and Maps like dicts", () => {
    const [[, c]] = loadCases<{ messages: unknown[]; known_tools: ToolLike[] }>("messages").filter(([s]) => s === "openai_refund_and_email_ok") as [[string, { messages: unknown[]; known_tools: ToolLike[] }]];
    const asDicts = checkMessages(c.messages, { knownTools: c.known_tools }).toDict();
    const asObjects = checkMessages(objects(c.messages) as unknown[], { knownTools: c.known_tools });
    expect(asObjects.toDict()).toEqual(asDicts);
    expect(asDicts["verdict"]).toBe("clean");
    expect((asDicts["ledger"] as unknown[]).length).toBe(2);
    const asMap = new Map<string, unknown>([["role", "assistant"], ["content", "Hi"]]);
    expect(isTurn(asMap)).toBe(true);
  });

  it("skips a reasoning item", () => {
    const messages = [
      { role: "user", content: "Refund order 4821." },
      { type: "reasoning", id: "rs_1", summary: [] },
      { type: "function_call", id: "fc_1", call_id: "call_1", name: "issue_refund", arguments: '{"order_id": "4821", "amount": 64}' },
      { type: "function_call_output", call_id: "call_1", output: "Refund issued" },
      { type: "message", role: "assistant", content: [{ type: "output_text", text: "I've refunded $64 for order 4821." }] },
    ];
    expect(runMessages(messages).length).toBe(4);
    expect(ledgerFromMessages(messages).entries.map((e) => e.status)).toEqual(["ok"]);
    expect(checkMessages(messages, { knownTools: TOOLS }).verdict).toBe("clean");
  });

  it("knownTools skips built-in tools without a name", () => {
    const specs = normalizeToolSpecs([
      { type: "web_search" },
      { type: "custom", custom: { name: "send_email", description: "Send an email." } },
      { type: "function", name: "issue_refund", description: "Issue a refund." },
    ]);
    expect(specs.map((s) => [s.name, s.description])).toEqual([["send_email", "Send an email."], ["issue_refund", "Issue a refund."]]);
  });

  it("settled entry defaults and errors", () => {
    expect(settled({ tool: "lookup_order" })!.status).toBe("ok");
    const e = settled({ tool: "x", status: "error" })!;
    expect([e.status, e.error]).toEqual(["error", NO_RESULT]);
    expect(settled({ tool: "x", status: "error", error: "boom" })!.error).toBe("boom");
    expect(settled({ tool: null })).toBeNull();
    expect(settled({ role: "user", tool: "x" })).toBeNull();
    expect(settled({ tool: "x", type: "function_call" })).toBeNull();
    expect(settled({ tool: "x", role: null, type: null })).not.toBeNull();
    for (const bad of [{ tool: "" }, { tool: 5 }, { tool: "x", status: "pending" }, { tool: "x", status: ["ok"] }]) {
      expect(() => settled(bad), JSON.stringify(bad)).toThrow(TypeError);
    }
  });

  it("structured result and error become text", () => {
    expect(settled({ tool: "lookup_order", result: { id: "450789469", name: "#1001" } })!.result).toBe('{"id": "450789469", "name": "#1001"}');
    expect(settled({ tool: "x", status: "error", error: { code: 504 } })!.error).toBe('{"code": 504}');
    expect(settled({ tool: "x", result: "plain" })!.result).toBe("plain");
    expect(settled({ tool: "x" })!.result).toBeNull();
  });

  it("timestamps must be numbers", () => {
    const e = settled({ tool: "x", started_at: 1, ended_at: 1.5 })!;
    expect([e.startedAt, e.endedAt]).toEqual([1, 1.5]);
    expect(settled({ tool: "x" })!.startedAt).toBeNull();
    for (const bad of [{ tool: "x", started_at: "2026-10-31T10:00" }, { tool: "x", ended_at: true }]) {
      expect(() => settled(bad)).toThrow(TypeError);
    }
  });

  it("settled errors do not leak the entry", () => {
    const secret = "dana.reyes@example.com owes $64";
    for (const bad of [{ tool: "", result: secret }, { tool: "x", status: "pending", args: { note: secret } }]) {
      let message = "";
      try {
        settled(bad);
      } catch (err) {
        message = (err as Error).message;
      }
      expect(message).not.toBe("");
      expect(message).not.toContain(secret);
      expect(message).not.toContain("dana");
    }
  });

  it("settled entry round-trips Ledger.toDict", () => {
    const dumped = [
      new LedgerEntry({ tool: "b", startedAt: 2.5, endedAt: 2.75, parentRunId: "r" }).toDict(),
      new LedgerEntry({ tool: "a", startedAt: 1.5, callId: "c1", args: { k: 1 }, result: "ok" }).toDict(),
    ];
    const entries = pair([{ role: "user", content: "hi" }, ...dumped]);
    expect(entries.map((e) => e.toDict())).toEqual(dumped);
    expect(new Ledger(entries).entries.map((e) => e.tool)).toEqual(["a", "b"]);
    expect(pair([new LedgerEntry({ tool: "a", status: "error", error: "boom" })])[0]!.error).toBe("boom");
  });

  it("final reply is empty after a settled entry", () => {
    const msgs = [{ role: "user", content: "x" }, { role: "assistant", content: "Done." }, { tool: "issue_refund" }];
    expect(finalReply(msgs)).toBe("");
    expect(finalReply(msgs.slice(0, 2))).toBe("Done.");
  });

  it("isTurn", () => {
    expect(isTurn({ role: "assistant", content: "Hi" })).toBe(true);
    expect(isTurn({ type: "message", role: "assistant", content: [{ type: "output_text", text: "Hi" }] })).toBe(true);
    expect(isTurn({ role: "assistant", content: null, tool_calls: [] })).toBe(false);
    expect(isTurn({ role: "assistant", content: "  \n" })).toBe(false);
    expect(isTurn({ role: "user", content: "Hi" })).toBe(false);
    expect(isTurn({ tool: "lookup_order" })).toBe(false);
  });
});
