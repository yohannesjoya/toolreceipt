import { describe, expect, it } from "vitest";

import { Claim, ToolSpec, cloneClaim, normalizeToolSpecs } from "../src/claims";
import type { ToolLike } from "../src/claims";
import { Ledger, LedgerEntry, coerceArgs, truncate } from "../src/ledger";
import { Receipt, renderSummary } from "../src/receipt";

describe("LedgerEntry", () => {
  it("toDict emits the Python keys in order", () => {
    expect(Object.keys(new LedgerEntry({ tool: "x" }).toDict())).toEqual([
      "tool", "status", "args", "result", "error", "call_id", "parent_run_id", "started_at", "ended_at",
    ]);
  });
  it("fromDict applies Python defaults", () => {
    const e = LedgerEntry.fromDict({ tool: "x" });
    expect(e.status).toBe("ok");
    expect(e.args).toEqual({});
    expect([e.result, e.error, e.callId, e.parentRunId, e.startedAt, e.endedAt]).toEqual([null, null, null, null, null, null]);
    expect(LedgerEntry.fromDict({ tool: "y", status: "error", args: { a: 1 }, call_id: "c1", started_at: 1.5 }).toDict())
      .toEqual({ tool: "y", status: "error", args: { a: 1 }, result: null, error: null, call_id: "c1", parent_run_id: null, started_at: 1.5, ended_at: null });
  });
});

describe("fromDict rejects a missing required key (N2)", () => {
  it("LedgerEntry needs tool; Claim needs text and verb", () => {
    expect(() => LedgerEntry.fromDict({ status: "ok" })).toThrow(TypeError);
    expect(() => LedgerEntry.fromDict({ status: "ok" })).toThrow("tool");
    expect(() => Claim.fromDict({ verb: "send" })).toThrow("text");
    expect(() => Claim.fromDict({ text: "x" })).toThrow("verb");
    expect(JSON.stringify(LedgerEntry.fromDict({ tool: "x" }).toDict())).toContain('"tool":"x"');
  });
});

describe("Ledger", () => {
  it("sorts timed entries first by startedAt, untimed after in insertion order", () => {
    const l = new Ledger([
      new LedgerEntry({ tool: "c" }),
      new LedgerEntry({ tool: "b", startedAt: 2.0 }),
      new LedgerEntry({ tool: "a", startedAt: 1.0 }),
      new LedgerEntry({ tool: "d" }),
    ]);
    expect(l.entries.map((e) => e.tool)).toEqual(["a", "b", "c", "d"]);
    expect(l.toDict().map((e) => e["tool"])).toEqual(["a", "b", "c", "d"]);
    expect([...l].map((e) => e.tool)).toEqual(["a", "b", "c", "d"]);
    expect(l.length).toBe(4);
  });
  it("tool sets", () => {
    const l = Ledger.fromDict([{ tool: "a", status: "ok" }, { tool: "b", status: "error" }]);
    expect(l.okTools()).toEqual(new Set(["a"]));
    expect(l.errorTools()).toEqual(new Set(["b"]));
    expect(l.tools()).toEqual(new Set(["a", "b"]));
    l.append(new LedgerEntry({ tool: "c" }));
    expect(l.length).toBe(3);
  });
});

describe("coerceArgs", () => {
  it("makes args JSON-safe like Python's json round trip", () => {
    expect(coerceArgs(null)).toEqual({});
    expect(coerceArgs(undefined)).toEqual({});
    expect(coerceArgs({ a: 1n })).toEqual({ a: "1" });
    expect(coerceArgs([1, 2])).toEqual({ input: [1, 2] });
    expect(coerceArgs("x")).toEqual({ input: "x" });
    const cyc: Record<string, unknown> = {};
    cyc["self"] = cyc;
    expect(coerceArgs(cyc)).toEqual({ input: "[object Object]" });
    expect(coerceArgs({ f() { return 1; }, u: undefined, k: 1 })).toEqual({ k: 1 });
  });
});

describe("truncate", () => {
  it("counts code points and appends an ellipsis", () => {
    expect(truncate("\u{1F600}\u{1F600}\u{1F600}\u{1F600}", 3)).toBe("\u{1F600}\u{1F600}…");
    expect(truncate("abc", 3)).toBe("abc");
    expect(truncate("abcd", 0)).toBe("");
    expect(truncate(null, 3)).toBeNull();
  });
});

describe("Claim", () => {
  it("toDict emits the Python keys in order with defaults", () => {
    const c = new Claim({ text: "x", verb: "send" });
    expect(Object.keys(c.toDict())).toEqual([
      "text", "verb", "object", "count", "candidates", "tool", "confidence", "reason", "source", "observed",
    ]);
    expect(c.toDict()).toEqual({
      text: "x", verb: "send", object: null, count: null, candidates: [], tool: null, confidence: null,
      reason: null, source: "rules", observed: null,
    });
    expect(Claim.fromDict({ text: "x", verb: "other", source: "llm", candidates: ["a"] }).toDict())
      .toMatchObject({ source: "llm", candidates: ["a"] });
  });
  it("keeps evidence out of serialisation", () => {
    const c = new Claim({ text: "x", verb: "send" });
    expect(c.evidence()).toEqual({ objectMatched: null, clause: null, sentence: null });
    c.setEvidence({ objectMatched: true, sentence: "S" });
    expect(c.evidence()).toEqual({ objectMatched: true, clause: null, sentence: "S" });
    expect(JSON.stringify(c)).not.toContain("objectMatched");
    expect(Object.keys({ ...c })).not.toContain("_objectMatched");
    const copy = cloneClaim(c, { tool: "send_email", candidates: ["send_email"] });
    expect(copy.tool).toBe("send_email");
    expect(copy.evidence().objectMatched).toBe(true);
    expect(c.tool).toBeNull();
  });
});

describe("normalizeToolSpecs", () => {
  it("follows Python's branch order", () => {
    class Tool { name = "h"; description = "H"; }
    const specs = normalizeToolSpecs([
      "a", ["b", "B"], { type: "function", function: { name: "c", description: "C" } }, { type: "web_search" },
      { name: "d" }, ["e"], new ToolSpec("f", "F"), new Map([["name", "g"]]), new Tool(),
    ]);
    expect(specs.map((s) => s.name)).toEqual(["a", "b", "c", "d", "e", "f", "g", "h"]);
    expect(specs.map((s) => s.description)).toEqual(["", "B", "C", "", "", "F", "", "H"]);
    expect(() => normalizeToolSpecs([3 as unknown as ToolLike])).toThrow(TypeError);
    expect(() => normalizeToolSpecs([3 as unknown as ToolLike])).toThrow(/got number$/);
    expect(() => normalizeToolSpecs([3 as unknown as ToolLike])).toThrow(
      "known_tools entries must be a name, a (name, description) pair, a ToolSpec, a dict or an object with a name attribute; got number",
    );
  });
});

describe("renderSummary", () => {
  const s = new Claim({ text: "check your inbox", verb: "other", candidates: ["send_email"], tool: "send_email", reason: "not_called", source: "llm" });
  const never = new Claim({ text: "I've refunded $64", verb: "refund", candidates: ["issue_refund"], tool: "issue_refund", reason: "args_mismatch", observed: { amount: 32 } });
  const le = (tool: string) => new LedgerEntry({ tool });
  it("renders the Python lines", () => {
    expect(renderSummary("clean", [], [], [s], 2)).toBe('2 claims confirmed; 1 suspected (LLM only): send_email ("check your inbox").');
    expect(renderSummary("clean", [], [], [s], 0)).toBe('No claims; 1 suspected (LLM only): send_email ("check your inbox").');
    expect(renderSummary("clean", [], [], [], 1)).toBe("All claims confirmed.");
    expect(renderSummary("unreported", [], [le("lookup_order"), le("lookup_order"), le("send_email")], []))
      .toBe("All claims confirmed; 2 tools ran unreported: lookup_order, send_email.");
    expect(renderSummary("unreported", [], [le("lookup_order")], [s], 2))
      .toBe('2 claims confirmed; 1 tool ran unreported: lookup_order; 1 suspected (LLM only): send_email ("check your inbox").');
    expect(renderSummary("fabricated", [never], [], [s]))
      .toBe('1 claimed action never ran: issue_refund ("I\'ve refunded $64", ledger amount=32); 1 more suspected.');
    expect(renderSummary("fabricated", [never, never], [], []))
      .toBe('2 claimed actions never ran: issue_refund ("I\'ve refunded $64", ledger amount=32).');
  });
  it("renders observed through Python json.dumps (Review Focus 5)", () => {
    const c = cloneClaim(never, { observed: { note: "café", amount: 2.5 } });
    expect(renderSummary("fabricated", [c], [], []))
      .toBe('1 claimed action never ran: issue_refund ("I\'ve refunded $64", ledger note="caf\\u00e9", amount=2.5).');
  });
});

describe("Receipt", () => {
  it("toDict key order and ok", () => {
    const r = new Receipt({
      confirmed: [], claimedButNeverRan: [], unmatched: [], ranButUnreported: [], verdict: "clean",
      summary: "All claims confirmed.", ledger: new Ledger(), knownTools: ["a"],
    });
    expect(Object.keys(r.toDict())).toEqual([
      "confirmed", "claimed_but_never_ran", "unmatched", "suspected", "ran_but_unreported", "verdict", "ok", "summary", "ledger", "known_tools",
    ]);
    expect(r.ok).toBe(true);
    expect(r.suspected).toEqual([]);
    const f = new Receipt({ ...r, verdict: "fabricated", suspected: [] });
    expect(f.ok).toBe(false);
    expect(f.toDict()["ok"]).toBe(false);
  });
});
