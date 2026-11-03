/** End-to-end fixtures: reply + bound tools + ledger -> verdict, through check(). */
import { describe, expect, it } from "vitest";

import { diff } from "../src/diff";
import { Ledger } from "../src/ledger";
import { check } from "../src/receipts";
import { type CheckCase, loadCases } from "./helpers";

const CASES = loadCases<CheckCase>("check");
const byStem = new Map(CASES);

function run(c: CheckCase) {
  return check(c.text, Ledger.fromDict(c.ledger), { knownTools: c.known_tools });
}

describe("check fixtures", () => {
  it.each(CASES)("%s", (_stem, c) => {
    const r = run(c);
    expect(r.verdict, JSON.stringify(r.toDict())).toBe(c.expected.verdict);
    if (c.expected.claimed_but_never_ran) {
      const got = r.claimedButNeverRan.map((cl) => [cl.tool, cl.reason, cl.observed]);
      const want = c.expected.claimed_but_never_ran.map((cl) => [cl["tool"], cl["reason"], cl["observed"] ?? null]);
      expect(got, JSON.stringify(r.toDict())).toEqual(want);
    }
    if (c.expected.summary !== undefined) expect(r.summary).toBe(c.expected.summary);
    expect(r.suspected).toEqual([]);
    expect([...r.confirmed, ...r.claimedButNeverRan, ...r.unmatched].every((cl) => cl.source === "rules")).toBe(true);
  });

  it("sibling tool paraphrase is unmatched, not never ran", () => {
    const r = run(byStem.get("sibling_raised_with_team")!);
    expect(r.claimedButNeverRan).toEqual([]);
    expect(r.unmatched.map((c) => [c.candidates, c.tool, c.reason])).toEqual([[["create_ticket"], null, null]]);
  });

  it("family-only candidate that failed still blocks", () => {
    const ledger = Ledger.fromDict([{ tool: "create_ticket", status: "error" }]);
    const r = check("I've raised this with our team.", ledger, { knownTools: ["create_ticket"] });
    expect(r.verdict).toBe("fabricated");
    expect(r.claimedButNeverRan.map((c) => [c.tool, c.reason])).toEqual([["create_ticket", "tool_failed"]]);
  });

  it("family-only candidate that ran is confirmed", () => {
    const ledger = Ledger.fromDict([{ tool: "create_ticket", status: "ok" }]);
    const r = check("I've raised this with our team.", ledger, { knownTools: ["create_ticket"] });
    expect(r.verdict).toBe("clean");
    expect(r.confirmed.map((c) => c.tool)).toEqual(["create_ticket"]);
  });

  it("dedupe keeps object evidence of any duplicate", () => {
    const r = check("I sent it. I sent you the confirmation email.", new Ledger(), { knownTools: ["send_email"] });
    expect(r.verdict).toBe("fabricated");
    expect(r.claimedButNeverRan.map((c) => [c.text, c.tool, c.reason])).toEqual([["I sent it", "send_email", "not_called"]]);
  });

  it("verb stem naming the tool object is not called", () => {
    for (const stem of ["verb_stem_refunded_order", "verb_stem_refunded_you"]) {
      const r = run(byStem.get(stem)!);
      expect(r.claimedButNeverRan.map((c) => [c.tool, c.reason]), stem).toEqual([["issue_refund", "not_called"]]);
    }
  });

  it("unmatched claims stay unmatched when diffed again", () => {
    const c = byStem.get("sibling_raised_with_team")!;
    const ledger = Ledger.fromDict(c.ledger);
    const r = check(c.text, ledger, { knownTools: c.known_tools });
    const again = diff(r.unmatched, ledger);
    expect(again.claimedButNeverRan).toEqual([]);
    expect(again.unmatched.length).toBe(1);
  });
});
