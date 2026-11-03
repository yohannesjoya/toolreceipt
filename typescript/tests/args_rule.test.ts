/** Argument-check pins, including the documented int/float divergence (Review Focus 3). */
import { describe, expect, it } from "vitest";

import { findAmount, findReference } from "../src/args";
import { Claim } from "../src/claims";
import { Dec } from "../src/decimal";
import { diff } from "../src/diff";
import { Ledger } from "../src/ledger";

describe("findAmount", () => {
  it("one cued amount, qualified by its neighbours", () => {
    expect(findAmount("I've refunded $64")!.equals(Dec.parse("64")!)).toBe(true);
    expect(findAmount("half of your $64 charge")).toBeNull();
    expect(findAmount("$19.99 for the mug and $5.99 for shipping")).toBeNull();
    expect(findAmount("I've refunded $16 apiece")).toBeNull();
    expect(findAmount("refunded EUR 1,250.00")!.equals(Dec.parse("1250")!)).toBe(true);
    expect(findAmount("I've refunded $64", "I've refunded $64, and the $5 fee")).toBeNull();
  });
});

describe("findReference", () => {
  it("one cued reference number", () => {
    expect(findReference("cancelled order 4821", ["order"])).toEqual(["order", "4821"]);
    expect(findReference("ticket #5521 and case 77", ["ticket", "case"])).toEqual(["ticket", "5521"]);
    expect(findReference("on 2024-01-15", ["order"])).toBeNull();
    expect(findReference("order 4821 and ticket 5521", ["order", "ticket"])).toBeNull();
    expect(findReference("tracking 9400 1118 9922", ["tracking"])).toEqual(["tracking", "940011189922"]);
  });
});

describe("whole-number floats in ledger arguments (accepted divergence, PRD 06 §7.3 item 8)", () => {
  // Python: 4821.0 is a float, `_reference_runs` returns None, the claim is confirmed either
  // way. TS: 4821.0 parses to the integer 4821, so the reference is judged.
  const claim = () => {
    const c = new Claim({ text: "I've cancelled order 4821", verb: "delete", candidates: ["cancel_order"], confidence: "high" });
    c.setEvidence({ objectMatched: true });
    return c;
  };
  it("judges 4821.0 as an int", () => {
    const same = diff([claim()], Ledger.fromDict([{ tool: "cancel_order", status: "ok", args: { order_id: 4821.0 } }]));
    expect(same.verdict).toBe("clean");
    const other = diff([claim()], Ledger.fromDict([{ tool: "cancel_order", status: "ok", args: { order_id: 4822.0 } }]));
    expect(other.verdict).toBe("fabricated");
    expect(other.claimedButNeverRan[0]!.reason).toBe("args_mismatch");
    expect(other.claimedButNeverRan[0]!.observed).toEqual({ order_id: 4822 });
  });
  it("never judges a non-integer number as a reference (Python float)", () => {
    const r = diff([claim()], Ledger.fromDict([{ tool: "cancel_order", status: "ok", args: { order_id: 4822.5 } }]));
    expect(r.verdict).toBe("clean");
  });
});
