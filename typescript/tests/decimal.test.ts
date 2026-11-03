import { describe, expect, it } from "vitest";

import { Dec } from "../src/decimal";

const d = (s: string): Dec => {
  const v = Dec.parse(s);
  if (v === null) throw new Error(`unparseable ${s}`);
  return v;
};

describe("Dec", () => {
  it("compares by value across scales", () => {
    expect(d("64").equals(d("64.00"))).toBe(true);          // Decimal("64") == Decimal("64.00")
    expect(d("64").equals(d("6400"))).toBe(false);
    expect(d("0.1").add(d("0.2")).equals(d("0.3"))).toBe(true);
    expect(d("0.64").times100().equals(d("64"))).toBe(true);
    expect(d("-64").abs().equals(d("64"))).toBe(true);
  });

  it("parses what Decimal(str(value)) accepts and rejects the rest", () => {
    expect(Dec.parse("1,250")).toBeNull();                   // commas are removed by the caller
    expect(d("1e-7").equals(d("0.0000001"))).toBe(true);     // Decimal("1e-7") == Decimal("0.0000001")
    expect(d("1e+21").equals(Dec.fromNumber(1e21)!)).toBe(true);
    expect(d(".5").equals(d("0.5"))).toBe(true);
    expect(Dec.parse("abc")).toBeNull();
    expect(Dec.parse("")).toBeNull();
    expect(Dec.parse("1.")).toBeNull();
    expect(Dec.parse("1.2.3")).toBeNull();
  });

  it("fromNumber uses String(n), null for non-finite", () => {
    expect(Dec.fromNumber(32.1)!.equals(d("32.1"))).toBe(true);   // Decimal(str(32.1))
    expect(Dec.fromNumber(32)!.equals(d("32.00"))).toBe(true);
    expect(Dec.fromNumber(1e-7)!.equals(d("0.0000001"))).toBe(true);
    expect(Dec.fromNumber(NaN)).toBeNull();
    expect(Dec.fromNumber(Infinity)).toBeNull();
    expect(Dec.fromNumber(-Infinity)).toBeNull();
  });
});
