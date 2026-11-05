import { describe, expect, it } from "vitest";

import { RuleExtractor, normalizeToolSpecs } from "../src/claims";
import { type ExtractorCase, loadCases } from "./helpers";

const CASES = loadCases<ExtractorCase>("extractor");
// Keys compared when present in a fixture's expected claim (python/tests/test_extractor_fixtures.py).
const COMPARED = ["verb", "object", "count", "text", "candidates", "confidence", "source"];

describe("extractor fixtures", () => {
  it("fixture count", () => {
    expect(CASES.length).toBeGreaterThanOrEqual(24);
  });

  it.each(CASES)("%s", (_stem, c) => {
    const claims = new RuleExtractor().extract(c.text, normalizeToolSpecs(c.known_tools));
    expect(claims.every((cl) => cl.source === "rules")).toBe(true);
    expect(claims.length, JSON.stringify(claims.map((cl) => cl.toDict()))).toBe(c.expected.length);
    claims.forEach((claim, i) => {
      const exp = c.expected[i]!;
      const dict = claim.toDict();
      for (const [key, value] of Object.entries(exp)) {
        if (COMPARED.includes(key)) expect(dict[key], `${key} of ${JSON.stringify(dict)}`).toEqual(value);
      }
    });
  });

  it.each(CASES.filter(([stem]) => stem.startsWith("neg_")))("%s is negative", (_stem, c) => {
    expect(c.expected).toEqual([]);
  });

  it("extract fills candidates but leaves diff fields empty", () => {
    const claims = new RuleExtractor().extract("I sent the email.", normalizeToolSpecs(["send_email"]));
    expect(claims.map((c) => [c.candidates, c.confidence, c.tool, c.reason])).toEqual([[["send_email"], "high", null, null]]);
  });
});
