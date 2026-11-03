/** Held-out set: live honest-mode replies and hand-written adversarial negatives. */
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { RuleExtractor, normalizeToolSpecs } from "../src/claims";
import { Ledger } from "../src/ledger";
import { check } from "../src/receipts";
import { FIXTURES, type HeldoutCase, readJson } from "./helpers";

const NEGATIVES = readJson<HeldoutCase[]>(join(FIXTURES, "heldout", "adversarial_negatives.json"));
const HONEST = readJson<HeldoutCase[]>(join(FIXTURES, "heldout", "honest_replies.json"));

function extract(c: HeldoutCase) {
  const claims = new RuleExtractor().extract(c.text, normalizeToolSpecs(c.known_tools));
  expect(claims.every((cl) => cl.source === "rules")).toBe(true);
  return claims;
}

describe("held-out", () => {
  it("sizes", () => {
    expect(NEGATIVES.length).toBeGreaterThanOrEqual(12);
    expect(HONEST.length).toBeGreaterThanOrEqual(10);
    expect(NEGATIVES.every((c) => c.expected.length === 0)).toBe(true);
  });

  it.each(NEGATIVES.map((c) => [c.id, c] as const))("negative %s produces no claim", (_id, c) => {
    expect(extract(c).map((cl) => cl.toDict())).toEqual([]);
  });

  it("honest reply precision", () => {
    let total = 0;
    let correct = 0;
    for (const c of HONEST) {
      const ran = new Set((c.ledger ?? []).filter((e) => e["status"] === "ok").map((e) => e["tool"] as string));
      for (const claim of extract(c)) {
        total++;
        if (claim.candidates.some((t) => ran.has(t))) correct++;
      }
    }
    const labelled = HONEST.reduce((n, c) => n + c.expected.length, 0);
    console.log(
      `held-out honest replies: ${HONEST.length} replies, ${labelled} labelled claims, ${total} extracted, ` +
        `${correct} backed by a successful tool call; precision ${(correct / total).toFixed(2)}`,
    );
    expect(total).toBeGreaterThan(0);
  });

  it.each(HONEST.map((c) => [c.id, c] as const))("honest %s is never fabricated", (_id, c) => {
    const r = check(c.text, Ledger.fromDict(c.ledger ?? []), { knownTools: c.known_tools });
    expect(r.ok, r.summary).toBe(true);
  });
});
