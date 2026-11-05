/** `fixtures/llm`: recorded or hand-written model output replayed through LLMExtractor. */
import { describe, expect, it } from "vitest";

import { normalizeToolSpecs } from "../src/claims";
import { Ledger } from "../src/ledger";
import { LLMExtractor } from "../src/llm";
import { check } from "../src/receipts";
import { loadCases } from "./helpers";

type Case = {
  reply: string;
  known_tools: Array<{ name: string; description: string }>;
  llm_output: string;
  ledger?: Record<string, unknown>[];
  expected: Record<string, unknown>[];
};
const CASES = loadCases<Case>("llm");
const COMPARED = ["verb", "object", "count", "text", "candidates", "confidence", "source"];
const extractor = (c: Case) => new LLMExtractor({ complete: () => c.llm_output });

describe("llm fixtures", () => {
  it("coverage", () => {
    const stems = CASES.map(([s]) => s);
    for (let n = 1; n <= 6; n++) expect(stems.some((s) => s.startsWith(`guard${n}_`)), String(n)).toBe(true);
    expect(stems.filter((s) => s.startsWith("merge_")).length).toBeGreaterThanOrEqual(4);
    expect(stems.filter((s) => s.startsWith("dedupe_")).length).toBeGreaterThanOrEqual(2);
  });

  it.each(CASES)("%s", (_stem, c) => {
    const claims = extractor(c).extract(c.reply, normalizeToolSpecs(c.known_tools));
    expect(claims.length, JSON.stringify(claims.map((x) => x.toDict()))).toBe(c.expected.length);
    claims.forEach((claim, i) => {
      const dict = claim.toDict();
      for (const [k, v] of Object.entries(c.expected[i]!)) if (COMPARED.includes(k)) expect(dict[k], k).toEqual(v);
    });
  });

  it.each(CASES)("%s: ok and fabricated agree with the rules", (_stem, c) => {
    const names = c.known_tools.map((t) => t.name);
    const ledgers = c.ledger
      ? [Ledger.fromDict(c.ledger)]
      : [new Ledger(), Ledger.fromDict(names.map((tool) => ({ tool, status: "ok" }))), Ledger.fromDict(names.map((tool) => ({ tool, status: "error", error: "boom" })))];
    for (const ledger of ledgers) {
      const withLlm = check(c.reply, ledger, { knownTools: c.known_tools, extractor: extractor(c) });
      const rules = check(c.reply, ledger, { knownTools: c.known_tools });
      expect(withLlm.ok, withLlm.summary).toBe(rules.ok);
      expect(withLlm.verdict === "fabricated").toBe(rules.verdict === "fabricated");
      expect(withLlm.claimedButNeverRan.map((x) => x.toDict())).toEqual(rules.claimedButNeverRan.map((x) => x.toDict()));
    }
  });

  it("a family-alone rule claim still blocks beside an LLM claim", () => {
    const c = new Map(CASES).get("merge_family_alone_keeps_both")!;
    const ledger = Ledger.fromDict([{ tool: "create_ticket", status: "error", error: "boom" }]);
    const r = check(c.reply, ledger, { knownTools: c.known_tools, extractor: extractor(c) });
    expect(r.verdict).toBe("fabricated");
    expect(r.claimedButNeverRan.map((x) => [x.tool, x.reason, x.source])).toEqual([["create_ticket", "tool_failed", "rules"]]);
    expect(r.suspected.map((x) => [x.tool, x.source])).toEqual([["escalate_ticket", "llm"]]);
  });
});
