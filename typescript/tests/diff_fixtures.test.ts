import { describe, expect, it } from "vitest";

import { Claim } from "../src/claims";
import { diff } from "../src/diff";
import { Ledger } from "../src/ledger";
import { type DiffCase, loadCases } from "./helpers";

const CASES = loadCases<DiffCase>("diff");
const REQUIRED = [
  "fabricated_tool_failed", "fabricated_not_called", "clean_all_confirmed",
  "unreported_lookup", "tie_resolved_by_ledger", "unmatched_only_is_clean",
  "two_fabricated_plural_summary", "ignore_tools_hides_lookup",
  "error_and_ok_same_tool_is_confirmed", "llm_suspected_not_called",
  "llm_suspected_tool_failed", "llm_blocks_fabricated", "llm_blocks_tool_failed",
  "llm_blocks_sibling_demoted", "llm_default_sibling_stays_suspected",
  "llm_confirmed_clears_unreported", "llm_ignore_tools_unmatched", "llm_row2_verb_keeps_llm_path",
  "args_mismatch_rule_claim", "args_mismatch_llm_claim_suspected",
  "args_mismatch_llm_claim_blocks", "args_mismatch_tool_stays_unreported",
  "args_mismatch_behind_not_called", "args_check_off", "observed_is_none_elsewhere",
];
const BUCKETS = {
  confirmed: "confirmed",
  claimed_but_never_ran: "claimedButNeverRan",
  unmatched: "unmatched",
  suspected: "suspected",
} as const;

describe("diff fixtures", () => {
  it("required fixtures present", () => {
    const stems = new Set(CASES.map(([s]) => s));
    for (const r of REQUIRED) expect(stems, r).toContain(r);
  });

  it.each(CASES)("%s", (_stem, c) => {
    const claims = c.claims.map((d) => Claim.fromDict(d));
    const ledger = Ledger.fromDict(c.ledger);
    const r = diff(claims, ledger, {
      knownTools: c.known_tools,
      ignoreTools: c.ignore_tools,
      llmBlocks: c.llm_blocks ?? false,
      checkArgs: c.check_args ?? true,
    });
    const exp = c.expected;
    expect(r.verdict).toBe(exp.verdict);
    expect(r.ok).toBe(exp.ok);
    expect(r.summary).toBe(exp.summary);
    for (const [key, prop] of Object.entries(BUCKETS)) {
      const got = r[prop];
      const want = (exp[key] as Record<string, unknown>[] | undefined) ?? [];
      expect(got.length, `${key}: ${JSON.stringify(got.map((cl) => cl.toDict()))}`).toBe(want.length);
      got.forEach((claim, i) => {
        const dict = claim.toDict();
        for (const [k, v] of Object.entries(want[i]!)) expect(dict[k], `${key}.${k} of ${JSON.stringify(dict)}`).toEqual(v);
      });
    }
    expect(r.ranButUnreported.map((e) => e.tool)).toEqual(exp.ran_but_unreported);
  });
});
