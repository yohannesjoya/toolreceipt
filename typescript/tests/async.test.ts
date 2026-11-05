/** acheck(): the async twin of check() (python/tests/test_async.py, the extractor cases). */
import { describe, expect, it } from "vitest";

import { Claim, ToolSpec, type ClaimExtractor } from "../src/claims";
import { Ledger } from "../src/ledger";
import { acheck, check } from "../src/receipts";
import { type CheckCase, loadCases } from "./helpers";

const CASES = loadCases<CheckCase>("check");

describe("acheck", () => {
  it.each(CASES)("equals check on %s", async (_stem, c) => {
    const ledger = Ledger.fromDict(c.ledger);
    const sync = check(c.text, ledger, { knownTools: c.known_tools });
    const a = await acheck(c.text, ledger, { knownTools: c.known_tools });
    expect(a.toDict()).toEqual(sync.toDict());
  });

  it("propagates extractor errors", async () => {
    const boom: ClaimExtractor = {
      extract: () => [],
      aextract: () => Promise.reject(new Error("boom")),
    };
    await expect(acheck("I sent it.", new Ledger(), { knownTools: ["send_email"], extractor: boom })).rejects.toThrow("boom");
  });

  it("uses extract when there is no aextract", async () => {
    const only: ClaimExtractor = {
      extract: () => [new Claim({ text: "I sent it", verb: "send", candidates: ["send_email"] })],
    };
    const r = await acheck("I sent it.", new Ledger(), { knownTools: [new ToolSpec("send_email", "")], extractor: only });
    expect(r.claimedButNeverRan.map((c) => c.tool)).toEqual(["send_email"]);
  });

  // Python ignores a sync `aextract` (inspect.iscoroutinefunction); JS cannot tell an async
  // function from a sync one returning a promise, so the TS rule awaits any `aextract`
  // (PRD 06 §7.7). python/tests/test_async.py::test_sync_aextract_is_ignored is not ported.
  it("awaits any aextract function (TS rule)", async () => {
    const odd = {
      extract: () => [new Claim({ text: "I sent it", verb: "send", candidates: ["send_email"] })],
      aextract: () => [] as Claim[],
    } as unknown as ClaimExtractor;
    const r = await acheck("I sent it.", new Ledger(), { knownTools: [new ToolSpec("send_email", "")], extractor: odd });
    expect(r.claimedButNeverRan).toEqual([]);
  });
});
