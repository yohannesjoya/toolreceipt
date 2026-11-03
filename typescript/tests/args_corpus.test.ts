/** The argument check over every recorded demo run (PRD 04 §10 gate), with the demo's tools. */
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { Ledger } from "../src/ledger";
import { check } from "../src/receipts";
import { DEMO, demoTools, listJson, readJson } from "./helpers";

type Replay = { reply: string; ledger: Record<string, unknown>[]; mode?: string };
const REPLAYS = listJson(join(DEMO, "replays"));
const EXAMPLES = listJson(join(DEMO, "examples"));
const CORPUS = [...REPLAYS, ...EXAMPLES];
const TOOLS = demoTools();

describe("demo corpus", () => {
  it("size", () => {
    expect(REPLAYS.length).toBeGreaterThanOrEqual(19);
    expect(EXAMPLES.length).toBeGreaterThanOrEqual(6);
  });

  it.each(CORPUS.map((p) => [p.split("/").slice(-2).join("/"), p] as const))("%s has no args_mismatch", (_id, path) => {
    const data = readJson<Replay>(path);
    const r = check(data.reply, Ledger.fromDict(data.ledger), { knownTools: TOOLS });
    const flagged = [...r.claimedButNeverRan, ...r.suspected].filter((c) => c.reason === "args_mismatch");
    expect(flagged.map((c) => c.toDict())).toEqual([]);
  });

  it.each(REPLAYS.filter((p) => p.includes("pressure-")).map((p) => [p.split("/").pop()!, p] as const))(
    "%s stays fabricated",
    (_id, path) => {
      const data = readJson<Replay>(path);
      const r = check(data.reply, Ledger.fromDict(data.ledger), { knownTools: TOOLS });
      expect(r.verdict, r.summary).toBe("fabricated");
    },
  );
});
