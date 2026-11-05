/**
 * The settled-entry reader over every recorded honest and dishonest reply (PRD 05 §10). Each
 * held-out reply, demo replay and demo example is rebuilt as a one-turn transcript: same verdict
 * and claims as check(), and the honest ones are ok.
 */
import { basename, join } from "node:path";
import { describe, expect, it } from "vitest";

import type { Claim } from "../src/claims";
import { Ledger } from "../src/ledger";
import { check } from "../src/receipts";
import { checkTurns } from "../src/turns";
import { DEMO, FIXTURES, type HeldoutCase, demoTools, listJson, readJson } from "./helpers";

type Row = [id: string, user: string, reply: string, ledger: Record<string, unknown>[], tools: never, honest: boolean];
type Replay = { reply: string; ledger: Record<string, unknown>[]; customer_message?: string };
const HONEST_EXAMPLES = new Set(["admits", "clean", "outcome", "unmatched", "vague"]);
const TOOLS = demoTools() as never;

const CORPUS: Row[] = [
  ...readJson<HeldoutCase[]>(join(FIXTURES, "heldout", "honest_replies.json")).map(
    (c): Row => [c.id, "", c.text, c.ledger ?? [], c.known_tools as never, true],
  ),
  ...[...listJson(join(DEMO, "replays")), ...listJson(join(DEMO, "examples"))].map((p): Row => {
    const data = readJson<Replay>(p);
    const stem = basename(p, ".json");
    return [stem, data.customer_message ?? "", data.reply, data.ledger, TOOLS, stem.startsWith("honest-") || HONEST_EXAMPLES.has(stem)];
  }),
];

const key = (claims: Claim[]) => claims.map((c) => [c.tool, c.reason, c.text]);

describe("turns corpus", () => {
  it("counts", () => {
    expect(CORPUS.length).toBe(16 + 19 + 6);
  });

  it.each(CORPUS)("%s", (_id, user, reply, ledger, tools, honest) => {
    const transcript = [{ role: "user", content: user }, ...ledger, { role: "assistant", content: reply }];
    const [r] = checkTurns(transcript, { knownTools: tools });
    const ref = check(reply, Ledger.fromDict(ledger), { knownTools: tools });
    expect([r!.verdict, key(r!.confirmed), key(r!.claimedButNeverRan)]).toEqual([ref.verdict, key(ref.confirmed), key(ref.claimedButNeverRan)]);
    if (honest) expect(r!.ok, r!.summary).toBe(true);
  });
});
