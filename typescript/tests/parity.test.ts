/**
 * Whole-object parity with the Python package (PRD 06 §10.2): every extract(), diff(),
 * check(), message-list and per-turn output recorded by scripts/record_parity.py must equal the
 * TypeScript output.
 *
 * Compared as JSON text: same keys, same order, `null` never `undefined` (PRD 06 §13).
 */
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { Claim, RuleExtractor, normalizeToolSpecs } from "../src/claims";
import { diff } from "../src/diff";
import { Ledger } from "../src/ledger";
import { checkMessages, finalReply, ledgerFromMessages } from "../src/messages";
import { check } from "../src/receipts";
import { checkTurns } from "../src/turns";
import {
  type CheckCase, DEMO, type DiffCase, type ExtractorCase, FIXTURES, type HeldoutCase, demoTools, readJson,
} from "./helpers";

type Recording = {
  toolreceipt: string;
  recorded_at: string;
  extract: Array<{ source: string; claims: unknown[] }>;
  diff: Array<{ source: string; receipt: unknown }>;
  check: Array<{ source: string; receipt: unknown }>;
  messages: Array<{ source: string; ledger: unknown; final_reply: string; receipt: unknown }>;
  turns: Array<{ source: string; receipts: unknown }>;
};
const REC = readJson<Recording>(join(FIXTURES, "recorded", "parity.json"));
const TOOLS = demoTools();

function heldout(file: string, id: string): HeldoutCase {
  const cases = readJson<HeldoutCase[]>(join(FIXTURES, "heldout", `${file}.json`));
  const found = cases.find((c) => c.id === id);
  if (!found) throw new Error(`no held-out case ${file}#${id}`);
  return found;
}

/** Resolve a recording `source` to its inputs. */
function inputs(source: string): { text: string; known_tools: unknown[]; ledger: Record<string, unknown>[] } {
  const slash = source.indexOf("/");
  const dir = source.slice(0, slash);
  const rest = source.slice(slash + 1);
  if (dir === "extractor" || dir === "check") {
    const c = readJson<ExtractorCase & Partial<CheckCase>>(join(FIXTURES, dir, `${rest}.json`));
    return { text: c.text, known_tools: c.known_tools, ledger: c.ledger ?? [] };
  }
  if (dir === "heldout") {
    const [file, id] = rest.split("#") as [string, string];
    const c = heldout(file, id);
    return { text: c.text, known_tools: c.known_tools, ledger: c.ledger ?? [] };
  }
  if (dir === "demo") {
    const data = readJson<{ reply: string; ledger: Record<string, unknown>[] }>(join(DEMO, `${rest}.json`));
    return { text: data.reply, known_tools: TOOLS, ledger: data.ledger };
  }
  throw new Error(`unknown source ${source}`);
}

describe("parity with the Python recording", () => {
  it("recording is for the Python version the fixtures were written against", () => {
    expect(REC.toolreceipt).toMatch(/^\d+\.\d+\.\d+$/);
    expect(REC.extract.length).toBeGreaterThan(100);
    expect(REC.diff.length).toBeGreaterThanOrEqual(25);
    expect(REC.check.length).toBeGreaterThan(100);
    expect(REC.messages.length).toBeGreaterThanOrEqual(29);
    expect(REC.turns.length).toBeGreaterThanOrEqual(31);
  });

  it.each(REC.extract.map((e) => [e.source, e] as const))("extract %s", (_source, e) => {
    const { text, known_tools } = inputs(e.source);
    const claims = new RuleExtractor().extract(text, normalizeToolSpecs(known_tools as never));
    expect(JSON.stringify(claims.map((c) => c.toDict()))).toBe(JSON.stringify(e.claims));
  });

  it.each(REC.diff.map((e) => [e.source, e] as const))("diff %s", (_source, e) => {
    const stem = e.source.slice("diff/".length);
    const c = readJson<DiffCase>(join(FIXTURES, "diff", `${stem}.json`));
    const r = diff(c.claims.map((d) => Claim.fromDict(d)), Ledger.fromDict(c.ledger), {
      knownTools: c.known_tools,
      ignoreTools: c.ignore_tools,
      llmBlocks: c.llm_blocks ?? false,
      checkArgs: c.check_args ?? true,
    });
    expect(JSON.stringify(r.toDict())).toBe(JSON.stringify(e.receipt));
  });

  it.each(REC.check.map((e) => [e.source, e] as const))("check %s", (_source, e) => {
    const { text, known_tools, ledger } = inputs(e.source);
    const r = check(text, Ledger.fromDict(ledger), { knownTools: known_tools as never });
    expect(JSON.stringify(r.toDict())).toBe(JSON.stringify(e.receipt));
  });

  type ListCase = { messages: unknown[]; known_tools: never };

  it.each(REC.messages.map((e) => [e.source, e] as const))("messages %s", (_source, e) => {
    const c = readJson<ListCase>(join(FIXTURES, `${e.source}.json`));
    expect(JSON.stringify(ledgerFromMessages(c.messages).toDict())).toBe(JSON.stringify(e.ledger));
    expect(finalReply(c.messages)).toBe(e.final_reply);
    expect(JSON.stringify(checkMessages(c.messages, { knownTools: c.known_tools }).toDict())).toBe(JSON.stringify(e.receipt));
  });

  it.each(REC.turns.map((e) => [e.source, e] as const))("turns %s", (_source, e) => {
    const c = readJson<ListCase>(join(FIXTURES, `${e.source}.json`));
    const got = checkTurns(c.messages, { knownTools: c.known_tools }).map((r) => r.toDict());
    expect(JSON.stringify(got)).toBe(JSON.stringify(e.receipts));
  });
});
