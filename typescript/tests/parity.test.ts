/**
 * Whole-object parity with the Python package (PRD 06 §10.2): every extract(), diff() and
 * check() output recorded by scripts/record_parity.py must deep-equal the TypeScript output.
 */
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { Claim, RuleExtractor, normalizeToolSpecs } from "../src/claims";
import { diff } from "../src/diff";
import { Ledger } from "../src/ledger";
import { check } from "../src/receipts";
import {
  type CheckCase, DEMO, type DiffCase, type ExtractorCase, FIXTURES, type HeldoutCase, demoTools, readJson,
} from "./helpers";

type Recording = {
  toolreceipt: string;
  recorded_at: string;
  extract: Array<{ source: string; claims: unknown[] }>;
  diff: Array<{ source: string; receipt: unknown }>;
  check: Array<{ source: string; receipt: unknown }>;
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
  const [dir, rest] = source.split("/", 2) as [string, string];
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
  });

  it.each(REC.extract.map((e) => [e.source, e] as const))("extract %s", (_source, e) => {
    const { text, known_tools } = inputs(e.source);
    const claims = new RuleExtractor().extract(text, normalizeToolSpecs(known_tools as never));
    expect(claims.map((c) => c.toDict())).toEqual(e.claims);
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
    expect(r.toDict()).toEqual(e.receipt);
  });

  it.each(REC.check.map((e) => [e.source, e] as const))("check %s", (_source, e) => {
    const { text, known_tools, ledger } = inputs(e.source);
    const r = check(text, Ledger.fromDict(ledger), { knownTools: known_tools as never });
    expect(r.toDict()).toEqual(e.receipt);
  });
});
