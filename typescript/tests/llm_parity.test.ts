/**
 * Whole-object parity for the LLM path: claims and receipts recorded from Python over every
 * `fixtures/llm` case and every recorded probe output (scripts/record_parity.py). No model is called.
 * Equal receipts carry Python's probe gates over: trap false alarms, no extra tool on honest replies.
 */
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { normalizeToolSpecs } from "../src/claims";
import { Ledger } from "../src/ledger";
import { LLMExtractor, PROMPT_V2, PROMPT_VERSION, buildPrompt } from "../src/llm";
import { check } from "../src/receipts";
import { FIXTURES, readJson } from "./helpers";

type Pair = { default: unknown; llm_blocks: unknown };
type Rec = {
  prompt_version: number; prompt: string; built: string;
  llm: Array<{ source: string; claims: unknown; receipts: Pair[] }>;
  probe: Array<{ source: string; claims: unknown } & Pair>;
};
type LlmCase = { reply: string; known_tools: never; llm_output: string | null; ledger?: Record<string, unknown>[] };
type ProbeFile = { cases: Array<LlmCase & { id: string; ledger: Record<string, unknown>[] }> };

const REC = readJson<Rec>(join(FIXTURES, "recorded", "parity_llm.json"));
const same = (a: unknown, b: unknown) => expect(JSON.stringify(a)).toBe(JSON.stringify(b));
const quiet = { warn: () => undefined };

function replay(output: string | null): LLMExtractor {
  return new LLMExtractor({
    logger: quiet,
    complete: () => {
      if (output === null) throw new Error("recorded call failed");
      return output;
    },
  });
}

function receipts(c: LlmCase, ledger: Ledger): Pair {
  const run = (llmBlocks: boolean) => check(c.reply, ledger, { knownTools: c.known_tools, extractor: replay(c.llm_output), llmBlocks }).toDict();
  return { default: run(false), llm_blocks: run(true) };
}

const PROBES = new Map<string, ProbeFile["cases"][number]>();
for (const name of ["dev", "heldout", "v01"]) {
  for (const c of readJson<ProbeFile>(join(FIXTURES, "probe", `${name}.recorded.json`)).cases) PROBES.set(`probe/${name}#${c.id}`, c);
}

describe("LLM parity with the Python recording", () => {
  it("prompt is byte-identical", () => {
    expect(PROMPT_VERSION).toBe(REC.prompt_version);
    expect(PROMPT_V2).toBe(REC.prompt);
    expect(buildPrompt("Done {reply}", normalizeToolSpecs([["send_email", "Send an email"], "issue_refund"]))).toBe(REC.built);
  });

  it("sections hold every case", () => {
    expect(REC.llm.length).toBeGreaterThanOrEqual(31);
    expect(REC.probe.length).toBe(154);
  });

  it.each(REC.llm.map((e) => [e.source, e] as const))("%s", (source, e) => {
    const c = readJson<Omit<LlmCase, "known_tools"> & { known_tools: Array<{ name: string; description: string }> }>(join(FIXTURES, `${source}.json`));
    same(replay(c.llm_output).extract(c.reply, normalizeToolSpecs(c.known_tools)).map((x) => x.toDict()), e.claims);
    const names = c.known_tools.map((t) => t.name);
    const ledgers = c.ledger ? [c.ledger] : [[], names.map((tool) => ({ tool, status: "ok" })), names.map((tool) => ({ tool, status: "error", error: "boom" }))];
    same(ledgers.map((ld) => receipts(c as unknown as LlmCase, Ledger.fromDict(ld))), e.receipts);
  });

  it.each(REC.probe.map((e) => [e.source, e] as const))("%s", (source, e) => {
    const c = PROBES.get(source)!;
    same(replay(c.llm_output).extract(c.reply, normalizeToolSpecs(c.known_tools)).map((x) => x.toDict()), e.claims);
    const got = receipts(c, Ledger.fromDict(c.ledger));
    same(got.default, e.default);
    same(got.llm_blocks, e.llm_blocks);
  });
});
