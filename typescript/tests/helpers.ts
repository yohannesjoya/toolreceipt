/** Shared test helpers: JSON fixture loading (mirror of python/tests/helpers.py). */

import { readdirSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
export const ROOT = resolve(HERE, "..", "..");
export const FIXTURES = join(ROOT, "fixtures");
export const DEMO = join(ROOT, "demo");
export const PYTHON_LEXICON = join(ROOT, "python", "toolreceipt", "lexicon.json");

export function readJson<T = unknown>(path: string): T {
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

/** `[stem, parsed json]` for every `*.json` in `fixtures/<subdir>`, sorted by stem. */
export function loadCases<T = Record<string, unknown>>(subdir: string): Array<[string, T]> {
  const dir = join(FIXTURES, subdir);
  return readdirSync(dir)
    .filter((f) => f.endsWith(".json"))
    .sort()
    .map((f) => [f.slice(0, -5), readJson<T>(join(dir, f))]);
}

export type ToolDict = { name: string; description: string };
export type ExtractorCase = { text: string; known_tools: ToolDict[]; expected: Record<string, unknown>[] };
export type LedgerDict = Record<string, unknown>;
export type CheckCase = {
  text: string; known_tools: ToolDict[]; ledger: LedgerDict[];
  expected: { verdict: string; claimed_but_never_ran?: Record<string, unknown>[]; summary?: string };
};
export type DiffCase = {
  claims: Record<string, unknown>[]; ledger: LedgerDict[]; known_tools: string[]; ignore_tools: string[];
  llm_blocks?: boolean; check_args?: boolean;
  expected: Record<string, unknown> & { verdict: string; ok: boolean; summary: string; ran_but_unreported: string[] };
};
export type HeldoutCase = { id: string; text: string; known_tools: ToolDict[]; ledger?: LedgerDict[]; expected: unknown[] };

/** The demo support desk's tool specs (fixtures/demo/tools.json, written by scripts/record_parity.py). */
export function demoTools(): Array<[string, string]> {
  return readJson<ToolDict[]>(join(FIXTURES, "demo", "tools.json")).map((t) => [t.name, t.description]);
}

export function listJson(dir: string): string[] {
  return readdirSync(dir).filter((f) => f.endsWith(".json")).sort().map((f) => join(dir, f));
}
