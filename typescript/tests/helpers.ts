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
