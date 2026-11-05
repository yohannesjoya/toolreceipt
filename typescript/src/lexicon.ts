/** Lexicon loading: the bundled copy of python/toolreceipt/lexicon.json. */

import data from "./lexicon.json";

export type Family = { forms: string[]; tokens: string[] };
export type Lexicon = { families: Record<string, Family>; [key: string]: unknown };

const DEFAULT: Lexicon = data;

/** The shared default lexicon. Never mutate it; use loadLexicon() for a copy. */
export function defaultLexicon(): Lexicon {
  return DEFAULT;
}

/** A fresh deep copy of the bundled lexicon, free to mutate. */
export function loadLexicon(): Lexicon {
  return structuredClone(DEFAULT);
}
