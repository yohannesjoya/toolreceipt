import { describe, expect, it } from "vitest";

import { defaultLexicon, loadLexicon } from "../src/lexicon";
import { PYTHON_LEXICON, readJson } from "./helpers";

type Lex = ReturnType<typeof loadLexicon>;

describe("lexicon", () => {
  it("bundled lexicon equals python/toolreceipt/lexicon.json", () => {
    expect(defaultLexicon()).toEqual(readJson(PYTHON_LEXICON));
  });

  it("loadLexicon returns a fresh copy", () => {
    const lex = loadLexicon();
    lex.families["send"]!.forms.push("zzz");
    expect(loadLexicon().families["send"]!.forms).not.toContain("zzz");
    expect(defaultLexicon().families["send"]!.forms).not.toContain("zzz");
  });

  // The assertions of python/tests/test_lexicon.py.
  it("has the required families and keys", () => {
    const lex: Lex = loadLexicon();
    const required = ["send", "create", "update", "delete", "refund", "charge", "schedule", "save", "submit", "trigger"];
    for (const name of required) expect(Object.keys(lex.families)).toContain(name);
    for (const fam of Object.values(lex.families)) {
      expect(fam.forms.length).toBeGreaterThan(0);
      expect(fam.tokens.length).toBeGreaterThan(0);
    }
    for (const key of ["exclusions", "negation", "future", "advice", "other_subject", "synonyms", "number_words"]) {
      expect(lex).toHaveProperty(key);
    }
  });

  it("forms do not overlap exclusions, and are unique across families", () => {
    const lex = loadLexicon();
    const forms = Object.values(lex.families).flatMap((f) => f.forms);
    const exclusions = new Set(lex["exclusions"] as string[]);
    expect(forms.filter((f) => exclusions.has(f))).toEqual([]);
    expect(new Set(forms).size).toBe(forms.length);
  });

  it("tokens are unique across families; noun_tokens are tokens; get is a read verb", () => {
    const lex = loadLexicon();
    const tokens = Object.values(lex.families).flatMap((f) => f.tokens);
    expect(new Set(tokens).size).toBe(tokens.length);
    for (const noun of lex["noun_tokens"] as string[]) expect(tokens).toContain(noun);
    expect(lex["read_verbs"] as string[]).toContain("get");
  });

  it("argument keys are present", () => {
    const lex = loadLexicon();
    expect(lex["amount_keys"]).toEqual(["amount", "total", "cents", "minor"]);
    expect(lex["id_cues"]).toEqual([
      "order", "ticket", "case", "invoice", "booking", "reservation", "account", "confirmation", "tracking", "reference",
    ]);
  });
});
