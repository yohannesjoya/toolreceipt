/** Small pins for the extractor internals; the fixtures and the parity recording are the real test. */
import { describe, expect, it } from "vitest";

import { extractObject, phrase, scan, splitClauses, splitSentences, subjectKind } from "../src/claims";
import { defaultLexicon } from "../src/lexicon";
import { pyRegex } from "../src/pystr";

const LEX = defaultLexicon();

describe("splitting", () => {
  it("splits sentences on Python line boundaries (Review Focus 1)", () => {
    expect(splitSentences("I sent it.\x85I refunded you.")).toEqual(["I sent it.", "I refunded you."]);
  });
  it("strips bullets", () => {
    expect(splitSentences("- a\n2) b\n• c")).toEqual(["a", "b", "c"]);
  });
  it("splits clauses on punctuation and conjunctions", () => {
    expect(splitClauses("I sent it, and I refunded you; then I logged it")).toEqual(["I sent it", "I refunded you", "I logged it"]);
  });
});

describe("phrase", () => {
  it("compiles hyphenated and contraction forms under the u flag (Review Focus 2)", () => {
    expect(pyRegex(phrase("e-mailed"), "i").test("i e-mailed you")).toBe(true);
    expect(pyRegex(phrase("n't")).test("haven't")).toBe(true);
    expect(pyRegex(phrase("sent")).test("consent")).toBe(false);
  });
});

describe("scan and subject", () => {
  it("finds the completed action with its subject and auxiliaries", () => {
    const a = scan("I have sent the email", LEX)!;
    expect([a.family, a.subject, a.aux]).toEqual(["send", ["i"], ["have"]]);
  });
  it("skips adjectival participles and exclusions that come first", () => {
    expect(scan("your scheduled call", LEX)).toBeNull();
    expect(scan("I checked and sent it", LEX)).toBeNull();
  });
  it("classifies the subject", () => {
    const other = scan("The customer has cancelled", LEX)!;
    expect(subjectKind(other, "The customer has cancelled", LEX)).toBe("other");
    const headline = scan("Refund issued", LEX)!;
    expect(subjectKind(headline, "Refund issued", LEX)).toBe("passive");
    const state = scan("you were charged", LEX)!;
    expect(subjectKind(state, "you were charged", LEX)).toBe("state");
  });
});

describe("extractObject", () => {
  it("returns the object phrase and a count", () => {
    const clause = "I sent the two emails to you";
    expect(extractObject(clause, clause.indexOf("sent") + 4, LEX)).toEqual(["two emails to you", 2]);
    expect(extractObject("Deleted 3 rows", 7, LEX)).toEqual(["3 rows", 3]);
    expect(extractObject("I sent it.", 6, LEX)).toEqual(["it", null]);
  });
});
