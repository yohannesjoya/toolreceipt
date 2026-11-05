/**
 * Inputs the fixtures do not hold, pinned against CPython's result for the same call.
 * Expected values come from running the Python package (toolreceipt 0.4.0).
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import {
  RuleExtractor, descriptionFamily, descriptionObject, normalizeToolSpecs, phrase, type ToolLike,
} from "../src/claims";
import { Ledger, LedgerEntry } from "../src/ledger";
import { loadLexicon } from "../src/lexicon";
import { pyRegex } from "../src/pystr";
import { check } from "../src/receipts";
import { ROOT } from "./helpers";

describe("package shape", () => {
  it("gives each module system its own type declarations", () => {
    const pkg = JSON.parse(readFileSync(join(ROOT, "typescript", "package.json"), "utf8")) as {
      exports: Record<string, unknown>;
    };
    expect(pkg.exports["."]).toEqual({
      import: { types: "./dist/index.d.ts", default: "./dist/index.js" },
      require: { types: "./dist/index.d.cts", default: "./dist/index.cjs" },
    });
  });
});

describe("release", () => {
  const pkg = JSON.parse(readFileSync(join(ROOT, "typescript", "package.json"), "utf8")) as {
    repository: { url: string };
    dependencies?: Record<string, string>;
    scripts: Record<string, string>;
  };

  it("names the repository in npm's normalised form", () => {
    expect(pkg.repository.url).toBe("git+https://github.com/yohannesjoya/toolreceipt.git");
  });

  it("cannot hand-publish an unbuilt or untested package", () => {
    expect(pkg.scripts["prepublishOnly"]).toBe("npm run check && npm run smoke");
  });

  it("has no runtime dependency", () => {
    expect(pkg.dependencies).toBeUndefined();
  });

  it("publishes from a tag the Python release does not match", () => {
    const workflow = readFileSync(join(ROOT, ".github", "workflows", "release-npm.yml"), "utf8");
    expect(workflow).toContain('tags: ["ts-v*"]');
    expect(workflow).toContain("git merge-base --is-ancestor");
    expect(readFileSync(join(ROOT, ".github", "workflows", "release.yml"), "utf8")).toContain('tags: ["v*"]');
  });
});

describe("release jobs", () => {
  const workflow = readFileSync(join(ROOT, ".github", "workflows", "release-npm.yml"), "utf8");
  const [build, publish] = workflow.split(/^ {2}# Publishes[^\n]*$/m) as [string, string];

  it("keeps the publish credential away from dependency code", () => {
    expect(build).not.toContain("id-token");
    expect(publish).toContain("id-token: write");
    expect(publish).not.toContain("npm ci");
    expect(publish).not.toContain("actions/checkout");
  });

  it("pins what runs beside the credential", () => {
    for (const line of publish.split("\n").filter((l) => l.includes("uses: "))) expect(line).toMatch(/@[0-9a-f]{40} /);
    expect(publish).toMatch(/npm install -g npm@\d+\.\d+\.\d+/);
  });
});

describe("README quickstart", () => {
  it("prints what the README says", () => {
    const ledger = Ledger.fromDict([{ tool: "issue_refund", status: "ok", args: { amount: 32 } }]);
    const receipt = check("I've refunded $64 and emailed you the confirmation.", ledger, {
      knownTools: [["issue_refund", "Issue a refund to the customer"], "send_email"],
    });
    const summary = '2 claimed actions never ran: issue_refund ("I\'ve refunded $64", ledger amount=32).';
    expect(receipt.verdict).toBe("fabricated");
    expect(receipt.ok).toBe(false);
    expect(receipt.summary).toBe(summary);
    expect(readFileSync(join(ROOT, "typescript", "README.md"), "utf8")).toContain(summary);
  });
});

describe("LedgerEntry.fromDict", () => {
  it("keeps status null, so the entry is neither ok nor error", () => {
    const ledger = Ledger.fromDict([{ tool: "send_email", status: null }]);
    expect(ledger.okTools().size).toBe(0);
    expect(ledger.errorTools().size).toBe(0);
    expect(ledger.toDict()[0]!["status"]).toBeNull();
    const r = check("I sent the email.", ledger, { knownTools: ["send_email"] });
    expect(r.verdict).toBe("fabricated");
    expect(r.summary).toBe('1 claimed action never ran: send_email ("I sent the email").');
  });

  it("defaults a missing status to ok", () => {
    expect(LedgerEntry.fromDict({ tool: "x" }).status).toBe("ok");
    expect(LedgerEntry.fromDict({ tool: "x", status: undefined }).status).toBe("ok");
  });

  it.each([[[1, 2]], ["abc"], [5], [true]])("rejects args %j", (args) => {
    expect(() => LedgerEntry.fromDict({ tool: "x", args })).toThrow(TypeError);
  });

  it.each([[[]], [""], [0], [false], [null], [undefined]])("reads falsy args %j as empty", (args) => {
    expect(LedgerEntry.fromDict({ tool: "x", args }).args).toEqual({});
  });

  it("copies a mapping", () => {
    const args = { a: 1 };
    const entry = LedgerEntry.fromDict({ tool: "x", args });
    expect(entry.args).toEqual({ a: 1 });
    expect(entry.args).not.toBe(args);
    expect(LedgerEntry.fromDict({ tool: "x", args: new Map([["a", 1]]) }).args).toEqual({ a: 1 });
  });
});

describe("normalizeToolSpecs", () => {
  it.each([
    [{ function: {}, name: "send_email" }, [["send_email", ""]]],
    [{ function: {}, custom: { name: "send_email" } }, [["send_email", ""]]],
    [{ function: {}, custom: {}, name: "send_email" }, []],
    [{ name: "d", description: {} }, [["d", ""]]],
    [{ function: "", name: "e" }, [["e", ""]]],
  ] as Array<[ToolLike, string[][]]>)("reads %j with truthy semantics", (tool, expected) => {
    expect(normalizeToolSpecs([tool]).map((s) => [s.name, s.description])).toEqual(expected);
  });

  it.each([[null, "null"], [undefined, "undefined"]])("names the type of %s", (tool, word) => {
    expect(() => normalizeToolSpecs([tool as unknown as ToolLike])).toThrow(TypeError);
    expect(() => normalizeToolSpecs([tool as unknown as ToolLike])).toThrow(new RegExp(`got ${word}$`));
  });
});

describe("two-word lexicon forms", () => {
  it("split on any whitespace", () => {
    const lex = loadLexicon();
    lex.families["create"]!.forms.push("stand  up");
    expect(descriptionFamily("Stand up a server", lex)).toBe("create");
    expect(descriptionObject("Stand up a server", lex)).toEqual(["server"]);
  });
});

describe("plan review focus, end to end", () => {
  it("splits sentences on NEL through extract", () => {
    const claims = new RuleExtractor().extract("I sent it.\x85I refunded you.", normalizeToolSpecs(["send_email", "issue_refund"]));
    expect(claims.map((c) => [c.verb, c.candidates])).toEqual([["send", ["send_email"]], ["refund", ["issue_refund"]]]);
  });

  it("compiles a phrase holding a space under the u flag", () => {
    expect(pyRegex(phrase("set up")).test("i set  up it")).toBe(true);
    expect(pyRegex(phrase("set up")).test("i reset up")).toBe(false);
  });
});

describe("lexicon without argument keys", () => {
  it("still extracts", () => {
    const lex = loadLexicon();
    delete lex["amount_keys"];
    delete lex["id_cues"];
    const claims = new RuleExtractor(lex).extract("I've refunded $64.", normalizeToolSpecs(["issue_refund"]));
    expect(claims.map((c) => c.candidates)).toEqual([["issue_refund"]]);
  });
});
