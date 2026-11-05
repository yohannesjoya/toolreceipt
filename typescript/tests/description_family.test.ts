/** Lead verb of a tool description (PRD 03 §7.2 rule 2), pinned by a committed sample. */
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { ToolSpec, descriptionFamily, descriptionObject, stemVerb, toolVerbTokens } from "../src/claims";
import { defaultLexicon, loadLexicon } from "../src/lexicon";
import { FIXTURES, readJson } from "./helpers";

type Entry = { description: string; family: string | null };
const SAMPLE = readJson<Entry[]>(join(FIXTURES, "descriptions", "sample.json"));
const LEX = defaultLexicon();

describe("description family", () => {
  it("sample size", () => {
    expect(SAMPLE.length).toBeGreaterThanOrEqual(30);
  });

  it.each(SAMPLE.map((e) => [e.description.slice(0, 30) || "<empty>", e] as const))("%s", (_id, e) => {
    expect(descriptionFamily(e.description, LEX)).toBe(e.family);
  });

  it("stemVerb third person", () => {
    const words = ["searches", "fetches", "queries", "sends", "tickets", "address", "closes", "processes"];
    expect(words.map(stemVerb)).toEqual(["search", "fetch", "query", "send", "ticket", "address", "close", "process"]);
  });

  it("lexicon without skip keys loads", () => {
    const lex = loadLexicon();
    delete lex["description_lead_skip"];
    delete lex["description_lead_nouns"];
    expect(descriptionFamily("Create a ticket", lex)).toBe("create");
    expect(descriptionFamily("Use this tool to create a ticket", lex)).toBeNull();
    expect(descriptionFamily("Email the customer", lex)).toBeNull();
  });

  it("tool verb tokens", () => {
    expect(toolVerbTokens(new ToolSpec("jira", ""), LEX)).toEqual(new Set());
    expect(toolVerbTokens(new ToolSpec("JiraV2", "Create a ticket"), LEX)).toContain("create");
    expect(toolVerbTokens(new ToolSpec("send_email", "Delete things"), LEX)).toEqual(new Set(["send", "email"]));
    expect(toolVerbTokens(new ToolSpec("slack2", "Send a message to a channel"), LEX)).toContain("send");
  });

  it.each([
    ["Create a ticket in Jira", ["ticket"]],
    ["Send an email to the customer about their order", ["email"]],
    ["Create a support ticket for the customer", ["support", "ticket"]],
    ["Jira: create a ticket", ["ticket"]],
    ["Email the customer", ["customer"]],
    ["Set up a meeting with the team", ["meeting"]],
    ["Use this tool to refund a payment.", ["payment"]],
    ["Create tickets and check their status", ["ticket"]],
    ["Create", []],
    ["", []],
  ] as Array<[string, string[]]>)("descriptionObject(%s)", (description, expected) => {
    expect(descriptionObject(description, LEX)).toEqual(expected);
  });
});
