/** Per-turn receipt fixtures (PRD 05 §10): a message list -> one receipt per assistant turn. */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import type { LedgerEntry } from "../src/ledger";
import { checkMessages } from "../src/messages";
import { checkTurns, windows } from "../src/turns";
import { ROOT, loadCases } from "./helpers";

type Want = {
  turn: number; message_index: number; text: string; verdict: string;
  claimed_but_never_ran?: Array<{ tool: string; reason: string; observed?: unknown }>;
  ran_but_unreported?: string[];
  ledger?: Array<string | Record<string, unknown>>;
  summary?: string;
};
type Case = { messages: unknown[]; known_tools: never; source?: string; raises?: unknown; expected?: Want[] };
const CASES = loadCases<Case>("turns");
const byStem = new Map(CASES);
const HONEST = CASES.filter(([s]) => s.startsWith("honest_multi_turn_")).map(([, c]) => c);
const REQUIRED = [
  "openai_abandoned_claim_no_call", "openai_claim_with_call_same_message",
  "openai_claim_then_pending_call", "openai_claim_confirmed_from_earlier_window",
  "openai_claim_before_call_in_next_window_flags", "openai_two_runs_ledger_carries",
  "openai_two_runs_new_claim_not_called", "openai_single_turn_equals_check_messages",
  "openai_args_mismatch_on_turn", "responses_abandoned_claim", "responses_two_turns",
  "anthropic_text_and_tool_use_one_turn", "anthropic_claim_then_silence",
  "transcript_silence_mid_auth", "transcript_tool_error_no_recovery",
  "transcript_claim_then_silence", "transcript_settled_entry_defaults",
  "transcript_bad_status_raises", "langchain_objects_rejected", "serialized_langchain_rejected",
  "stored_chat_message_rejected",
  "transcript_settled_entries_out_of_time_order", "transcript_settled_entry_after_claim",
  "readme_transcript_example", "transcript_empty_tool_raises",
  "transcript_silence_mid_auth_no_entry", "transcript_settled_entry_structured_result",
  "transcript_bad_timestamp_raises", "transcript_non_string_tool_raises",
  "transcript_tool_none_is_not_an_entry",
];

function entryMatches(entry: LedgerEntry, want: string | Record<string, unknown>): boolean {
  if (typeof want === "string") return entry.tool === want;
  const dict = entry.toDict() as Record<string, unknown>;
  return Object.entries(want).every(([k, v]) => JSON.stringify(dict[k]) === JSON.stringify(v));
}

describe("turns fixtures", () => {
  it("honest multi-turn fixtures exist and are honest", () => {
    expect(HONEST.length).toBeGreaterThanOrEqual(6);
    for (const c of HONEST) {
      expect(c.expected!.length).toBe(3);
      expect(c.expected!.every((r) => r.verdict !== "fabricated")).toBe(true);
      expect(c.source).toMatch(/^derived from demo\/replays\/honest-/);
    }
  });

  it("fixture schema and coverage", () => {
    for (const [, c] of CASES) {
      expect(c).toHaveProperty("messages");
      expect(c).toHaveProperty("known_tools");
      expect("expected" in c).not.toBe("raises" in c);
      for (const r of c.expected ?? []) for (const k of ["turn", "message_index", "text", "verdict"]) expect(r).toHaveProperty(k);
    }
    const stems = new Set(CASES.map(([s]) => s));
    for (const r of REQUIRED) expect(stems, r).toContain(r);
  });

  it.each(CASES)("%s", (_stem, c) => {
    if ("raises" in c) {
      expect(() => checkTurns(c.messages, { knownTools: c.known_tools })).toThrow(TypeError);
      return;
    }
    const receipts = checkTurns(c.messages, { knownTools: c.known_tools });
    const dump = JSON.stringify(receipts.map((r) => r.toDict()));
    expect(receipts.length, dump).toBe(c.expected!.length);
    receipts.forEach((r, i) => {
      const want = c.expected![i]!;
      expect([r.turn, r.messageIndex, r.text, r.verdict], dump).toEqual([want.turn, want.message_index, want.text, want.verdict]);
      expect(r.ok).toBe(want.verdict !== "fabricated");
      const never = want.claimed_but_never_ran ?? [];
      expect(r.claimedButNeverRan.map((cl) => [cl.tool, cl.reason]), dump).toEqual(never.map((cl) => [cl.tool, cl.reason]));
      never.forEach((w, j) => {
        if ("observed" in w) expect(r.claimedButNeverRan[j]!.observed).toEqual(w.observed);
      });
      expect(r.ranButUnreported.map((e) => e.tool), dump).toEqual(want.ran_but_unreported ?? []);
      if (want.ledger) {
        const entries = [...r.ledger];
        expect(entries.length, dump).toBe(want.ledger.length);
        entries.forEach((e, j) => expect(entryMatches(e, want.ledger![j]!), dump).toBe(true));
      }
      if (want.summary !== undefined) expect(r.summary).toBe(want.summary);
    });
  });

  it("single turn equals checkMessages", () => {
    const c = byStem.get("openai_single_turn_equals_check_messages")!;
    const [r] = checkTurns(c.messages, { knownTools: c.known_tools });
    const { turn: _t, message_index: _m, text: _x, ...got } = r!.toDict();
    expect(got).toEqual(checkMessages(c.messages, { knownTools: c.known_tools }).toDict());
  });

  // PRD 05 §7.4: a receipt never changes when messages are appended after its window.
  it.each(CASES.filter(([, c]) => c.expected?.length))("%s is prefix-pure", (_stem, c) => {
    const run = (msgs: unknown[]) => checkTurns(msgs, { knownTools: c.known_tools }).map((r) => JSON.stringify(r.toDict()));
    const full = run(c.messages);
    const ends = new Set(windows(c.messages).map(([, end]) => end));
    for (let j = 0; j <= c.messages.length; j++) {
      const partial = run(c.messages.slice(0, j));
      const settled = ends.has(j) || j === c.messages.length ? partial : partial.slice(0, -1);
      expect(settled, String(j)).toEqual(full.slice(0, settled.length));
    }
  });

  it("README transcript literal is the readme_transcript_example fixture", () => {
    const text = readFileSync(join(ROOT, "typescript", "README.md"), "utf8");
    const m = /^const transcript = (\[.*?^\]);/ms.exec(text);
    expect(m, "README has no `const transcript = [...]` literal").not.toBeNull();
    const c = byStem.get("readme_transcript_example")!;
    expect(JSON.parse(m![1]!)).toEqual(c.messages);
    const receipts = checkTurns(c.messages, { knownTools: ["lookup_order", "issue_refund"] });
    expect(receipts.map((r) => r.verdict)).toEqual(["unreported", "fabricated"]);
  });
});
