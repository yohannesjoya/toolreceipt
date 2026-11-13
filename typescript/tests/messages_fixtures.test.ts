/** Message-list fixtures: a conversation in an SDK shape -> ledger, final reply, verdict. */
import { describe, expect, it } from "vitest";

import { checkMessages, finalReply, ledgerFromMessages } from "../src/messages";
import { loadCases } from "./helpers";

type Case = {
  messages: unknown[];
  known_tools: never;
  raises?: unknown;
  expected: { ledger: Record<string, unknown>[]; final_reply: string; verdict: string };
};
const CASES = loadCases<Case>("messages");
const REQUIRED = [
  "serialized_langchain_rejected", "stored_chat_message_rejected",
  "openai_refund_and_email_ok", "openai_missing_tool_result", "openai_never_called",
  "openai_no_final_reply", "openai_run_ends_on_call_after_text", "openai_multi_turn",
  "openai_bad_arguments_json", "openai_content_blocks", "openai_tool_content_blocks",
  "openai_orphan_tool_message", "openai_empty_ids", "openai_repeated_id",
  "openai_legacy_function_call_rejected", "langchain_objects_rejected",
  "responses_function_call_ok", "responses_output_list", "responses_missing_output",
  "responses_ends_on_function_call", "responses_item_id_not_call_id", "responses_multi_turn",
  "anthropic_tool_use_ok", "anthropic_tool_result_blocks", "anthropic_is_error",
  "anthropic_error_string_not_inferred", "anthropic_multi_turn_tool_result_user",
  "anthropic_tool_result_with_text_user", "openai_custom_tool_call_rejected",
  "responses_custom_tool_call_rejected", "responses_mcp_call_rejected",
  "openai_image_only_user_continues_run", "responses_builtin_tool_in_list",
  "transcript_settled_entries_last_run", "transcript_settled_entry_after_reply",
  "transcript_settled_entry_structured_result",
];

describe("messages fixtures", () => {
  it("required fixtures present", () => {
    const stems = new Set(CASES.map(([s]) => s));
    for (const r of REQUIRED) expect(stems, r).toContain(r);
  });

  it.each(CASES)("%s", (stem, c) => {
    if ("raises" in c) {
      expect(() => ledgerFromMessages(c.messages)).toThrow(TypeError);
      expect(() => checkMessages(c.messages, { knownTools: c.known_tools })).toThrow(TypeError);
      return;
    }
    const entries = ledgerFromMessages(c.messages).entries;
    expect(entries.length, JSON.stringify(entries.map((e) => e.toDict()))).toBe(c.expected.ledger.length);
    entries.forEach((entry, i) => {
      const dict = entry.toDict();
      for (const [key, value] of Object.entries(c.expected.ledger[i]!)) expect(dict[key], key).toEqual(value);
      expect(entry.startedAt).toBeNull();
      if (!stem.startsWith("transcript_")) expect(entry.callId).not.toBeNull();
    });
    expect(finalReply(c.messages)).toBe(c.expected.final_reply);
    const r = checkMessages(c.messages, { knownTools: c.known_tools });
    expect(r.verdict, JSON.stringify(r.toDict())).toBe(c.expected.verdict);
  });
});
