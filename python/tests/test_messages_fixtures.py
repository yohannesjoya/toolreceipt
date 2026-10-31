"""Message-list fixtures: a conversation in an SDK shape -> ledger, final reply, verdict."""

from __future__ import annotations

import pytest

from tests.helpers import load_cases
from toolreceipt.messages import check_messages, final_reply, ledger_from_messages

CASES = load_cases("messages")
REQUIRED = {
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
}


def test_required_fixtures_present():
    assert REQUIRED <= {stem for stem, _ in CASES}


@pytest.mark.parametrize("stem,case", CASES, ids=lambda c: c if isinstance(c, str) else "")
def test_messages_fixture(stem, case):
    messages = case["messages"]
    if "raises" in case:
        with pytest.raises(ValueError):
            ledger_from_messages(messages)
        with pytest.raises(ValueError):
            check_messages(messages, known_tools=case["known_tools"])
        return
    exp = case["expected"]
    entries = ledger_from_messages(messages).entries
    assert len(entries) == len(exp["ledger"]), [e.to_dict() for e in entries]
    for entry, want in zip(entries, exp["ledger"], strict=True):
        for key, value in want.items():
            assert getattr(entry, key) == value, (key, entry.to_dict())
        assert entry.started_at is None
        assert entry.call_id is not None or stem.startswith("transcript_")
    assert final_reply(messages) == exp["final_reply"]
    receipt = check_messages(messages, known_tools=case["known_tools"])
    assert receipt.verdict == exp["verdict"], receipt.to_dict()
