import time

import pytest

from toolreceipt.claims import (
    RuleExtractor,
    ToolSpec,
    extract_object,
    find_action,
    is_filtered,
    load_lexicon,
    match_tools,
    score_tool,
    split_clauses,
    split_sentences,
    tokenize_tool_name,
)

LEX = load_lexicon()


def test_split_sentences_handles_bullets_and_newlines():
    assert split_sentences("Done!\n- Sent the email.\n- Created ticket #42") == [
        "Done!", "Sent the email.", "Created ticket #42",
    ]


def test_split_sentences_numbered_list_and_decimals():
    assert split_sentences("1. Charged $20.50 to the card. Then I stopped.\n\n2) Done") == [
        "Charged $20.50 to the card.", "Then I stopped.", "Done",
    ]


def test_split_clauses():
    assert split_clauses("I couldn't send the email, but I created ticket #42") == [
        "I couldn't send the email", "I created ticket #42",
    ]


def test_split_clauses_on_dashes_and_colons():
    assert split_clauses("Done - sent the email") == ["Done", "sent the email"]
    assert split_clauses("Done\u2014sent it: all good") == ["Done", "sent it", "all good"]
    assert split_clauses("I e-mailed it at 3:00 today") == ["I e-mailed it at 3:00 today"]


def test_our_customer_is_not_the_agent():
    assert is_filtered("Our customer cancelled the order", LEX)
    assert is_filtered("Our users created duplicates", LEX)
    assert RuleExtractor().extract("Our customer called and then cancelled the order.", []) == []
    assert [c.verb for c in RuleExtractor().extract("Our team called and cancelled it.", [])] == [
        "delete"
    ]


def test_split_clauses_conjunctions_and_numbers():
    assert split_clauses("I refunded 1,000 orders; then I emailed you and so it's done") == [
        "I refunded 1,000 orders", "I emailed you", "it's done",
    ]
    assert split_clauses("However, I sent it because you asked") == [
        "I sent it", "you asked",
    ]


def test_find_action_forms():
    assert find_action("I have emailed you the confirmation", LEX)[0] == "send"
    assert find_action("The ticket has been created", LEX)[0] == "create"
    assert find_action("I checked the order", LEX) is None


def test_find_action_span_and_multiword():
    clause = "I've set up the meeting"
    fam, start, end = find_action(clause, LEX)
    assert (fam, clause[start:end]) == ("create", "set up")


def test_find_action_whole_words_only():
    assert find_action("Use the send_email tool", LEX) is None
    assert find_action("I resent nothing", LEX) is None


def test_find_action_exclusion_first_wins():
    assert find_action("I checked that the email was sent", LEX) is None
    assert find_action("I sent the email you found", LEX)[0] == "send"
    assert find_action("I logged in to the portal", LEX) is None


def test_find_action_skips_non_completed_uses():
    assert find_action("Your scheduled call starts soon", LEX) is None
    assert find_action("Refunds are processed within 5 days", LEX) is None
    assert find_action("The refund is being processed", LEX) is None
    assert find_action("Your subscription is now cancelled", LEX)[0] == "delete"
    assert find_action('You said "it was cancelled"', LEX) is None


def test_filters():
    assert is_filtered("I couldn't send the email", LEX)
    assert is_filtered("I will send the email", LEX)
    assert is_filtered("Would you like me to send the email?", LEX)
    assert is_filtered("You can create a ticket from the portal", LEX)
    assert not is_filtered("I have sent the email", LEX)


def test_filters_use_word_boundaries():
    assert not is_filtered("I notified the customer", LEX)
    assert not is_filtered("I cancelled the order", LEX)
    assert is_filtered("I'll have it sent", LEX)
    assert is_filtered("I haven't emailed you yet", LEX)
    assert is_filtered("The email wasn\u2019t sent", LEX)


def test_filters_other_subjects_and_states():
    assert is_filtered("You sent me the invoice", LEX)
    assert is_filtered("The customer created a duplicate", LEX)
    assert find_action("You're booked for 3pm on Friday", LEX) is None  # present passive
    assert is_filtered("You were charged twice", LEX)
    assert is_filtered("Your card was charged $20 on the 3rd", LEX)
    assert is_filtered("The order was processed by the warehouse", LEX)
    assert is_filtered("The order was created last week", LEX)
    assert is_filtered("The nightly sync ran at 3am", LEX)


def test_filters_keep_agent_confirmations():
    assert not is_filtered("Your refund has been issued", LEX)
    assert not is_filtered("Refund issued", LEX)
    assert not is_filtered("I sent the invoice yesterday", LEX)
    assert not is_filtered("The receipt was sent by email", LEX)
    assert not is_filtered("The ticket was created by me", LEX)
    assert not is_filtered("You've been refunded", LEX)


def test_extract_object_and_count():
    clause = "I updated 3 rows in the orders table"
    fam, start, end = find_action(clause, LEX)
    assert extract_object(clause, end, LEX) == ("3 rows in the orders table", 3)


@pytest.mark.parametrize(
    "clause,expected",
    [
        ("I have emailed you the confirmation.", ("confirmation", None)),
        ("I deleted two duplicate tickets", ("two duplicate tickets", 2)),
        ("I created a ticket", ("ticket", None)),
        ("Your refund has been issued.", (None, None)),
        ("I charged $20 to your card", ("$20 to your card", None)),
        ("Here is the ticket I created: #42.", ("#42", None)),
    ],
)
def test_extract_object_variants(clause, expected):
    _, _, end = find_action(clause, LEX)
    assert extract_object(clause, end, LEX) == expected


def test_extract_question_is_per_clause_and_conditional_lead_is_sentence_level():
    ex = RuleExtractor()
    assert [c.verb for c in ex.extract("I sent the email, did you get it?", [])] == ["send"]
    assert ex.extract("If it went through, your order was created.", []) == []
    assert [c.text for c in ex.extract("When I checked, I cancelled it.", [])] == ["I cancelled it"]


def test_filter_exempt_phrases():
    assert not is_filtered("I can confirm that I have refunded order 4821", LEX)
    assert not is_filtered("Please note that I have cancelled your subscription", LEX)
    assert is_filtered("You can create a ticket from the portal", LEX)
    assert is_filtered("I can see you logged in yesterday", LEX)
    assert is_filtered("Please note that I will cancel it", LEX)


@pytest.mark.parametrize(
    "clause",
    [
        "the customer has cancelled the order",
        "John already refunded your order",
        "Stripe charged your card",
        "Looks like John refunded the order",
    ],
)
def test_third_party_active_subjects_are_filtered(clause):
    assert find_action(clause, LEX) is not None
    assert is_filtered(clause, LEX)


@pytest.mark.parametrize(
    "clause",
    [
        "Our team has sent the email",
        "Our billing team refunded the order",
        "Refund issued",
        "Ticket #42 created.",
        "Refund already issued",
        "Email successfully sent.",
        "Your refund has been processed by our team",
        "The ticket was created by us",
    ],
)
def test_agent_subjects_and_headlines_are_kept(clause):
    assert not is_filtered(clause, LEX)


def test_long_whitespace_runs_are_linear():
    text = "I sent" + " " * 100_000 + "the email, and then x"
    t0 = time.perf_counter()
    claims = RuleExtractor().extract(text, [])
    assert time.perf_counter() - t0 < 0.5
    assert [c.verb for c in claims] == ["send"]


def test_extract_subjectless_clause_after_negation_stands_alone():
    claims = RuleExtractor().extract("I couldn't send the email, but created ticket #42.", [])
    assert [(c.verb, c.text) for c in claims] == [("create", "created ticket #42")]


def test_extract_subjectless_clause_inherits_other_subject():
    assert RuleExtractor().extract("The customer sent the form and then cancelled it.", []) == []


def test_extract_accepts_custom_lexicon():
    lex = load_lexicon()
    lex["families"]["send"]["forms"].append("pinged")
    claims = RuleExtractor(lex).extract("I pinged the team.", [])
    assert [(c.verb, c.object) for c in claims] == [("send", "team")]


# --- Tool matching and dedupe -----------------------------------------------

def test_tokenize_tool_name():
    assert tokenize_tool_name("sendEmail") == ["send", "email"]
    assert tokenize_tool_name("send-email") == ["send", "email"]
    assert tokenize_tool_name("create_tickets") == ["create", "ticket"]


def test_tokenize_tool_name_edge_cases():
    assert tokenize_tool_name("HTTPRequest") == ["http", "request"]
    assert tokenize_tool_name("send_sms") == ["send", "sms"]
    assert tokenize_tool_name("update_address") == ["update", "address"]
    assert tokenize_tool_name("get.v2 Orders") == ["get", "v2", "order"]


def test_match_high_medium_none():
    tools = [
        ToolSpec("send_email", "Send an email to the customer"),
        ToolSpec("issue_refund", "Refund an order"),
    ]
    assert match_tools("send", "you the confirmation", tools, LEX) == (["send_email"], "high")
    assert match_tools("send", "the summary", [ToolSpec("notify_customer", "")], LEX) == (
        ["notify_customer"], "medium"
    )
    # open~create, case~ticket
    assert match_tools("create", "a ticket", [ToolSpec("open_case", "")], LEX) == (
        ["open_case"], "high"
    )
    assert match_tools("send", "the summary", [ToolSpec("issue_refund", "")], LEX) == ([], None)


def test_match_keeps_ties():
    tools = [ToolSpec("send_email", ""), ToolSpec("send_fax", "")]
    cands, conf = match_tools("send", "it", tools, LEX)
    assert cands == ["send_email", "send_fax"] and conf == "medium"


def test_score_components():
    spec = ToolSpec("send_email", "Send an email to the customer")
    # The description only adds an object word the name lacks.
    assert score_tool("send", [], spec, LEX) == 2  # family +2, no description object word
    assert score_tool("send", ["email"], spec, LEX) == 4  # "email" is already in the name
    assert score_tool("send", ["customer"], spec, LEX) == 3  # family +2, description +1
    assert score_tool("delete", ["customer"], spec, LEX) == 1  # description only, once
    assert score_tool("refund", ["money", "back"], ToolSpec("refund_order", ""), LEX) == 2


def test_one_word_does_not_count_twice():
    # "email" is a send-family token and the object: one tool token, one +2.
    assert score_tool("send", ["email"], ToolSpec("email_report", ""), LEX) == 2
    # Distinct tool tokens may each be hit.
    assert score_tool("send", ["email"], ToolSpec("send_email", ""), LEX) == 4
    assert score_tool("refund", ["refund"], ToolSpec("issue_refund", ""), LEX) == 4


def test_dual_sense_tokens_do_not_cross_senses():
    # "issue" is a refund verb and a ticket synonym; "record" a save verb and a row synonym.
    assert match_tools("create", "a ticket", [ToolSpec("issue_refund", "")], LEX) == ([], None)
    assert match_tools("refund", "order 4821", [ToolSpec("create_ticket", "")], LEX) == ([], None)
    assert match_tools("save", "the notes", [ToolSpec("update_row", "")], LEX) == ([], None)
    assert match_tools("update", "the row", [ToolSpec("edit_record", "")], LEX) == (
        ["edit_record"], "medium"
    )
    # Literal uses still match.
    assert match_tools("create", "an issue", [ToolSpec("create_issue", "")], LEX) == (
        ["create_issue"], "high"
    )


def test_stopwords_do_not_score_against_description():
    tools = [ToolSpec("send_email", "Send it to you")]
    assert match_tools("delete", "it to you", tools, LEX) == ([], None)


def test_extract_wires_matching():
    tools = [ToolSpec("send_email", "Send an email"), ToolSpec("send_sms", "")]
    claims = RuleExtractor().extract("I sent the email.", tools)
    assert [(c.candidates, c.confidence, c.tool, c.reason) for c in claims] == [
        (["send_email"], "high", None, None)
    ]


def test_passive_and_headline_object_is_the_subject():
    tools = [ToolSpec("issue_refund", ""), ToolSpec("create_ticket", "")]
    text = "Your refund has been issued. Ticket #42 has been created."
    claims = RuleExtractor().extract(text, tools)
    assert [(c.object, c.candidates) for c in claims] == [
        ("refund", ["issue_refund"]), ("ticket #42", ["create_ticket"]),
    ]


def test_dedupe_same_candidates_keeps_first():
    tools = [ToolSpec("send_email", "")]
    text = "I emailed you. The email has been sent. I sent it again."
    claims = RuleExtractor().extract(text, tools)
    assert [c.text for c in claims] == ["I emailed you"]


def test_dedupe_unmatched_on_verb_and_object():
    ex = RuleExtractor()
    claims = ex.extract("I created the notes. I created the notes. I created the plan.", [])
    assert [c.text for c in claims] == ["I created the notes", "I created the plan"]


def test_dedupe_ties_are_a_set():
    tools = [ToolSpec("send_email", ""), ToolSpec("send_sms", "")]
    claims = RuleExtractor().extract("I sent it. I sent the email.", tools)
    assert [c.candidates for c in claims] == [["send_email", "send_sms"], ["send_email"]]


def test_lexicon_review_forms():
    assert find_action("I assigned the ticket to billing", LEX)[0] == "update"
    assert find_action("I e-mailed you the receipt", LEX)[0] == "send"
    assert find_action("I have summarised the issue", LEX) is None  # no family


@pytest.mark.parametrize(
    "family,obj,tool",
    [
        ("delete", "two duplicate tickets", ToolSpec("create_ticket", "Open a support ticket")),
        ("refund", "the order", ToolSpec("update_payment", "")),
        ("update", "the order", ToolSpec("lookup_order", "Look up an order")),
        ("schedule", "a call", ToolSpec("cancel_appointment", "")),
    ],
)
def test_candidacy_requires_a_verb_family_hit(family, obj, tool):
    assert match_tools(family, obj, [tool], LEX) == ([], None)


def test_noun_tokens_still_match_as_objects():
    tools = [ToolSpec("book_appointment", ""), ToolSpec("update_payment", "")]
    assert match_tools("schedule", "the appointment", tools, LEX) == (["book_appointment"], "high")
    assert match_tools("update", "the payment", tools, LEX) == (["update_payment"], "high")


def _cands(text, names):
    claims = RuleExtractor().extract(text, [ToolSpec(n, "") for n in names])
    return [(c.candidates, c.confidence) for c in claims]


@pytest.mark.parametrize(
    "text,names,expected",
    [
        # A different family's verb or a read verb in the name excludes the tool.
        ("I notified you.", ["read_email", "send_email"], ["send_email"]),
        ("I emailed you the receipt.", ["send_email", "delete_email"], ["send_email"]),
        ("I've sent you a message.", ["get_messages", "send_sms"], ["send_sms"]),
        ("I've sent it.", ["delete_email"], []),
        ("I've emailed you the details.", ["read_email", "lookup_order"], []),
        ("I've emailed you the details.", ["lookUpEmail"], []),
        ("I refunded the order.", ["issue_refund", "create_issue"], ["issue_refund"]),
        ("I issued a refund.", ["create_issue"], []),
        # The claim's own verb in the name keeps the tool (Stripe-style names).
        ("I refunded your order.", ["create_refund"], ["create_refund"]),
        ("I sent the invoice.", ["send_invoice"], ["send_invoice"]),
        ("I charged your card.", ["create_charge"], ["create_charge"]),
    ],
)
def test_conflicting_verbs_exclude_tools(text, names, expected):
    assert [c for c, _ in _cands(text, names)] == [expected]


def test_surface_stem_is_an_object_word():
    assert _cands("I emailed you.", ["send_email", "send_sms"]) == [(["send_email"], "high")]
    assert _cands("I texted you.", ["send_email", "send_sms"]) == [(["send_sms"], "high")]
    assert _cands("I messaged you.", ["send_message"]) == [(["send_message"], "high")]


def test_charge_is_its_own_family():
    assert find_action("I billed your card", LEX)[0] == "charge"
    assert find_action("I paid the invoice", LEX)[0] == "charge"
    assert _cands("I charged your card.", ["issue_refund"]) == [([], None)]
    assert _cands("Your refund has been processed.", ["issue_refund"]) == [
        (["issue_refund"], "high")
    ]


def test_closed_is_not_an_action():
    # "closed" belongs to no family: closing a ticket is often an update, not a delete.
    assert find_action("I closed the ticket", LEX) is None


def test_subject_phrase_stops_at_prepositions():
    claims = RuleExtractor().extract(
        "The email to your manager has been sent.", [ToolSpec("send_email", "")]
    )
    assert [(c.object, c.candidates, c.confidence) for c in claims] == [
        ("email", ["send_email"], "high")
    ]


@pytest.mark.parametrize(
    "text,names",
    [
        ("I paid the invoice.", ["payment_history"]),
        ("I emailed you.", ["email_customer"]),
        ("I sent the report.", ["email_report"]),
        ("I issued a refund.", ["issue_tracker"]),
    ],
)
def test_noun_tokens_alone_never_make_a_candidate(text, names):
    assert _cands(text, names) == [([], None)]
