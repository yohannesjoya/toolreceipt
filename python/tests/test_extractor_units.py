import time

import pytest

from toolreceipt.claims import (
    RuleExtractor,
    extract_object,
    find_action,
    is_filtered,
    load_lexicon,
    split_clauses,
    split_sentences,
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
