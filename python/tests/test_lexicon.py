from toolreceipt.claims import load_lexicon


def test_lexicon_shape():
    lex = load_lexicon()
    required = {
        "send", "create", "update", "delete", "refund",
        "schedule", "save", "submit", "trigger",
    }
    assert required <= set(lex["families"])
    for fam in lex["families"].values():
        assert fam["forms"] and fam["tokens"]
    keys = (
        "exclusions", "negation", "future", "advice",
        "other_subject", "synonyms", "number_words",
    )
    for key in keys:
        assert key in lex


def test_lexicon_forms_do_not_overlap_exclusions():
    lex = load_lexicon()
    forms = {f for fam in lex["families"].values() for f in fam["forms"]}
    assert not forms & set(lex["exclusions"])


def test_default_load_is_not_mutable_through_cache():
    lex = load_lexicon()
    lex["families"]["send"]["forms"].append("zzz")
    assert "zzz" not in load_lexicon()["families"]["send"]["forms"]


def test_lexicon_forms_unique_across_families():
    lex = load_lexicon()
    all_forms = [f for fam in lex["families"].values() for f in fam["forms"]]
    assert len(all_forms) == len(set(all_forms))
