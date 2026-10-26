from toolreceipt.claims import Claim, ToolSpec, normalize_tool_specs


def test_claim_to_dict_exact_fields():
    c = Claim(text="I sent it", verb="send")
    assert set(c.to_dict()) == {
        "text", "verb", "object", "count", "candidates", "tool", "confidence", "reason",
        "source", "observed",
    }
    assert Claim.from_dict(c.to_dict()) == c


def test_normalize_tool_specs_shapes():
    specs = normalize_tool_specs([
        "a",
        ("b", "B desc"),
        ToolSpec("c", "C"),
        {"name": "d", "description": "D"},
        {"type": "function", "function": {"name": "e", "description": "E"}},
    ])
    assert specs == [
        ToolSpec("a", ""), ToolSpec("b", "B desc"), ToolSpec("c", "C"),
        ToolSpec("d", "D"), ToolSpec("e", "E"),
    ]
