import json

from toolreceipt.ledger import Ledger, LedgerEntry, coerce_args, truncate


def test_entry_to_dict_has_exact_fields():
    e = LedgerEntry(tool="send_email")
    assert set(e.to_dict()) == {
        "tool", "status", "args", "result", "error",
        "call_id", "parent_run_id", "started_at", "ended_at",
    }
    assert e.to_dict()["status"] == "ok"


def test_entry_roundtrip():
    e = LedgerEntry(
        tool="x", status="error", args={"a": 1}, error="boom",
        call_id="c1", started_at=1.0, ended_at=2.0,
    )
    assert LedgerEntry.from_dict(e.to_dict()) == e


def test_coerce_args_makes_json_safe():
    class Obj:
        def __str__(self):
            return "OBJ"

    assert coerce_args({"o": Obj(), "n": 1}) == {"o": "OBJ", "n": 1}
    assert coerce_args("raw") == {"input": "raw"}
    assert coerce_args(None) == {}


def test_truncate():
    assert truncate("abcdef", 4) == "abc…"
    assert truncate("abc", 4) == "abc"
    assert truncate(None, 4) is None


def test_ledger_sets_and_order():
    ledger = Ledger([
        LedgerEntry(tool="b", started_at=2.0),
        LedgerEntry(tool="a", started_at=1.0, status="error"),
        LedgerEntry(tool="c"),
    ])
    assert [e.tool for e in ledger.entries] == ["a", "b", "c"]
    assert ledger.ok_tools() == {"b", "c"}
    assert ledger.error_tools() == {"a"}
    assert ledger.tools() == {"a", "b", "c"}
    assert json.dumps(ledger.to_dict())
    assert Ledger.from_dict(ledger.to_dict()).to_dict() == ledger.to_dict()
