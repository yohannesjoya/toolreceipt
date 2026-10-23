"""``LLMExtractor`` core: prompt assembly, output parsing, fallback and ``from_chat_model``."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from toolreceipt.claims import RuleExtractor, normalize_tool_specs
from toolreceipt.llm import (
    PROMPT_V2,
    PROMPT_VERSION,
    LLMExtractor,
    build_prompt,
    message_text,
    parse_output,
)

TOOLS = normalize_tool_specs(
    [("send_email", "Send an email"), ("issue_refund", "Refund an {order}")]
)
REPLY = "I sent the email."


def _rules(text: str = REPLY) -> list[dict[str, object]]:
    return [c.to_dict() for c in RuleExtractor().extract(text, TOOLS)]


def test_prompt_version_and_placeholders():
    assert PROMPT_VERSION == 2
    assert "{tools}" in PROMPT_V2 and "{reply}" in PROMPT_V2


def test_prompt_contains_tools_in_order_and_reply_last():
    p = build_prompt("Done {reply}", TOOLS)
    assert p.index("- send_email: Send an email") < p.index("- issue_refund: Refund an {order}")
    assert p.rstrip().endswith("Done {reply}\n>>>")
    assert p.count("Done {reply}") == 1


def test_prompt_placeholders_in_descriptions_are_not_expanded():
    tools = normalize_tool_specs([("odd_tool", "Issue {reply} refund }{ and {tools}")])
    p = build_prompt("MY REPLY", tools)
    assert p.count("MY REPLY") == 1
    assert "- odd_tool: Issue {reply} refund }{ and {tools}" in p
    assert p.rstrip().endswith("MY REPLY\n>>>")


def test_parse_output_tolerates_fences_and_prose():
    raw = 'Sure:\n```json\n{"claims": [{"text": "I sent it", "tool": "send_email"}]}\n```'
    assert parse_output(raw) == [("I sent it", "send_email")]
    assert parse_output('{"claims": []}') == []
    assert parse_output('{"claims": [{"text": "all done", "tool": null}]}') == [("all done", None)]


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "no json here",
        '{"claims": "x"}',
        '{"claims": [{"text": 5}]}',
        "[1, 2]",
        '{"claims": [{"tool": "send_email"}]}',
        '{"claims": [{"text": "x", "tool": 3}]}',
    ],
)
def test_parse_output_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_output(bad)


def test_extract_adds_llm_claims_with_source():
    ex = LLMExtractor(
        complete=lambda p: '{"claims": [{"text": "check your inbox", "tool": "send_email"}]}'
    )
    claims = ex.extract("Refund done, check your inbox.", TOOLS)
    llm = [c for c in claims if c.source == "llm"]
    assert [(c.text, c.candidates, c.verb, c.confidence, c.object, c.count) for c in llm] == [
        ("check your inbox", ["send_email"], "other", "medium", None, None)
    ]
    assert llm[0]._object_matched is True
    assert ex.last_raw is not None and ex.last_raw.startswith('{"claims"')
    assert ex.last_error is None


def test_extract_passes_prompt_with_reply_and_tools():
    seen: list[str] = []

    def complete(p: str) -> str:
        seen.append(p)
        return '{"claims": []}'

    LLMExtractor(complete=complete).extract(REPLY, TOOLS)
    assert len(seen) == 1 and REPLY in seen[0] and "- issue_refund:" in seen[0]


def test_fallback_returns_rule_claims_and_records_error(caplog):
    def boom(p: str) -> str:
        raise RuntimeError("rate limited")

    ex = LLMExtractor(complete=boom)
    with caplog.at_level(logging.WARNING, logger="toolreceipt.llm"):
        got = ex.extract(REPLY, TOOLS)
    assert [c.to_dict() for c in got] == _rules()
    assert isinstance(ex.last_error, RuntimeError)
    assert "RuntimeError" in caplog.text and REPLY not in caplog.text


@pytest.mark.parametrize("bad", ["", "nope", '{"claims": "x"}'])
def test_fallback_on_unparsable_output(bad):
    ex = LLMExtractor(complete=lambda p: bad)
    assert [c.to_dict() for c in ex.extract(REPLY, TOOLS)] == _rules()
    assert isinstance(ex.last_error, ValueError) and ex.last_raw == bad


def test_on_error_raise_propagates():
    def boom(p: str) -> str:
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        LLMExtractor(complete=boom, on_error="raise").extract("x", TOOLS)
    with pytest.raises(ValueError):
        LLMExtractor(complete=lambda p: "nope", on_error="raise").extract("x", TOOLS)


def test_from_chat_model_reads_text_property_method_and_blocks():
    class Msg:
        text = '{"claims": [{"text": "I sent the email", "tool": "send_email"}]}'

    class Model:
        def invoke(self, prompt: str) -> Msg:
            return Msg()

    ex = LLMExtractor.from_chat_model(Model())
    assert ex.extract(REPLY, TOOLS)
    assert ex.last_raw == Msg.text
    assert message_text(SimpleNamespace(text=lambda: "a")) == "a"
    assert message_text(SimpleNamespace(content="b")) == "b"
    blocks = [{"type": "thinking", "thinking": "x"}, {"type": "text", "text": "c"}]
    assert message_text(SimpleNamespace(content=blocks)) == "c"
    from langchain_core.messages import AIMessage

    assert message_text(AIMessage(content='{"claims": []}')) == '{"claims": []}'


def test_from_chat_model_forwards_kwargs():
    class Model:
        def invoke(self, prompt: str) -> str:
            raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        LLMExtractor.from_chat_model(Model(), on_error="raise").extract("x", TOOLS)


def test_top_level_export():
    import toolreceipt

    assert toolreceipt.LLMExtractor is LLMExtractor and "LLMExtractor" in toolreceipt.__all__


def test_aextract_runs_complete_on_a_thread():
    import asyncio
    import threading

    seen: list[int] = []

    def complete(prompt: str) -> str:
        seen.append(threading.get_ident())
        return '{"claims": [{"text": "check your inbox", "tool": "send_email"}]}'

    ex = LLMExtractor(complete=complete)
    text = "Refund done, check your inbox."
    sync = [c.to_dict() for c in ex.extract(text, TOOLS)]
    got = asyncio.run(ex.aextract(text, TOOLS))
    assert [c.to_dict() for c in got] == sync
    assert seen and seen[-1] != threading.get_ident()
    assert ex.last_raw is not None and ex.last_error is None


def test_aextract_fallback_sets_last_error_after_await():
    import asyncio

    def boom(prompt: str) -> str:
        raise RuntimeError("rate limited")

    ex = LLMExtractor(complete=boom)
    rules = RuleExtractor().extract("I sent the email.", TOOLS)
    got = asyncio.run(ex.aextract("I sent the email.", TOOLS))
    assert [c.to_dict() for c in got] == [c.to_dict() for c in rules]
    assert isinstance(ex.last_error, RuntimeError) and ex.last_raw is None


def test_aextract_on_error_raise_propagates():
    import asyncio

    def boom(prompt: str) -> str:
        raise RuntimeError("rate limited")

    with pytest.raises(RuntimeError):
        asyncio.run(LLMExtractor(complete=boom, on_error="raise").aextract("x", TOOLS))


def test_none_completion_is_an_error_like_0_2():
    rules = RuleExtractor().extract("I sent the email.", TOOLS)
    ex = LLMExtractor(complete=lambda p: None)  # type: ignore[arg-type,return-value]
    assert [c.to_dict() for c in ex.extract("I sent the email.", TOOLS)] == [
        c.to_dict() for c in rules
    ]
    assert isinstance(ex.last_error, TypeError)
    with pytest.raises(TypeError):
        LLMExtractor(complete=lambda p: None, on_error="raise").extract("x", TOOLS)  # type: ignore[arg-type,return-value]
