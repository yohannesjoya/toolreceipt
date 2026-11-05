"""Record the Python package's output for every shared fixture, for the TypeScript parity tests.

Writes three generated files. ``fixtures/recorded/parity.json`` holds ``extract``, ``diff``,
``check``,
``messages`` and ``turns`` output for ``fixtures/extractor``, ``heldout``, ``diff``, ``check``,
``messages``, ``turns`` and the demo corpus. ``fixtures/recorded/parity_llm.json`` holds the prompt,
plus claims and receipts for every ``fixtures/llm`` case and recorded probe output.
``fixtures/demo/tools.json`` holds the demo's tool specs, one list.

``python/tests/test_recorded_parity_current.py`` fails when they are stale. Usage, from the
repository root::

    python/.venv/bin/python scripts/record_parity.py
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import toolreceipt  # noqa: E402
from toolreceipt import Ledger, check  # noqa: E402
from toolreceipt.claims import Claim, RuleExtractor, normalize_tool_specs  # noqa: E402
from toolreceipt.diff import diff  # noqa: E402
from toolreceipt.llm import PROMPT_V2, PROMPT_VERSION, LLMExtractor, build_prompt  # noqa: E402
from toolreceipt.messages import check_messages, final_reply, ledger_from_messages  # noqa: E402
from toolreceipt.turns import check_turns  # noqa: E402

FIXTURES = ROOT / "fixtures"
DEMO = ROOT / "demo"
PARITY = FIXTURES / "recorded" / "parity.json"
PARITY_LLM = FIXTURES / "recorded" / "parity_llm.json"
DEMO_TOOLS = FIXTURES / "demo" / "tools.json"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _cases(subdir: str) -> list[tuple[str, dict[str, Any]]]:
    return [(p.stem, _load(p)) for p in sorted((FIXTURES / subdir).glob("*.json"))]


def demo_tool_specs(mode: str = "honest") -> list[dict[str, str]]:
    """``{name, description}`` of the demo support desk's tools, loaded without the demo venv."""
    path = DEMO / "scenarios" / "support_desk.py"
    spec = importlib.util.spec_from_file_location("_support_desk", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [{"name": t.name, "description": t.description} for t in module.build_tools(mode)]


def _extract(text: str, known_tools: Any) -> list[dict[str, Any]]:
    claims = RuleExtractor().extract(text, normalize_tool_specs(known_tools))
    return [c.to_dict() for c in claims]


def _check(case: dict[str, Any]) -> Any:
    return check(case["text"], Ledger.from_dict(case["ledger"]), known_tools=case["known_tools"])


def build(demo_tools: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """The recording, minus ``recorded_at``."""
    tools = demo_tools if demo_tools is not None else demo_tool_specs()
    extract: list[dict[str, Any]] = []
    for stem, case in _cases("extractor"):
        claims = _extract(case["text"], case["known_tools"])
        extract.append({"source": f"extractor/{stem}", "claims": claims})
    for name in ("adversarial_negatives", "honest_replies"):
        for case in _load(FIXTURES / "heldout" / f"{name}.json"):
            extract.append({
                "source": f"heldout/{name}#{case['id']}",
                "claims": _extract(case["text"], case["known_tools"]),
            })

    diffs: list[dict[str, Any]] = []
    for stem, case in _cases("diff"):
        receipt = diff(
            [Claim.from_dict(c) for c in case["claims"]],
            Ledger.from_dict(case["ledger"]),
            known_tools=case["known_tools"],
            ignore_tools=case["ignore_tools"],
            llm_blocks=case.get("llm_blocks", False),
            check_args=case.get("check_args", True),
        )
        diffs.append({"source": f"diff/{stem}", "receipt": receipt.to_dict()})

    checks: list[dict[str, Any]] = []
    for stem, case in _cases("check"):
        checks.append({"source": f"check/{stem}", "receipt": _check(case).to_dict()})
    for case in _load(FIXTURES / "heldout" / "honest_replies.json"):
        source = f"heldout/honest_replies#{case['id']}"
        checks.append({"source": source, "receipt": _check(case).to_dict()})
    demo_specs = [(t["name"], t["description"]) for t in tools]
    for sub in ("replays", "examples"):
        for path in sorted((DEMO / sub).glob("*.json")):
            data = _load(path)
            case = {"text": data["reply"], "ledger": data["ledger"], "known_tools": demo_specs}
            checks.append({"source": f"demo/{sub}/{path.stem}", "receipt": _check(case).to_dict()})

    messages: list[dict[str, Any]] = []
    for stem, case in _cases("messages"):
        if "raises" in case:
            continue
        msgs, known = case["messages"], case["known_tools"]
        messages.append({
            "source": f"messages/{stem}",
            "ledger": ledger_from_messages(msgs).to_dict(),
            "final_reply": final_reply(msgs),
            "receipt": check_messages(msgs, known_tools=known).to_dict(),
        })

    turns: list[dict[str, Any]] = []
    for stem, case in _cases("turns"):
        if "raises" in case:
            continue
        receipts = check_turns(case["messages"], known_tools=case["known_tools"])
        turns.append({"source": f"turns/{stem}", "receipts": [r.to_dict() for r in receipts]})

    return {
        "toolreceipt": toolreceipt.__version__,
        "extract": extract,
        "diff": diffs,
        "check": checks,
        "messages": messages,
        "turns": turns,
    }


def _replay(llm_output: str | None) -> LLMExtractor:
    def complete(_prompt: str) -> str:
        if llm_output is None:
            raise RuntimeError("recorded call failed")
        return llm_output

    return LLMExtractor(complete=complete)


def _llm_receipts(reply: str, ledger: Ledger, known: Any, output: str | None) -> dict[str, Any]:
    def run(llm_blocks: bool) -> dict[str, Any]:
        receipt = check(
            reply, ledger, known_tools=known, extractor=_replay(output), llm_blocks=llm_blocks
        )
        return receipt.to_dict()

    return {"default": run(False), "llm_blocks": run(True)}


def build_llm() -> dict[str, Any]:
    """The LLM recording: prompt, ``fixtures/llm`` and the recorded probe outputs."""
    llm: list[dict[str, Any]] = []
    for stem, case in _cases("llm"):
        known = case["known_tools"]
        names = [t["name"] for t in known]
        ledgers = (
            [case["ledger"]] if "ledger" in case
            else [
                [],
                [{"tool": n, "status": "ok"} for n in names],
                [{"tool": n, "status": "error", "error": "boom"} for n in names],
            ]
        )
        claims = _replay(case["llm_output"]).extract(case["reply"], normalize_tool_specs(known))
        llm.append({
            "source": f"llm/{stem}",
            "claims": [c.to_dict() for c in claims],
            "receipts": [
                _llm_receipts(case["reply"], Ledger.from_dict(ld), known, case["llm_output"])
                for ld in ledgers
            ],
        })
    probe: list[dict[str, Any]] = []
    for name in ("dev", "heldout", "v01"):
        for case in _load(FIXTURES / "probe" / f"{name}.recorded.json")["cases"]:
            known, output = case["known_tools"], case["llm_output"]
            claims = _replay(output).extract(case["reply"], normalize_tool_specs(known))
            probe.append({
                "source": f"probe/{name}#{case['id']}",
                "claims": [c.to_dict() for c in claims],
                **_llm_receipts(case["reply"], Ledger.from_dict(case["ledger"]), known, output),
            })
    tools = normalize_tool_specs([("send_email", "Send an email"), "issue_refund"])
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt": PROMPT_V2,
        "built": build_prompt("Done {reply}", tools),
        "llm": llm,
        "probe": probe,
    }


def _dump_lines(data: dict[str, Any]) -> str:
    """One key per line and one case per line, so a behaviour change gives a readable diff."""
    parts: list[str] = []
    for key, value in data.items():
        if isinstance(value, list):
            rows = ",\n".join(json.dumps(v, ensure_ascii=False) for v in value)
            parts.append(f"{json.dumps(key)}: [\n{rows}\n]")
        else:
            parts.append(f"{json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}")
    return "{\n" + ",\n".join(parts) + "\n}\n"


def _dump(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    tools = demo_tool_specs()
    DEMO_TOOLS.parent.mkdir(parents=True, exist_ok=True)
    DEMO_TOOLS.write_text(_dump(tools), encoding="utf-8")
    recording = build(tools)
    recording = {
        "toolreceipt": recording["toolreceipt"],
        "recorded_at": dt.date.today().isoformat(),
        **{k: v for k, v in recording.items() if k != "toolreceipt"},
    }
    PARITY.parent.mkdir(parents=True, exist_ok=True)
    PARITY.write_text(_dump(recording), encoding="utf-8")
    llm_recording = build_llm()
    PARITY_LLM.write_text(_dump_lines(llm_recording), encoding="utf-8")
    counts = f"llm {len(llm_recording['llm'])}, probe {len(llm_recording['probe'])}"
    print(f"wrote {PARITY_LLM.relative_to(ROOT)} ({counts})")
    print(f"wrote {DEMO_TOOLS.relative_to(ROOT)} ({len(tools)} tools)")
    print(
        f"wrote {PARITY.relative_to(ROOT)} (extract {len(recording['extract'])}, "
        f"diff {len(recording['diff'])}, check {len(recording['check'])}, "
        f"messages {len(recording['messages'])}, turns {len(recording['turns'])})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
