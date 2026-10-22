"""Evaluate ``LLMExtractor`` on the probe sets and record the model output for offline replay.

Opt-in and network-bound: not part of ``pytest``. Reads ``GROQ_API_KEY`` from the environment
(or from the repository's ``.env``), runs one set through a real model, writes
``fixtures/probe/<set>.recorded.json`` and prints the README table. The committed recording
is the artifact: ``tests/test_probe_replay.py`` replays it with no network and gates on it.

    ../demo/.venv/bin/python scripts/eval_llm.py --set dev
    ../demo/.venv/bin/python scripts/eval_llm.py --set heldout --model openai/gpt-oss-120b

The scoring helpers (``score``, ``evaluate``, ``replay_extractor``, ``load_set``) are
imported by the replay test so the live run and the offline gate measure the same way.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBE = ROOT / "fixtures" / "probe"
sys.path.insert(0, str(HERE.parent))

from toolreceipt import Ledger, LLMExtractor, check  # noqa: E402
from toolreceipt.claims import normalize_tool_specs  # noqa: E402
from toolreceipt.llm import PROMPT_VERSION  # noqa: E402

DEFAULT_MODEL = "openai/gpt-oss-20b"  # the model behind the committed recordings
SETS = ("dev", "heldout", "v01")


# --- sets -------------------------------------------------------------------------------


def load_set(name: str) -> list[dict[str, Any]]:
    """Probe cases for ``dev``, ``heldout`` or ``v01`` (the 0.1 held-out honest replies)."""
    if name in ("dev", "heldout"):
        cases: list[dict[str, Any]] = json.loads((PROBE / f"{name}.json").read_text("utf-8"))
        return cases
    if name != "v01":
        raise ValueError(f"unknown set {name!r}; expected one of {SETS}")
    honest = json.loads((ROOT / "fixtures" / "heldout" / "honest_replies.json").read_text("utf-8"))
    out: list[dict[str, Any]] = []
    for case in honest:
        ledger = [
            {k: v for k, v in e.items() if k in ("tool", "status", "error") and v is not None}
            for e in case["ledger"]
        ]
        out.append({
            "id": case["id"], "category": "V01_honest", "domain": "support",
            "reply": case["text"], "known_tools": case["known_tools"], "ledger": ledger,
            "expected_tools": sorted({t for c in case["expected"] for t in c["candidates"]}),
        })
    return out


def recording_path(name: str) -> Path:
    return PROBE / f"{name}.recorded.json"


# --- scoring ----------------------------------------------------------------------------


def replay_extractor(llm_output: str | None) -> LLMExtractor:
    """An extractor that returns the recorded text; ``None`` (a failed call) takes the
    fallback path, so the replay reproduces the miss instead of re-running guards on nothing."""

    def complete(_prompt: str) -> str:
        if llm_output is None:
            raise RuntimeError("recorded call failed")
        return llm_output

    return LLMExtractor(complete=complete)


def truth_fabricated(case: dict[str, Any]) -> bool:
    ok = {e["tool"] for e in case["ledger"] if e["status"] == "ok"}
    return any(t not in ok for t in case["expected_tools"])


def evaluate(
    case: dict[str, Any], llm_output: str | None, *, llm_blocks: bool, rules_only: bool = False
) -> dict[str, Any]:
    """Resolved tools and the caught/false-alarm outcome for one case under one setting.
    ``rules_only`` scores the default extractor alone (the README's "rules only" row)."""
    ledger = Ledger.from_dict(case["ledger"])
    rules = check(case["reply"], ledger, known_tools=case["known_tools"])
    receipt = rules if rules_only else check(
        case["reply"], ledger, known_tools=case["known_tools"],
        extractor=replay_extractor(llm_output), llm_blocks=llm_blocks,
    )
    judged = receipt.confirmed + receipt.claimed_but_never_ran + receipt.suspected
    resolved = {c.tool for c in judged if c.tool}
    caught = receipt.verdict == "fabricated" or bool(receipt.suspected)
    return {
        "resolved": sorted(resolved), "verdict": receipt.verdict, "caught": caught,
        "rules_caught": rules.verdict == "fabricated",
        "suspected": [c.tool for c in receipt.suspected],
        "want_fab": truth_fabricated(case),
        "fallback": llm_output is None and not rules_only,
    }


def score(
    cases: list[dict[str, Any]], *, llm_blocks: bool, rules_only: bool = False
) -> dict[str, dict[str, int]]:
    """Per-category and ``TOTAL`` counts: claims, hit, spur, fab, caught, false_alarm,
    llm_false_alarm, fallbacks.

    ``hit``/``spur`` are tool-level (resolved tools in / not in the labels); ``caught`` is a
    labelled fabrication the receipt flags (``fabricated``, or non-empty ``suspected`` under
    the default); ``false_alarm`` is a non-fabrication the receipt flags, and
    ``llm_false_alarm`` the subset the rules alone would not have flagged (the gate: the
    rules' own false alarms are 0.1 behaviour, unchanged by this feature).
    """
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for case in cases:
        r = evaluate(
            case, case.get("llm_output"), llm_blocks=llm_blocks, rules_only=rules_only
        )
        want = set(case["expected_tools"])
        got = set(r["resolved"])
        for key in (case["category"], "TOTAL"):
            s = stats[key]
            s["cases"] += 1
            s["claims"] += len(want)
            s["hit"] += len(want & got)
            s["spur"] += len(got - want)
            s["fab"] += r["want_fab"]
            s["caught"] += r["want_fab"] and r["caught"]
            s["false_alarm"] += (not r["want_fab"]) and r["caught"]
            s["llm_false_alarm"] += (not r["want_fab"]) and r["caught"] and not r["rules_caught"]
            s["fallbacks"] += r["fallback"]
    return {k: dict(v) for k, v in stats.items()}


def render_table(stats: dict[str, dict[str, int]]) -> str:
    head = f"{'category':12} cases claims  hit spur | fab caught falseAlarm (llm) fallbacks"
    rows = [head]
    for cat in sorted(k for k in stats if k != "TOTAL") + ["TOTAL"]:
        s = stats[cat]
        rows.append(
            f"{cat:12} {s['cases']:5} {s['claims']:6} {s['hit']:4} {s['spur']:4} | "
            f"{s['fab']:3} {s['caught']:6} {s['false_alarm']:10} {s['llm_false_alarm']:5} "
            f"{s['fallbacks']:9}"
        )
    t = stats["TOTAL"]
    resolved = t["hit"] + t["spur"]
    precision = t["hit"] / resolved if resolved else float("nan")
    recall = t["caught"] / t["fab"] if t["fab"] else float("nan")
    rows.append(
        f"claims found {t['hit']}/{t['claims']}, precision {precision:.2f}, "
        f"fabrications caught {t['caught']}/{t['fab']} ({recall:.0%}), "
        f"false alarms {t['false_alarm']} ({t['llm_false_alarm']} from the LLM path)"
    )
    return "\n".join(rows)


def summary_tables(cases: list[dict[str, Any]]) -> str:
    """The three README rows for one set: rules only, rules + LLM under the default, and
    rules + LLM under ``llm_blocks=True``."""
    out = [f"== rules only\n{render_table(score(cases, llm_blocks=False, rules_only=True))}"]
    for llm_blocks in (False, True):
        label = "llm_blocks=True" if llm_blocks else "default (suspected)"
        out.append(f"== {label}\n{render_table(score(cases, llm_blocks=llm_blocks))}")
    return "\n\n".join(out)


# --- live run ---------------------------------------------------------------------------


def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text("utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip():
                os.environ.setdefault(key.strip(), value.strip().strip('"'))


def groq_complete(model: str, retries: int = 8) -> Any:
    """A ``complete`` callable on Groq's JSON mode, temperature 0, with backoff on errors.

    A 429 carries ``retry-after``; it is honoured when present, else the backoff grows
    10 s per attempt. The last error's class and message are kept for the failure log.
    """
    from groq import Groq

    client = Groq(max_retries=0)

    def complete(prompt: str) -> str:
        last: Exception | None = None
        for attempt in range(retries):
            try:
                r = client.chat.completions.create(
                    model=model, temperature=0, response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}],
                )
                return str(r.choices[0].message.content)
            except Exception as exc:  # rate limits, transient provider errors
                last = exc
                headers = getattr(getattr(exc, "response", None), "headers", None) or {}
                wait = float(headers.get("retry-after") or 0) or 10.0 * (attempt + 1)
                time.sleep(min(wait + 1, 120))
        detail = f"{type(last).__name__}: {str(last)[:160]}"
        raise RuntimeError(f"gave up after {retries} attempts: {detail}")

    return complete


def record(name: str, model: str, workers: int = 2) -> Path:
    cases = load_set(name)
    complete = groq_complete(model)

    def one(case: dict[str, Any]) -> dict[str, Any]:
        # ``last_raw`` is per-extractor state, so each call gets its own instance.
        extractor = LLMExtractor(complete=complete, on_error="raise")
        specs = normalize_tool_specs((t["name"], t["description"]) for t in case["known_tools"])
        try:
            extractor.extract(case["reply"], specs)
            raw: str | None = extractor.last_raw
        except Exception as exc:
            print(f"{case['id']}: call failed ({exc})", file=sys.stderr)
            raw = None
        return {**case, "llm_output": raw}

    started = time.time()
    with ThreadPoolExecutor(workers) as pool:
        recorded = list(pool.map(one, cases))
    payload = {
        "prompt_version": PROMPT_VERSION, "model": model,
        "recorded_at": dt.date.today().isoformat(), "cases": recorded,
    }
    path = recording_path(name)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", "utf-8")
    took = time.time() - started
    print(f"recorded {len(recorded)} cases to {path.relative_to(ROOT)} in {took:.0f}s")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", choices=SETS, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--workers", type=int, default=2,
        help="parallel calls; use 1 on Groq's free tier (8k tokens per minute)",
    )
    parser.add_argument(
        "--replay", action="store_true", help="print the table from the recording only"
    )
    args = parser.parse_args(argv)
    if not args.replay:
        _load_env()
        if not os.environ.get("GROQ_API_KEY"):
            print("GROQ_API_KEY is not set", file=sys.stderr)
            return 2
        record(args.set, args.model, args.workers)
    data = json.loads(recording_path(args.set).read_text("utf-8"))
    print(f"set={args.set} model={data['model']} prompt_version={data['prompt_version']} "
          f"recorded_at={data['recorded_at']}\n")
    print(summary_tables(data["cases"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
