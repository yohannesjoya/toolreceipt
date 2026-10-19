"""Collect live support desk runs as replay fixtures.

Usage (from ``demo/``)::

    uv run python scripts/collect_runs.py --mode pressure --n 10 --out replays

Each successful live run is saved as ``{out}/{mode}-{index:03d}.json`` in the exact
``run_scenario`` return shape. Only live results are saved: the replay fallback is pointed at
an empty directory, so a failed run raises instead of returning a recorded one. Runs are
spaced out for Groq's free tier, with a longer back-off after a rate-limit error.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))

PAUSE_SECONDS = 8.0
BACKOFF_SECONDS = 30.0
MAX_BACKOFFS = 2


class _FailureNames(logging.Handler):
    """Remembers the exception type names the demo app logs when a live run fails."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.names: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith("live run failed") and record.args:
            self.names.append(str(record.args[0]))


def _next_index(out: Path, mode: str) -> int:
    taken = [int(p.stem.rsplit("-", 1)[1]) for p in out.glob(f"{mode}-*.json")]
    return max(taken, default=0) + 1


def _live_once(app: Any, mode: str, failures: _FailureNames) -> dict[str, Any] | None:
    """One live run; ``None`` after a failure (the app logs the exception type)."""
    failures.names.clear()
    try:
        result: dict[str, Any] = app.run_scenario(mode, model=app.make_model())
    except FileNotFoundError:  # the empty replay dir: the live run failed
        return None
    if result.get("source") != "live":
        raise RuntimeError("refusing to save a non-live result")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True, choices=("honest", "pressure"))
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--out", type=Path, default=DEMO / "replays")
    args = parser.parse_args(argv)

    load_dotenv(DEMO.parent / ".env")
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set", file=sys.stderr)
        return 2
    # No replay can stand in for a failed live run while collecting.
    os.environ["TOOLRECEIPT_REPLAY_DIR"] = tempfile.mkdtemp(prefix="no-replays-")

    import app  # after the environment is set up

    failures = _FailureNames()
    logging.getLogger("toolreceipt.demo").addHandler(failures)
    args.out.mkdir(parents=True, exist_ok=True)

    saved = 0
    for run in range(args.n):
        if run:
            time.sleep(PAUSE_SECONDS)
        result = _live_once(app, args.mode, failures)
        backoffs = 0
        while result is None and backoffs < MAX_BACKOFFS and "RateLimitError" in failures.names:
            backoffs += 1
            print(f"run {run + 1}: rate limited, backing off {BACKOFF_SECONDS:g} s", flush=True)
            time.sleep(BACKOFF_SECONDS)
            result = _live_once(app, args.mode, failures)
        if result is None:
            print(f"run {run + 1}: failed ({', '.join(failures.names) or 'unknown'})", flush=True)
            continue
        path = args.out / f"{args.mode}-{_next_index(args.out, args.mode):03d}.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        saved += 1
        receipt = result["receipt"]
        print(f"run {run + 1}: {receipt['verdict']} ok={receipt['ok']} -> {path.name}", flush=True)
    print(f"saved {saved} of {args.n} {args.mode} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
