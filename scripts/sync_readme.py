"""Generate python/README.md from the root README.md with absolute URLs.

PyPI renders the package README without the repository checkout, so relative
links and images must point at GitHub. Usage: sync_readme.py [--check].
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "README.md"
TARGET = ROOT / "python" / "README.md"
RAW = "https://raw.githubusercontent.com/yohannesjoya/toolreceipt/main/"
BLOB = "https://github.com/yohannesjoya/toolreceipt/blob/main/"
# Plain links/images, plus badge links of the form [![alt](img)](target).
LINK = re.compile(r"(!?)(\[(?:[^\[\]]|!\[[^\]]*\]\([^)]*\))*\])\(([^)\s]+)\)")
ABSOLUTE = re.compile(r"^([a-z][a-z0-9+.-]*:|#|/)", re.IGNORECASE)


def _rewrite(match: re.Match[str]) -> str:
    bang, label, target = match.groups()
    if ABSOLUTE.match(target):
        return match.group(0)
    path = target.removeprefix("./")
    return f"{bang}{label}({(RAW if bang else BLOB)}{path})"


def render(text: str) -> str:
    return LINK.sub(_rewrite, text)


def main(argv: list[str]) -> int:
    expected = render(SOURCE.read_text(encoding="utf-8"))
    if "--check" in argv:
        current = TARGET.read_text(encoding="utf-8") if TARGET.is_file() else None
        if current != expected:
            print("python/README.md is out of date: run python scripts/sync_readme.py")
            return 1
        return 0
    if TARGET.is_symlink():
        TARGET.unlink()
    TARGET.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
