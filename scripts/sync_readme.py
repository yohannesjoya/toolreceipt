"""Generate python/README.md from the root README.md with absolute URLs, and copy
LICENSE into python/ so the wheel and sdist ship it.

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
LICENSE = ROOT / "LICENSE"
LICENSE_TARGET = ROOT / "python" / "LICENSE"
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
    outputs = {
        TARGET: render(SOURCE.read_text(encoding="utf-8")),
        LICENSE_TARGET: LICENSE.read_text(encoding="utf-8"),
    }
    if "--check" in argv:
        stale = [
            path for path, expected in outputs.items()
            if (path.read_text(encoding="utf-8") if path.is_file() else None) != expected
        ]
        for path in stale:
            print(f"{path.relative_to(ROOT)} is out of date: run python scripts/sync_readme.py")
        return 1 if stale else 0
    for path, expected in outputs.items():
        if path.is_symlink():
            path.unlink()
        path.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
