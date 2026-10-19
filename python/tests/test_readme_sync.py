"""python/README.md and python/LICENSE must be the generated copies of the root files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_package_readme_is_in_sync() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sync_readme.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
    assert not (ROOT / "python" / "README.md").is_symlink()
    assert not (ROOT / "python" / "LICENSE").is_symlink()
