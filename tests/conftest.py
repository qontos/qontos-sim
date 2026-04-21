"""Test path bootstrap for local qontos-sim development."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATHS = (
    REPO_ROOT / "src",
    REPO_ROOT.parent / "qontos" / "src",
)

for path in SRC_PATHS:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
