"""Conftest for livedemo tests — lean, no main_fastapi import.

The parent conftest pulls in the entire FastAPI app for E2E framework. The
livedemo unit tests don't need it; this conftest just adds `src/` to sys.path.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
