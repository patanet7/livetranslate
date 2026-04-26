"""livedemo — DRY, config-driven E2E pipeline harness.

See `docs/plans/PLAN_7_LIVEDEMO_E2E.md` for the design.

Public surface:
    LiveDemoConfig                 — single Pydantic config (env > YAML > defaults)
    CheckResult, check_all,
    register_check                 — preflight registry

Higher-level surfaces (sources, sinks, harness, recorder, pipeline) are exported
incrementally as phases land.
"""
# Ensure sibling packages (`services`, `bot`, `clients`, ...) are importable
# whether livedemo is mounted as `livedemo` (test conftest path) or
# `src.livedemo` (console-script via wheel). The orchestration-service uses
# bare names like `from services.pipeline...` throughout — match that.
import sys as _sys
from pathlib import Path as _Path

_SRC = _Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

from .config import LiveDemoConfig  # noqa: E402
from .preflight import CheckResult, check_all, register_check  # noqa: E402

__all__ = [
    "LiveDemoConfig",
    "CheckResult",
    "check_all",
    "register_check",
]
