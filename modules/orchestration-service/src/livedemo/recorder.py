"""WSRecorder — captures every event in a run to `messages.jsonl`.

Recording is the load-bearing primitive that makes runs reproducible (B8). Every
caption emitted by a source, every frame pushed to the bot, every WS message
traded with the orchestration service goes through `record(kind, payload)`.

The output JSONL is replayable via :class:`livedemo.sources.file.FileSource`.
"""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class WSRecorder:
    def __init__(self, *, run_dir: Path | str, enabled: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "messages.jsonl"
        self.enabled = enabled
        self._fh = self.path.open("a", buffering=1) if enabled else None

    def record(self, kind: str, payload: dict[str, Any]) -> None:
        if not self.enabled or self._fh is None:
            return
        line = {
            "ts": time.monotonic(),
            "wall": datetime.now(UTC).isoformat(),
            "kind": kind,
            "payload": payload,
        }
        self._fh.write(json.dumps(line) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "WSRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
