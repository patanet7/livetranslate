"""FileSource — replay recorder JSONL as a SubtitleSource.

Skips non-caption kinds (ws_send/ws_recv/frame). Honors `replay_speed`:
- 0.0  → emit instantly with no sleeps
- 1.0  → real-time
- 10.0 → 10x faster

Used for:
- CI smoke tests (deterministic, offline, no Chromium)
- Reproducing bugs from recorded runs
- Theme/layout iteration
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

from .base import SubtitleSource
from services.pipeline.adapters.source_adapter import CaptionEvent


class FileSource(SubtitleSource):
    def __init__(self, *, jsonl_path: Path | str, replay_speed: float = 1.0) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.replay_speed = float(replay_speed)

    async def stream(self) -> AsyncIterator[CaptionEvent]:
        prev_ts: float | None = None
        with self.jsonl_path.open("r") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    line = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if line.get("kind") != "caption":
                    continue
                payload = line.get("payload") or {}
                ts = float(line.get("ts") or 0.0)
                if self.replay_speed > 0.0 and prev_ts is not None:
                    delta = (ts - prev_ts) / self.replay_speed
                    if delta > 0:
                        await asyncio.sleep(delta)
                prev_ts = ts
                yield _payload_to_event(payload)


def _payload_to_event(p: dict) -> CaptionEvent:
    return CaptionEvent(
        event_type=p.get("event_type", "added"),
        caption_id=p["caption_id"],
        text=p.get("text", ""),
        speaker_name=p.get("speaker_name"),
        speaker_id=p.get("speaker_id"),
        source_lang=p.get("source_lang", "auto"),
        target_lang=p.get("target_lang"),
        translated_text=p.get("translated_text"),
        confidence=float(p.get("confidence", 1.0)),
        is_draft=bool(p.get("is_draft", False)),
    )
