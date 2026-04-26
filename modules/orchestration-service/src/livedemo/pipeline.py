"""pipeline.run_once — drives one Source into one Sink with optional recording.

Single async function, deliberately small. The CLI wraps it with config
loading + preflight (Phase 6).
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from livetranslate_common.logging import get_logger

from .recorder import WSRecorder
from .sinks.base import CaptionSink
from .sources.base import SubtitleSource
from services.pipeline.adapters.source_adapter import CaptionEvent

logger = get_logger()


async def run_once(
    *,
    source: SubtitleSource,
    sink: CaptionSink,
    recorder: WSRecorder | None = None,
) -> int:
    """Pump events from source → sink. Records each event if recorder given.

    Returns the number of events processed.
    """
    count = 0
    logger.info(
        "run_started",
        source=type(source).__name__,
        sink=type(sink).__name__,
        recording=recorder is not None,
    )
    async with sink:
        async for evt in source.stream():
            if recorder is not None:
                recorder.record("caption", _to_payload(evt))
            await sink.consume(evt)
            count += 1
    logger.info("run_completed", count=count, sink=type(sink).__name__)
    return count


def _to_payload(evt: CaptionEvent) -> dict[str, Any]:
    """Project CaptionEvent → recorder-compatible payload dict.

    Drops timestamp/expires_at (recorder records its own ts) and serialisable
    runtime fields only — see test_smoke_e2e B8 round-trip.
    """
    if is_dataclass(evt):
        d = asdict(evt)
    else:  # pragma: no cover - defensive
        d = {k: getattr(evt, k) for k in evt.__annotations__}
    d.pop("timestamp", None)
    d.pop("expires_at", None)
    d.pop("speaker_color", None)
    return d
