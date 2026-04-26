"""CaptionSink — async context manager that consumes :class:`CaptionEvent`s.

The single :meth:`consume` method is the only abstract method, so concrete
sinks stay tiny. `__aenter__` / `__aexit__` may do bot/IO setup but default
to no-ops.

Shared helper :func:`apply_meeting_config_snapshot` flows live
:class:`MeetingSessionConfig` updates into a :class:`WebcamConfig` so all
three sinks pick up `/mode`, `/theme`, `/font` etc. without restart.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from bot.virtual_webcam import WebcamConfig
from livetranslate_common.theme import DisplayMode

from services.pipeline.adapters.source_adapter import CaptionEvent


class CaptionSink(ABC):
    async def __aenter__(self) -> "CaptionSink":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    @abstractmethod
    async def consume(self, caption: CaptionEvent) -> None: ...


_DISPLAY_MODE_LOOKUP = {
    "subtitle": DisplayMode.SUBTITLE,
    "split": DisplayMode.SPLIT,
    "interpreter": DisplayMode.INTERPRETER,
}


def apply_meeting_config_snapshot(target: WebcamConfig, snapshot: dict[str, Any]) -> None:
    """Mutate ``target`` in place with values from a MeetingSessionConfig snapshot.

    Mapping (MeetingSessionConfig → WebcamConfig):
      display_mode (str)        → target.display_mode  (DisplayMode enum)
      show_speakers (bool)      → target.show_speaker_names
      show_original (bool)      → (no direct match — handled by add_caption)
      font_size (int)           → target.font_size
      theme (str)               → (no direct WebcamConfig field; consumer
                                   reads via target_lang/source_lang theme bridge)

    Unrecognised values are left untouched. This is the single place sinks call
    to keep their renderer state in sync with the live MeetingSessionConfig.
    """
    mode_str = snapshot.get("display_mode")
    if isinstance(mode_str, str):
        mode = _DISPLAY_MODE_LOOKUP.get(mode_str.lower())
        if mode is not None:
            target.display_mode = mode

    show_speakers = snapshot.get("show_speakers")
    if isinstance(show_speakers, bool):
        target.show_speaker_names = show_speakers

    font_size = snapshot.get("font_size")
    if isinstance(font_size, int) and font_size > 0:
        target.font_size = font_size
