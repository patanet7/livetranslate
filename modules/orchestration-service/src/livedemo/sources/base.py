"""SubtitleSource — async iterator of :class:`CaptionEvent`.

Every concrete source (mic, fireflies, file) implements `stream()`. The
pipeline (`livedemo.pipeline.run_once`) iterates one source into one sink and
optionally records each event via :class:`WSRecorder`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from services.pipeline.adapters.source_adapter import CaptionEvent


class SubtitleSource(ABC):
    @abstractmethod
    def stream(self) -> AsyncIterator[CaptionEvent]: ...
