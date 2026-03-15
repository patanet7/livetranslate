"""TranscriptionBackend protocol — any transcription engine implements this."""
from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

import numpy as np
from livetranslate_common.models import ModelInfo, TranscriptionResult


@runtime_checkable
class TranscriptionBackend(Protocol):
    """Protocol for pluggable transcription backends."""

    async def transcribe(
        self, audio: np.ndarray, language: str | None = None, **kwargs
    ) -> TranscriptionResult: ...

    async def transcribe_stream(
        self, audio: np.ndarray, language: str | None = None, **kwargs
    ) -> AsyncIterator[TranscriptionResult]: ...

    def supports_language(self, lang: str) -> bool: ...

    def get_model_info(self) -> ModelInfo: ...

    async def load_model(self, model_name: str, device: str = "cuda") -> None: ...

    async def unload_model(self) -> None: ...

    async def warmup(self) -> None: ...

    def vram_usage_mb(self) -> int: ...
