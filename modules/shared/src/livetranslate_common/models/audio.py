"""Audio-related shared Pydantic models and protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class AudioChunk(BaseModel):
    """A single chunk of raw PCM audio data.

    NOTE: Internal Python-to-Python use only. WebSocket transport
    uses raw binary frames, not serialized AudioChunk instances.

    Args:
        data: Raw PCM bytes for this chunk.
        timestamp_ms: Capture timestamp in milliseconds since epoch.
        sequence_number: Monotonically increasing chunk index within the stream.
        source_id: Identifier for the audio source (e.g. device ID or session ID).
    """

    data: bytes
    timestamp_ms: int
    sequence_number: int
    source_id: str


@runtime_checkable
class MeetingAudioStream(Protocol):
    """Protocol for meeting audio stream sources.

    Any object that implements these attributes and the ``read_chunk``
    coroutine satisfies this protocol via structural subtyping.

    Attributes:
        source_type: Identifier for the audio source type (e.g. "browser", "microphone").
        sample_rate: Sample rate in Hz (e.g. 16000, 48000).
        channels: Number of audio channels (1 = mono, 2 = stereo).
        encoding: PCM encoding format (e.g. "pcm_s16le").
    """

    source_type: str
    sample_rate: int
    channels: int
    encoding: str

    async def read_chunk(self) -> AudioChunk | None:
        """Read the next audio chunk from the stream.

        Returns:
            The next AudioChunk, or None when the stream is exhausted.
        """
        ...
