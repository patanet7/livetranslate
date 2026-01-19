"""
Fireflies Chunk Adapter

Converts Fireflies WebSocket transcript chunks to unified TranscriptChunk format.

Fireflies API Reference: https://docs.fireflies.ai/realtime-api/event-schema

Fireflies chunk format (from WebSocket):
{
    "type": "transcript",
    "transcript_id": "abc123",
    "chunk_id": "chunk_001",
    "text": "Hello world",
    "speaker_name": "Alice",
    "start_time": 0.0,      # seconds
    "end_time": 1.25        # seconds
}
"""

from datetime import UTC, datetime
from typing import Any

from .base import ChunkAdapter, TranscriptChunk


class FirefliesChunkAdapter(ChunkAdapter):
    """
    Adapts Fireflies WebSocket chunks to unified TranscriptChunk format.

    Handles both dict and FirefliesChunk model inputs for flexibility.
    """

    @property
    def source_type(self) -> str:
        return "fireflies"

    def adapt(self, raw_chunk: Any) -> TranscriptChunk:
        """
        Convert Fireflies chunk to unified format.

        Args:
            raw_chunk: Either a dict from WebSocket or FirefliesChunk model

        Returns:
            TranscriptChunk with normalized fields
        """
        # Handle both dict and model inputs
        if hasattr(raw_chunk, "model_dump"):
            # Pydantic model (FirefliesChunk)
            data = raw_chunk.model_dump()
        elif hasattr(raw_chunk, "__dict__"):
            # Dataclass or object
            data = vars(raw_chunk)
        elif isinstance(raw_chunk, dict):
            data = raw_chunk
        else:
            raise ValueError(f"Unsupported raw_chunk type: {type(raw_chunk)}")

        # Extract fields with defaults
        text = data.get("text", "")
        speaker = data.get("speaker_name") or data.get("speaker", "Unknown")
        start_time = float(data.get("start_time", 0))
        end_time = float(data.get("end_time", start_time))
        transcript_id = data.get("transcript_id", "")
        chunk_id = data.get("chunk_id", f"ff_{datetime.now(UTC).timestamp()}")

        # Calculate timestamp in milliseconds
        timestamp_ms = int(start_time * 1000)

        return TranscriptChunk(
            text=text,
            speaker_name=speaker,
            timestamp_ms=timestamp_ms,
            chunk_id=chunk_id,
            transcript_id=transcript_id,
            start_time_seconds=start_time,
            end_time_seconds=end_time,
            is_final=data.get("is_final", True),  # Fireflies sends final chunks
            confidence=float(data.get("confidence", 1.0)),
            metadata={
                "source": "fireflies",
                "raw_type": data.get("type"),
                "original_chunk_id": chunk_id,
            },
        )

    def extract_speaker(self, raw_chunk: Any) -> str | None:
        """Extract speaker from Fireflies chunk."""
        if hasattr(raw_chunk, "speaker_name"):
            speaker: str | None = raw_chunk.speaker_name
            return speaker
        if isinstance(raw_chunk, dict):
            result: str | None = raw_chunk.get("speaker_name") or raw_chunk.get("speaker")
            return result
        return None

    def validate(self, raw_chunk: Any) -> bool:
        """Validate Fireflies chunk has required fields."""
        if isinstance(raw_chunk, dict):
            return bool(raw_chunk.get("text")) and bool(
                raw_chunk.get("speaker_name") or raw_chunk.get("speaker")
            )
        return hasattr(raw_chunk, "text") and hasattr(raw_chunk, "speaker_name")

    def adapt_from_model(self, chunk: Any) -> TranscriptChunk:
        """
        Adapt directly from FirefliesChunk model.

        This is a convenience method when you already have a typed model.

        Args:
            chunk: FirefliesChunk model instance

        Returns:
            TranscriptChunk
        """
        return TranscriptChunk(
            text=chunk.text,
            speaker_name=chunk.speaker_name,
            timestamp_ms=int(chunk.start_time * 1000),
            chunk_id=chunk.chunk_id,
            transcript_id=chunk.transcript_id,
            start_time_seconds=chunk.start_time,
            end_time_seconds=chunk.end_time,
            is_final=True,
            confidence=1.0,
            metadata={
                "source": "fireflies",
                "duration_ms": chunk.duration_ms,
                "word_count": chunk.word_count,
            },
        )
