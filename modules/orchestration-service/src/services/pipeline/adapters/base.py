"""
Base Chunk Adapter

Defines the interface for source-specific adapters and the unified TranscriptChunk format.
All sources convert their raw chunks to TranscriptChunk, which is then converted to
FirefliesChunk internally (since existing services are built around that format).

The adapter pattern isolates source-specific parsing logic while keeping the pipeline DRY.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranscriptChunk:
    """
    Unified chunk format - all sources convert to this.

    This is the DRY contract between adapters and the coordinator.
    The coordinator then converts this to FirefliesChunk for internal processing.

    Attributes:
        text: The transcribed text content
        speaker_name: Name or identifier of the speaker
        timestamp_ms: Timestamp in milliseconds (relative to meeting start)
        chunk_id: Unique identifier for this chunk
        transcript_id: Identifier for the overall transcript/meeting
        start_time_seconds: Start time in seconds (for duration calculation)
        end_time_seconds: End time in seconds (for duration calculation)
        is_final: Whether this is a final (vs interim) transcription
        confidence: Confidence score (0.0 - 1.0)
        metadata: Source-specific metadata for debugging/logging
    """

    text: str
    speaker_name: str | None
    timestamp_ms: int
    chunk_id: str
    transcript_id: str = ""
    start_time_seconds: float = 0.0
    end_time_seconds: float = 0.0
    is_final: bool = False
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return (self.end_time_seconds - self.start_time_seconds) * 1000

    @property
    def word_count(self) -> int:
        """Approximate word count."""
        return len(self.text.split()) if self.text else 0


class ChunkAdapter(ABC):
    """
    Abstract base for source-specific chunk adapters.

    Each source (Fireflies, Google Meet, Whisper, etc.) implements this interface
    to convert its native chunk format to the unified TranscriptChunk.

    The adapter is stateless - it simply transforms data formats.

    Usage:
        adapter = FirefliesChunkAdapter()
        chunk = adapter.adapt(raw_fireflies_chunk)
    """

    @property
    @abstractmethod
    def source_type(self) -> str:
        """
        Return the source type identifier.

        Used for database storage (source_type column) and logging.
        Examples: "fireflies", "google_meet", "whisper"
        """
        pass

    @abstractmethod
    def adapt(self, raw_chunk: Any) -> TranscriptChunk:
        """
        Convert source-specific chunk to unified TranscriptChunk.

        Args:
            raw_chunk: The raw chunk data from the source API.
                       Type varies by source (dict, Pydantic model, etc.)

        Returns:
            TranscriptChunk with normalized fields

        Raises:
            ValueError: If the raw chunk is malformed or missing required fields
        """
        pass

    @abstractmethod
    def extract_speaker(self, raw_chunk: Any) -> str | None:
        """
        Extract speaker name from source-specific format.

        Some sources have complex speaker identification (e.g., Google Meet
        has both speaker_id and speaker_name). This method provides a
        consistent way to extract the best available speaker identifier.

        Args:
            raw_chunk: The raw chunk data from the source API

        Returns:
            Speaker name or identifier, or None if unknown
        """
        pass

    def validate(self, raw_chunk: Any) -> bool:
        """
        Validate that a raw chunk has required fields.

        Override in subclasses for source-specific validation.

        Args:
            raw_chunk: The raw chunk data to validate

        Returns:
            True if valid, False otherwise
        """
        return True
