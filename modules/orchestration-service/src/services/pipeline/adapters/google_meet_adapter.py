"""
Google Meet Chunk Adapter

Converts Google Meet browser automation transcript chunks to unified TranscriptChunk format.

Google Meet transcription comes from browser automation (Puppeteer/Playwright)
that captures Google's live captions. The format varies based on how captions
are extracted from the DOM.

Expected format (from browser_audio_capture.py):
{
    "transcript": "Hello world",
    "speaker_id": "SPEAKER_00",       # Diarization ID
    "speaker_name": "John Doe",       # Human name if available
    "timestamp": 1234567890,          # Unix timestamp in ms
    "confidence": 0.95
}
"""

from typing import Any, Dict, Optional
from datetime import datetime, timezone

from .base import ChunkAdapter, TranscriptChunk


class GoogleMeetChunkAdapter(ChunkAdapter):
    """
    Adapts Google Meet transcription chunks to unified TranscriptChunk format.

    Google Meet chunks come from browser automation and may have either
    speaker_name (human name) or speaker_id (diarization ID like SPEAKER_00).
    This adapter handles both cases.
    """

    @property
    def source_type(self) -> str:
        return "google_meet"

    def adapt(self, raw_chunk: Any) -> TranscriptChunk:
        """
        Convert Google Meet chunk to unified format.

        Args:
            raw_chunk: Dict from browser automation or model

        Returns:
            TranscriptChunk with normalized fields
        """
        # Handle various input types
        if hasattr(raw_chunk, "model_dump"):
            data = raw_chunk.model_dump()
        elif hasattr(raw_chunk, "__dict__") and not isinstance(raw_chunk, dict):
            data = vars(raw_chunk)
        elif isinstance(raw_chunk, dict):
            data = raw_chunk
        else:
            raise ValueError(f"Unsupported raw_chunk type: {type(raw_chunk)}")

        # Text field - Google Meet uses "transcript" not "text"
        text = data.get("transcript") or data.get("text", "")

        # Speaker - prefer human name over diarization ID
        speaker = (
            data.get("speaker_name")
            or data.get("speaker_id")
            or data.get("speaker")
            or "Unknown"
        )

        # Timestamp handling - Google Meet sends Unix timestamps in ms
        timestamp_ms = int(data.get("timestamp", 0))
        if timestamp_ms == 0:
            timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        # Calculate seconds for compatibility
        start_time_seconds = timestamp_ms / 1000.0

        # Duration - estimate if not provided
        # Google Meet typically sends word-level chunks, estimate ~0.5s per chunk
        duration_ms = float(data.get("duration_ms", 500))
        end_time_seconds = start_time_seconds + (duration_ms / 1000.0)

        # Generate chunk ID
        chunk_id = data.get("chunk_id", f"gm_{timestamp_ms}")

        return TranscriptChunk(
            text=text,
            speaker_name=speaker,
            timestamp_ms=timestamp_ms,
            chunk_id=chunk_id,
            transcript_id=data.get("meeting_id", data.get("transcript_id", "")),
            start_time_seconds=start_time_seconds,
            end_time_seconds=end_time_seconds,
            is_final=True,  # Google Meet sends final captions
            confidence=float(data.get("confidence", 0.9)),
            metadata={
                "source": "google_meet",
                "speaker_id": data.get("speaker_id"),
                "speaker_name": data.get("speaker_name"),
                "meeting_id": data.get("meeting_id"),
            },
        )

    def extract_speaker(self, raw_chunk: Any) -> Optional[str]:
        """
        Extract best available speaker identifier.

        Prefers human name over diarization ID.
        """
        if isinstance(raw_chunk, dict):
            return (
                raw_chunk.get("speaker_name")
                or raw_chunk.get("speaker_id")
                or raw_chunk.get("speaker")
            )
        # Model/object access
        return getattr(raw_chunk, "speaker_name", None) or getattr(
            raw_chunk, "speaker_id", None
        )

    def validate(self, raw_chunk: Any) -> bool:
        """Validate Google Meet chunk has required fields."""
        if isinstance(raw_chunk, dict):
            has_text = bool(raw_chunk.get("transcript") or raw_chunk.get("text"))
            has_speaker = bool(
                raw_chunk.get("speaker_name")
                or raw_chunk.get("speaker_id")
                or raw_chunk.get("speaker")
            )
            return has_text and has_speaker
        return hasattr(raw_chunk, "transcript") or hasattr(raw_chunk, "text")

    def adapt_with_diarization(
        self, raw_chunk: Dict, diarization_map: Dict[str, str]
    ) -> TranscriptChunk:
        """
        Adapt with speaker name lookup from diarization map.

        When Google Meet only provides SPEAKER_00 style IDs, this method
        can map them to human names using a diarization_map.

        Args:
            raw_chunk: Raw chunk from browser
            diarization_map: Dict mapping speaker_id -> human name

        Returns:
            TranscriptChunk with resolved speaker name
        """
        chunk = self.adapt(raw_chunk)

        # Try to resolve speaker ID to human name
        speaker_id = raw_chunk.get("speaker_id")
        if speaker_id and speaker_id in diarization_map:
            chunk.speaker_name = diarization_map[speaker_id]
            chunk.metadata["resolved_from"] = speaker_id

        return chunk
