"""
Audio Upload Chunk Adapter

Converts Whisper transcription results (from audio upload processing) to unified TranscriptChunk format.
This adapter is used when processing audio uploads through the same pipeline as live transcription data.

Whisper result segment format:
{
    "text": "Hello world",
    "start": 0.0,           # start time in seconds
    "end": 1.25,            # end time in seconds
    "words": [...],         # optional word-level timestamps
    "speaker": "SPEAKER_00", # from diarization (optional)
    "speaker_id": "speaker_1", # optional
    "confidence": 0.95,     # optional confidence score
    "language": "en",       # detected language (optional)
    "segment_index": 0      # segment index (optional)
}
"""

from typing import Any, Dict, Optional
from datetime import datetime, timezone
import uuid

from .base import ChunkAdapter, TranscriptChunk


class AudioUploadChunkAdapter(ChunkAdapter):
    """
    Adapts Whisper transcription results to unified TranscriptChunk format.

    Used for processing audio uploads through the same pipeline as live data,
    ensuring DRY and consistent handling across all transcript sources.

    This adapter handles:
    - Direct Whisper API results
    - Results with speaker diarization info
    - Streaming chunk results from audio coordinator
    """

    def __init__(self, session_id: str = ""):
        """
        Initialize the adapter.

        Args:
            session_id: Session ID for the audio upload session
        """
        self._session_id = session_id

    @property
    def source_type(self) -> str:
        return "audio_upload"

    def adapt(self, raw_chunk: Any) -> TranscriptChunk:
        """
        Convert Whisper result segment to unified format.

        Args:
            raw_chunk: Dict containing Whisper transcription result segment

        Returns:
            TranscriptChunk with normalized fields
        """
        # Handle both dict and model inputs
        if hasattr(raw_chunk, "model_dump"):
            data = raw_chunk.model_dump()
        elif hasattr(raw_chunk, "__dict__"):
            data = vars(raw_chunk)
        elif isinstance(raw_chunk, dict):
            data = raw_chunk
        else:
            raise ValueError(f"Unsupported raw_chunk type: {type(raw_chunk)}")

        # Extract text - handle different Whisper response formats
        text = data.get("text", "")
        if not text and "clean_text" in data:
            text = data.get("clean_text", "")

        # Extract timing - Whisper uses "start"/"end", not "start_time"/"end_time"
        start_time = float(data.get("start", 0) or data.get("start_time", 0))
        end_time = float(data.get("end", start_time) or data.get("end_time", start_time))

        # Extract speaker info - from diarization or direct fields
        speaker = self._extract_speaker_info(data)

        # Get transcript/session ID
        transcript_id = data.get("transcript_id", "") or data.get("session_id", "") or self._session_id

        # Generate chunk_id
        chunk_id = data.get("chunk_id") or data.get("segment_id")
        if chunk_id is None:
            segment_index = data.get("segment_index", data.get("index"))
            if segment_index is not None:
                chunk_id = f"audio_{segment_index}"
            else:
                chunk_id = f"audio_{uuid.uuid4().hex[:8]}"
        else:
            chunk_id = str(chunk_id)

        # Calculate timestamp in milliseconds
        timestamp_ms = int(start_time * 1000)

        # Extract confidence - Whisper may provide word-level or segment-level confidence
        confidence = self._extract_confidence(data)

        # Build metadata
        metadata = {
            "source": "audio_upload",
            "session_id": self._session_id,
            "language": data.get("language"),
            "no_speech_prob": data.get("no_speech_prob"),
            "segment_index": data.get("segment_index", data.get("index")),
        }

        # Add diarization info to metadata if present
        diarization = data.get("diarization", {})
        if diarization:
            metadata["diarization"] = diarization

        # Add word-level timestamps if present
        if "words" in data:
            metadata["words"] = data["words"]

        return TranscriptChunk(
            text=text,
            speaker_name=speaker,
            timestamp_ms=timestamp_ms,
            chunk_id=chunk_id,
            transcript_id=transcript_id,
            start_time_seconds=start_time,
            end_time_seconds=end_time,
            is_final=data.get("is_final", True),  # Audio uploads are typically final
            confidence=confidence,
            metadata=metadata,
        )

    def _extract_speaker_info(self, data: Dict[str, Any]) -> str:
        """
        Extract speaker information from Whisper result.

        Handles various formats:
        - Direct speaker field
        - Diarization results
        - Speaker ID mappings
        """
        # Check direct speaker fields
        if data.get("speaker_name"):
            return data["speaker_name"]
        if data.get("speaker"):
            return data["speaker"]

        # Check diarization info
        diarization = data.get("diarization", {})
        if isinstance(diarization, dict):
            if diarization.get("speaker_label"):
                return diarization["speaker_label"]
            if diarization.get("speaker_id"):
                return diarization["speaker_id"]
            if diarization.get("speaker"):
                return diarization["speaker"]

        # Check speaker_id field
        if data.get("speaker_id"):
            return data["speaker_id"]

        return "Unknown"

    def _extract_confidence(self, data: Dict[str, Any]) -> float:
        """
        Extract confidence score from Whisper result.

        Handles various confidence formats:
        - Direct confidence field
        - Avg_logprob conversion
        - Word-level confidence aggregation
        """
        # Direct confidence
        if "confidence" in data:
            return float(data["confidence"])

        # Convert avg_logprob to confidence (logprob is negative, closer to 0 is better)
        if "avg_logprob" in data:
            avg_logprob = data["avg_logprob"]
            # Convert logprob to rough confidence estimate
            # avg_logprob typically ranges from -0.5 (high confidence) to -2.0 (low confidence)
            import math
            confidence = math.exp(avg_logprob)  # Convert from log space
            return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]

        # Check diarization confidence
        diarization = data.get("diarization", {})
        if isinstance(diarization, dict) and "confidence" in diarization:
            return float(diarization["confidence"])

        # Default confidence for Whisper results
        return 0.9

    def extract_speaker(self, raw_chunk: Any) -> Optional[str]:
        """Extract speaker from Whisper result."""
        if isinstance(raw_chunk, dict):
            return self._extract_speaker_info(raw_chunk)
        if hasattr(raw_chunk, "speaker_name"):
            return raw_chunk.speaker_name
        if hasattr(raw_chunk, "speaker"):
            return raw_chunk.speaker
        return None

    def validate(self, raw_chunk: Any) -> bool:
        """Validate Whisper result has required fields."""
        if isinstance(raw_chunk, dict):
            # Must have text and timing
            has_text = bool(raw_chunk.get("text") or raw_chunk.get("clean_text"))
            has_timing = "start" in raw_chunk or "start_time" in raw_chunk
            return has_text and has_timing
        if hasattr(raw_chunk, "text"):
            return bool(raw_chunk.text)
        return False

    def create_batch_chunks(
        self,
        segments: list,
        session_id: str = "",
    ) -> list[TranscriptChunk]:
        """
        Convert a batch of Whisper segments to TranscriptChunks.

        Convenience method for processing entire transcription results.

        Args:
            segments: List of segment dicts from Whisper result
            session_id: Session ID for the audio upload

        Returns:
            List of TranscriptChunks ready for pipeline processing
        """
        # Update session_id if provided
        if session_id:
            self._session_id = session_id

        chunks = []
        for i, segment in enumerate(segments):
            # Add index if not present
            if "segment_index" not in segment and "index" not in segment:
                segment["segment_index"] = i

            if self.validate(segment):
                chunks.append(self.adapt(segment))

        return chunks
