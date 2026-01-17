"""
Import Chunk Adapter

Converts imported transcript sentences (from Fireflies or other sources) to unified TranscriptChunk format.
This adapter is used when importing historical transcripts to process them through the same pipeline
as live transcription data.

Imported sentence format (from Fireflies API or local storage):
{
    "text": "Hello world",
    "speaker_name": "Alice",
    "speaker_id": "speaker_1",
    "start_time": 0.0,      # seconds
    "end_time": 1.25,       # seconds
    "raw_text": "hello world",  # optional, original unprocessed text
    "index": 0              # sentence index in transcript
}
"""

from typing import Any, Dict, Optional
from datetime import datetime, timezone
import uuid

from .base import ChunkAdapter, TranscriptChunk


class ImportChunkAdapter(ChunkAdapter):
    """
    Adapts imported transcript sentences to unified TranscriptChunk format.

    Used for processing historical/imported transcripts through the same pipeline
    as live data, ensuring DRY and consistent handling.
    """

    def __init__(self, source_name: str = "import"):
        """
        Initialize the adapter.

        Args:
            source_name: Source identifier (e.g., "fireflies_import", "local_import")
        """
        self._source_name = source_name

    @property
    def source_type(self) -> str:
        return self._source_name

    def adapt(self, raw_chunk: Any) -> TranscriptChunk:
        """
        Convert imported sentence to unified format.

        Args:
            raw_chunk: Dict containing sentence data from imported transcript

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

        # Extract fields with defaults
        text = data.get("text", "")
        speaker = data.get("speaker_name") or data.get("speaker") or "Unknown"
        start_time = float(data.get("start_time", 0))
        end_time = float(data.get("end_time", start_time + 1.0))
        transcript_id = data.get("transcript_id", "")

        # Generate chunk_id if not provided
        chunk_id = data.get("chunk_id") or data.get("index")
        if chunk_id is not None:
            chunk_id = f"import_{chunk_id}"
        else:
            chunk_id = f"import_{uuid.uuid4().hex[:8]}"

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
            is_final=True,  # Imported data is always final
            confidence=float(data.get("confidence", 0.95)),
            metadata={
                "source": self._source_name,
                "original_index": data.get("index"),
                "speaker_id": data.get("speaker_id"),
                "raw_text": data.get("raw_text"),
            },
        )

    def extract_speaker(self, raw_chunk: Any) -> Optional[str]:
        """Extract speaker from imported sentence."""
        if hasattr(raw_chunk, "speaker_name"):
            return raw_chunk.speaker_name
        if isinstance(raw_chunk, dict):
            return raw_chunk.get("speaker_name") or raw_chunk.get("speaker")
        return None

    def validate(self, raw_chunk: Any) -> bool:
        """Validate imported sentence has required fields."""
        if isinstance(raw_chunk, dict):
            return bool(raw_chunk.get("text"))
        return hasattr(raw_chunk, "text") and bool(raw_chunk.text)

    def create_batch_chunks(
        self,
        sentences: list,
        transcript_id: str = "",
    ) -> list[TranscriptChunk]:
        """
        Convert a batch of sentences to TranscriptChunks.

        Convenience method for processing entire imported transcripts.

        Args:
            sentences: List of sentence dicts from imported transcript
            transcript_id: ID of the source transcript

        Returns:
            List of TranscriptChunks ready for pipeline processing
        """
        chunks = []
        for i, sentence in enumerate(sentences):
            # Add index if not present
            if "index" not in sentence:
                sentence["index"] = i
            if not sentence.get("transcript_id"):
                sentence["transcript_id"] = transcript_id

            chunks.append(self.adapt(sentence))

        return chunks
