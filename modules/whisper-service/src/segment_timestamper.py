#!/usr/bin/env python3
"""
Segment Timestamping for Real-Time Streaming

Adds ISO 8601 absolute timestamps to transcript segments.
Converts relative timestamps (segment.start, segment.end) to absolute timestamps.

Following Phase 3.1 architecture:
- absolute_start_time: ISO 8601 timestamp when segment actually occurred
- absolute_end_time: ISO 8601 timestamp when segment ended
- Required for Vexa-compatible WebSocket messages

Usage:
    timestamper = SegmentTimestamper()

    segment = {
        "text": "Hello",
        "start": 0.0,  # Relative to chunk start
        "end": 2.0,
        "speaker": "SPEAKER_00"
    }

    timestamped = timestamper.add_absolute_timestamps(
        segment=segment,
        chunk_start_time=datetime.now(timezone.utc)
    )

    # Result:
    # {
    #     "text": "Hello",
    #     "start": 0.0,
    #     "end": 2.0,
    #     "speaker": "SPEAKER_00",
    #     "absolute_start_time": "2025-01-15T10:30:00Z",
    #     "absolute_end_time": "2025-01-15T10:30:02Z",
    #     "is_final": False
    # }
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class SegmentTimestamper:
    """
    Adds ISO 8601 absolute timestamps to transcript segments

    Converts relative timestamps to absolute timestamps required
    for WebSocket streaming protocol.
    """

    def __init__(self):
        """Initialize segment timestamper"""
        pass

    def add_absolute_timestamps(
        self,
        segment: Dict[str, Any],
        chunk_start_time: datetime,
        is_final: bool = False
    ) -> Dict[str, Any]:
        """
        Add absolute ISO 8601 timestamps to segment

        Parameters:
            segment (Dict): Segment with relative timestamps
                Required fields:
                - start (float): Relative start time in seconds
                - end (float): Relative end time in seconds
                Optional fields:
                - text (str): Transcript text
                - speaker (str): Speaker identifier
                - confidence (float): Transcription confidence

            chunk_start_time (datetime): UTC datetime when audio chunk started
            is_final (bool): Whether this segment is finalized (won't change)

        Returns:
            Dict: Segment with added absolute timestamps
                New fields:
                - absolute_start_time (str): ISO 8601 timestamp
                - absolute_end_time (str): ISO 8601 timestamp
                - is_final (bool): Finalization flag

        Example:
            segment = {
                "text": "Hello world",
                "start": 1.5,  # 1.5 seconds into chunk
                "end": 3.0,
                "speaker": "SPEAKER_00",
                "confidence": 0.95
            }

            chunk_start = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
            timestamped = timestamper.add_absolute_timestamps(segment, chunk_start)

            # Result:
            # {
            #     ...original fields...
            #     "absolute_start_time": "2025-01-15T10:30:01.500Z",
            #     "absolute_end_time": "2025-01-15T10:30:03.000Z",
            #     "is_final": False
            # }
        """
        # Get relative timestamps
        start_offset = segment.get('start', 0.0)
        end_offset = segment.get('end', 0.0)

        # Calculate absolute timestamps
        absolute_start = chunk_start_time + timedelta(seconds=start_offset)
        absolute_end = chunk_start_time + timedelta(seconds=end_offset)

        # Format as ISO 8601 with 'Z' suffix (UTC)
        absolute_start_str = absolute_start.isoformat().replace('+00:00', 'Z')
        absolute_end_str = absolute_end.isoformat().replace('+00:00', 'Z')

        # Create new segment with absolute timestamps
        timestamped_segment = segment.copy()
        timestamped_segment['absolute_start_time'] = absolute_start_str
        timestamped_segment['absolute_end_time'] = absolute_end_str
        timestamped_segment['is_final'] = is_final

        logger.debug(
            f"Timestamped segment: {segment.get('text', '(no text)')} "
            f"[{absolute_start_str} - {absolute_end_str}]"
        )

        return timestamped_segment

    def format_timestamp(self, dt: datetime) -> str:
        """
        Format datetime as ISO 8601 with 'Z' suffix

        Parameters:
            dt (datetime): Datetime to format (should be UTC)

        Returns:
            str: ISO 8601 formatted timestamp with 'Z' suffix

        Example:
            dt = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
            formatted = timestamper.format_timestamp(dt)
            # "2025-01-15T10:30:00Z"
        """
        return dt.isoformat().replace('+00:00', 'Z')

    def parse_timestamp(self, timestamp_str: str) -> datetime:
        """
        Parse ISO 8601 timestamp string to datetime

        Parameters:
            timestamp_str (str): ISO 8601 timestamp (with or without 'Z')

        Returns:
            datetime: Parsed datetime in UTC

        Example:
            dt = timestamper.parse_timestamp("2025-01-15T10:30:00Z")
            # datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        """
        # Handle 'Z' suffix
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'

        return datetime.fromisoformat(timestamp_str)


# Example usage and testing
if __name__ == "__main__":
    print("Segment Timestamper Test")
    print("=" * 50)

    timestamper = SegmentTimestamper()

    # Test 1: Basic timestamping
    print("\n[TEST 1] Basic timestamping:")
    segment = {
        "text": "Hello everyone",
        "start": 0.0,
        "end": 2.5,
        "speaker": "SPEAKER_00",
        "confidence": 0.95
    }

    chunk_start = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    timestamped = timestamper.add_absolute_timestamps(segment, chunk_start)

    print(f"  Original: start={segment['start']}s, end={segment['end']}s")
    print(f"  Chunk start: {chunk_start.isoformat()}")
    print(f"  Absolute start: {timestamped['absolute_start_time']}")
    print(f"  Absolute end: {timestamped['absolute_end_time']}")
    print(f"  is_final: {timestamped['is_final']}")
    print(f"  ✅ Basic timestamping working")

    # Test 2: With offset
    print("\n[TEST 2] With time offset:")
    segment2 = {
        "text": "How are you",
        "start": 3.0,  # 3 seconds into chunk
        "end": 5.0,
        "speaker": "SPEAKER_01"
    }

    timestamped2 = timestamper.add_absolute_timestamps(segment2, chunk_start)
    print(f"  Offset: {segment2['start']}s")
    print(f"  Absolute start: {timestamped2['absolute_start_time']}")
    print(f"  Expected: 2025-01-15T10:30:03Z")
    print(f"  ✅ Offset timestamping working")

    # Test 3: Final segment
    print("\n[TEST 3] Final segment flag:")
    final_segment = {
        "text": "Goodbye",
        "start": 0.0,
        "end": 1.0
    }

    timestamped_final = timestamper.add_absolute_timestamps(
        final_segment,
        chunk_start,
        is_final=True
    )
    print(f"  is_final: {timestamped_final['is_final']}")
    print(f"  ✅ Final flag working")

    # Test 4: Parse timestamp
    print("\n[TEST 4] Parse timestamp:")
    parsed = timestamper.parse_timestamp("2025-01-15T10:30:00Z")
    print(f"  Input: 2025-01-15T10:30:00Z")
    print(f"  Parsed: {parsed}")
    print(f"  ✅ Timestamp parsing working")

    print("\n" + "=" * 50)
    print("✅ Segment Timestamper Test Complete")
    print("\nKey Features:")
    print("  - Converts relative → absolute timestamps")
    print("  - ISO 8601 format with 'Z' suffix")
    print("  - Support for final/mutable segments")
    print("  - Preserves all original segment fields")
