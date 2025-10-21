#!/usr/bin/env python3
"""
Segment Deduplication for Real-Time Transcripts

Deduplicates transcript segments using absolute_start_time as key.
Implements the Vexa WebSocket protocol deduplication algorithm.

Following Vexa reference pattern (vexa/docs/websocket.md):
- Use absolute_start_time as unique key
- When both have updated_at, keep the newer one
- Discard segments with empty/whitespace-only text
- Maintain sorted order by absolute_start_time

Usage:
    deduplicator = SegmentDeduplicator()

    # Initial segments (e.g., from REST bootstrap)
    deduplicator.merge_segments(rest_segments)

    # WebSocket updates
    deduplicator.merge_segments(ws_segments)

    # Get all deduplicated segments
    segments = deduplicator.get_all_segments()
"""

from typing import List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SegmentDeduplicator:
    """
    Deduplicates transcript segments by absolute_start_time

    This class implements the deduplication algorithm from Vexa's
    WebSocket documentation, ensuring no duplicate segments.
    """

    def __init__(self):
        """
        Initialize segment deduplicator

        Uses a dictionary keyed by absolute_start_time for O(1) lookups
        """
        self.segments_by_abs_start: Dict[str, Dict[str, Any]] = {}

    def merge_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge new segments with existing segments (upsert by absolute_start_time)

        Algorithm (from Vexa reference):
        1. For each segment with absolute_start_time:
           - If text is empty/whitespace → skip
           - If segment exists with updated_at on both → keep newer
           - Otherwise → upsert (add or replace)

        Parameters:
            segments (List[Dict]): New segments to merge
                Required fields:
                - absolute_start_time (str): ISO 8601 timestamp (key)
                - text (str): Transcript text
                Optional fields:
                - updated_at (str): ISO 8601 timestamp for precedence

        Returns:
            List[Dict]: Segments that were added/updated

        Example:
            # Initial segments
            deduplicator.merge_segments([
                {"text": "Hello", "absolute_start_time": "2025-01-15T10:30:00Z"}
            ])

            # Update same segment (same absolute_start_time)
            deduplicator.merge_segments([
                {"text": "Hello everyone", "absolute_start_time": "2025-01-15T10:30:00Z",
                 "updated_at": "2025-01-15T10:30:05Z"}
            ])

            # Result: "Hello everyone" (updated)
        """
        merged = []

        for segment in segments:
            abs_start = segment.get('absolute_start_time')
            text = segment.get('text', '').strip()

            # Skip segments without absolute_start_time
            if not abs_start:
                logger.debug("Skipping segment without absolute_start_time")
                continue

            # Skip segments with empty/whitespace-only text
            if not text:
                logger.debug(f"Skipping empty segment at {abs_start}")
                continue

            # Check if we already have this segment
            existing = self.segments_by_abs_start.get(abs_start)

            if existing:
                # Both have updated_at → keep newer
                if existing.get('updated_at') and segment.get('updated_at'):
                    if self._is_newer(segment['updated_at'], existing['updated_at']):
                        self.segments_by_abs_start[abs_start] = segment
                        merged.append(segment)
                        logger.debug(f"Updated segment at {abs_start} (newer updated_at)")
                    else:
                        logger.debug(f"Kept existing segment at {abs_start} (newer)")
                else:
                    # No updated_at comparison → just update
                    self.segments_by_abs_start[abs_start] = segment
                    merged.append(segment)
                    logger.debug(f"Updated segment at {abs_start}")
            else:
                # New segment → add
                self.segments_by_abs_start[abs_start] = segment
                merged.append(segment)
                logger.debug(f"Added new segment at {abs_start}")

        logger.info(f"Merged {len(merged)} segments, total: {len(self.segments_by_abs_start)}")
        return merged

    def get_all_segments(self, sorted: bool = True) -> List[Dict[str, Any]]:
        """
        Get all segments

        Parameters:
            sorted (bool): If True, sort by absolute_start_time ascending

        Returns:
            List[Dict]: All segments

        Example:
            segments = deduplicator.get_all_segments()
            for segment in segments:
                print(f"{segment['absolute_start_time']}: {segment['text']}")
        """
        segments = list(self.segments_by_abs_start.values())

        if sorted:
            segments.sort(key=lambda s: s['absolute_start_time'])

        return segments

    def get_segment_count(self) -> int:
        """
        Get total number of unique segments

        Returns:
            int: Segment count
        """
        return len(self.segments_by_abs_start)

    def clear(self):
        """
        Clear all segments

        Useful for starting a new session
        """
        self.segments_by_abs_start.clear()
        logger.info("Cleared all segments")

    def _is_newer(self, timestamp1: str, timestamp2: str) -> bool:
        """
        Compare two ISO 8601 timestamps

        Parameters:
            timestamp1 (str): First timestamp
            timestamp2 (str): Second timestamp

        Returns:
            bool: True if timestamp1 > timestamp2
        """
        try:
            dt1 = datetime.fromisoformat(timestamp1.replace('Z', '+00:00'))
            dt2 = datetime.fromisoformat(timestamp2.replace('Z', '+00:00'))
            return dt1 > dt2
        except Exception as e:
            logger.warning(f"Error comparing timestamps: {e}")
            # If comparison fails, assume newer
            return True

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get deduplicator statistics

        Returns:
            Dict: Statistics
                - total_segments (int): Total unique segments
                - speakers (List[str]): Unique speakers
                - time_range (Dict): Start and end times
        """
        segments = self.get_all_segments()

        if not segments:
            return {
                "total_segments": 0,
                "speakers": [],
                "time_range": None
            }

        speakers = list(set(s.get('speaker', 'Unknown') for s in segments))

        return {
            "total_segments": len(segments),
            "speakers": speakers,
            "time_range": {
                "start": segments[0]['absolute_start_time'],
                "end": segments[-1]['absolute_end_time']
            }
        }


# Example usage and testing
if __name__ == "__main__":
    print("Segment Deduplication Test")
    print("=" * 50)

    deduplicator = SegmentDeduplicator()

    # Test 1: Initial segments
    print("\n[TEST 1] Initial segments:")
    initial_segments = [
        {
            "text": "Hello",
            "speaker": "John",
            "absolute_start_time": "2025-01-15T10:30:00Z",
            "absolute_end_time": "2025-01-15T10:30:01Z"
        },
        {
            "text": "World",
            "speaker": "John",
            "absolute_start_time": "2025-01-15T10:30:02Z",
            "absolute_end_time": "2025-01-15T10:30:03Z"
        }
    ]

    deduplicator.merge_segments(initial_segments)
    print(f"  Added {len(initial_segments)} segments")
    print(f"  Total: {deduplicator.get_segment_count()} unique segments")

    # Test 2: Update existing segment
    print("\n[TEST 2] Update existing segment:")
    updated_segments = [
        {
            "text": "Hello everyone",  # Updated text
            "speaker": "John",
            "absolute_start_time": "2025-01-15T10:30:00Z",  # Same key!
            "absolute_end_time": "2025-01-15T10:30:01Z",
            "updated_at": "2025-01-15T10:30:05Z"
        }
    ]

    deduplicator.merge_segments(updated_segments)
    all_segments = deduplicator.get_all_segments()
    print(f"  Total: {deduplicator.get_segment_count()} unique segments (should still be 2)")
    print(f"  First segment text: '{all_segments[0]['text']}' (should be updated)")

    # Test 3: Add new segment + skip empty
    print("\n[TEST 3] Add new + skip empty:")
    new_segments = [
        {
            "text": "Nice to meet you",
            "speaker": "Jane",
            "absolute_start_time": "2025-01-15T10:30:04Z",
            "absolute_end_time": "2025-01-15T10:30:05Z"
        },
        {
            "text": "",  # Empty - should be skipped
            "speaker": "John",
            "absolute_start_time": "2025-01-15T10:30:06Z",
            "absolute_end_time": "2025-01-15T10:30:07Z"
        }
    ]

    deduplicator.merge_segments(new_segments)
    print(f"  Total: {deduplicator.get_segment_count()} unique segments (added 1, skipped 1)")

    # Test 4: Get statistics
    print("\n[TEST 4] Statistics:")
    stats = deduplicator.get_statistics()
    print(f"  Total segments: {stats['total_segments']}")
    print(f"  Speakers: {', '.join(stats['speakers'])}")
    print(f"  Time range: {stats['time_range']['start']} to {stats['time_range']['end']}")

    # Test 5: Display all segments
    print("\n[TEST 5] All segments (sorted):")
    for i, segment in enumerate(deduplicator.get_all_segments(), 1):
        start = segment['absolute_start_time'][11:19]  # Extract time part
        print(f"  {i}. [{start}] {segment['speaker']}: {segment['text']}")

    print("\n" + "=" * 50)
    print("✅ Segment Deduplication Test Complete")
    print("\nKey Features:")
    print("  - Deduplication by absolute_start_time")
    print("  - updated_at precedence (keep newer)")
    print("  - Empty segment filtering")
    print("  - O(1) lookups with dictionary")
