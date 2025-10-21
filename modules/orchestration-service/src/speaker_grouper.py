#!/usr/bin/env python3
"""
Speaker Grouping for Real-Time Transcripts

Groups consecutive segments by same speaker for readability.
This improves transcript display by merging fragmented speech.

Following Vexa reference pattern (vexa/docs/websocket.md):
- Merge consecutive segments by speaker
- Preserve start time from first segment
- Preserve end time from last segment
- Concatenate text with spaces

Usage:
    grouper = SpeakerGrouper()
    segments = [...]  # List of transcript segments
    groups = grouper.group_by_speaker(segments)

    for group in groups:
        print(f"[{group['start_time']}] {group['speaker']}: {group['text']}")
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SpeakerGrouper:
    """
    Groups consecutive transcript segments by speaker

    This class implements the speaker grouping algorithm from Vexa's
    WebSocket documentation for improved transcript readability.
    """

    def __init__(self):
        """Initialize speaker grouper"""
        pass

    def group_by_speaker(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group consecutive segments by same speaker

        Algorithm (from Vexa reference):
        1. Iterate through sorted segments
        2. If same speaker as current group → merge text
        3. If different speaker → start new group
        4. Preserve timing: start from first, end from last

        Parameters:
            segments (List[Dict]): Sorted transcript segments
                Required fields:
                - text (str): Transcript text
                - speaker (str): Speaker identifier
                - absolute_start_time (str): ISO 8601 timestamp
                - absolute_end_time (str): ISO 8601 timestamp

        Returns:
            List[Dict]: Grouped segments
                Fields:
                - speaker (str): Speaker identifier
                - text (str): Concatenated text
                - start_time (str): Start timestamp (from first segment)
                - end_time (str): End timestamp (from last segment)

        Example:
            segments = [
                {"text": "Hello", "speaker": "John", "absolute_start_time": "..."},
                {"text": "world", "speaker": "John", "absolute_start_time": "..."},
                {"text": "Hi", "speaker": "Jane", "absolute_start_time": "..."}
            ]

            groups = grouper.group_by_speaker(segments)
            # [
            #   {"speaker": "John", "text": "Hello world", ...},
            #   {"speaker": "Jane", "text": "Hi", ...}
            # ]
        """
        if not segments:
            return []

        groups = []
        current_group = None

        for segment in segments:
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '').strip()

            # Skip empty segments
            if not text:
                continue

            # Check if we should merge with current group
            if current_group and current_group['speaker'] == speaker:
                # Same speaker - merge text and update end time
                current_group['text'] += ' ' + text
                current_group['end_time'] = segment['absolute_end_time']
            else:
                # Different speaker - save current group and start new one
                if current_group:
                    groups.append(current_group)

                current_group = {
                    'speaker': speaker,
                    'text': text,
                    'start_time': segment['absolute_start_time'],
                    'end_time': segment['absolute_end_time']
                }

        # Don't forget the last group!
        if current_group:
            groups.append(current_group)

        logger.debug(f"Grouped {len(segments)} segments into {len(groups)} speaker groups")
        return groups

    def format_group_for_display(self, group: Dict[str, Any]) -> str:
        """
        Format a speaker group for display

        Parameters:
            group (Dict): Speaker group with fields:
                - speaker (str)
                - text (str)
                - start_time (str)
                - end_time (str)

        Returns:
            str: Formatted string for display

        Example:
            "[10:30:00 - 10:30:05] John: Hello everyone, how are you today?"
        """
        speaker = group['speaker']
        text = group['text']

        # Format timestamps (show only time part)
        try:
            from datetime import datetime
            start = datetime.fromisoformat(group['start_time'].replace('Z', '+00:00'))
            end = datetime.fromisoformat(group['end_time'].replace('Z', '+00:00'))

            start_str = start.strftime('%H:%M:%S')
            end_str = end.strftime('%H:%M:%S')

            return f"[{start_str} - {end_str}] {speaker}: {text}"
        except Exception as e:
            logger.warning(f"Error formatting timestamps: {e}")
            return f"{speaker}: {text}"


# Example usage and testing
if __name__ == "__main__":
    print("Speaker Grouping Test")
    print("=" * 50)

    # Test data: consecutive segments from same speaker
    segments = [
        {
            "text": "Hello",
            "speaker": "John",
            "absolute_start_time": "2025-01-15T10:30:00Z",
            "absolute_end_time": "2025-01-15T10:30:01Z"
        },
        {
            "text": "everyone",
            "speaker": "John",
            "absolute_start_time": "2025-01-15T10:30:01Z",
            "absolute_end_time": "2025-01-15T10:30:02Z"
        },
        {
            "text": "how",
            "speaker": "John",
            "absolute_start_time": "2025-01-15T10:30:02Z",
            "absolute_end_time": "2025-01-15T10:30:03Z"
        },
        {
            "text": "Hi John",
            "speaker": "Jane",
            "absolute_start_time": "2025-01-15T10:30:04Z",
            "absolute_end_time": "2025-01-15T10:30:05Z"
        },
        {
            "text": "Thanks",
            "speaker": "John",
            "absolute_start_time": "2025-01-15T10:30:06Z",
            "absolute_end_time": "2025-01-15T10:30:07Z"
        }
    ]

    grouper = SpeakerGrouper()
    groups = grouper.group_by_speaker(segments)

    print(f"\nInput: {len(segments)} segments")
    print(f"Output: {len(groups)} speaker groups\n")

    for i, group in enumerate(groups, 1):
        formatted = grouper.format_group_for_display(group)
        print(f"{i}. {formatted}")

    print("\n" + "=" * 50)
    print("✅ Speaker Grouping Test Complete")
    print("\nKey Features:")
    print("  - Merges consecutive segments by speaker")
    print("  - Preserves start/end timestamps")
    print("  - Improves transcript readability")
    print("  - Handles speaker changes gracefully")
