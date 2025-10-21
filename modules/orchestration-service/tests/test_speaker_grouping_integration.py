#!/usr/bin/env python3
"""
COMPREHENSIVE INTEGRATION TESTS: Speaker Grouping (Orchestration Service)

Tests the speaker grouper that merges consecutive segments by speaker.

Following Vexa reference pattern (vexa/docs/websocket.md):
- Group consecutive segments by same speaker
- Preserve start time from first segment
- Preserve end time from last segment
- Concatenate text with spaces

NO MOCKS - Only real speaker grouping with real segment data!

Architecture:
    Deduplicator → Speaker Grouper → Frontend (readable transcript)
"""

import pytest
from datetime import datetime, timezone
import sys
from pathlib import Path

# Add src directory to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


class TestSpeakerGrouping:
    """
    Integration tests for speaker grouping
    """

    @pytest.mark.integration
    def test_consecutive_speaker_merging(self):
        """
        Test merging consecutive segments by same speaker

        Segments from same speaker in sequence → merge into one group
        """
        print("\n[SPEAKER GROUPING] Testing consecutive merging...")

        from speaker_grouper import SpeakerGrouper

        grouper = SpeakerGrouper()

        segments = [
            {
                "text": "Hello",
                "speaker": "John",
                "absolute_start_time": "2025-01-15T10:30:00Z",
                "absolute_end_time": "2025-01-15T10:30:01Z"
            },
            {
                "text": "everyone",  # Same speaker, consecutive
                "speaker": "John",
                "absolute_start_time": "2025-01-15T10:30:01Z",
                "absolute_end_time": "2025-01-15T10:30:02Z"
            },
            {
                "text": "how are you",  # Same speaker, consecutive
                "speaker": "John",
                "absolute_start_time": "2025-01-15T10:30:02Z",
                "absolute_end_time": "2025-01-15T10:30:03Z"
            }
        ]

        groups = grouper.group_by_speaker(segments)

        assert len(groups) == 1, "Should merge into 1 group"
        assert groups[0]["speaker"] == "John"
        assert groups[0]["text"] == "Hello everyone how are you"

        print(f"   Input: 3 segments (all from John)")
        print(f"   Output: {len(groups)} group")
        print(f"   Merged text: '{groups[0]['text']}'")
        print(f"✅ Consecutive speaker merging working")

    @pytest.mark.integration
    def test_speaker_change_creates_new_group(self):
        """
        Test that speaker change creates a new group

        Different speaker → new group (don't merge)
        """
        print("\n[SPEAKER GROUPING] Testing speaker changes...")

        from speaker_grouper import SpeakerGrouper

        grouper = SpeakerGrouper()

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
                "text": "Hi John",  # Different speaker
                "speaker": "Jane",
                "absolute_start_time": "2025-01-15T10:30:03Z",
                "absolute_end_time": "2025-01-15T10:30:04Z"
            },
            {
                "text": "how are you",  # Different speaker again
                "speaker": "Jane",
                "absolute_start_time": "2025-01-15T10:30:04Z",
                "absolute_end_time": "2025-01-15T10:30:05Z"
            }
        ]

        groups = grouper.group_by_speaker(segments)

        assert len(groups) == 2, "Should create 2 groups (John + Jane)"
        assert groups[0]["speaker"] == "John"
        assert groups[0]["text"] == "Hello everyone"
        assert groups[1]["speaker"] == "Jane"
        assert groups[1]["text"] == "Hi John how are you"

        print(f"   Input: 4 segments (2 John, 2 Jane)")
        print(f"   Output: {len(groups)} groups")
        print(f"   Group 1: {groups[0]['speaker']}: '{groups[0]['text']}'")
        print(f"   Group 2: {groups[1]['speaker']}: '{groups[1]['text']}'")
        print(f"✅ Speaker change handling working")

    @pytest.mark.integration
    def test_timing_preservation(self):
        """
        Test that group timing is preserved correctly

        Start time from first segment, end time from last segment
        """
        print("\n[SPEAKER GROUPING] Testing timing preservation...")

        from speaker_grouper import SpeakerGrouper

        grouper = SpeakerGrouper()

        segments = [
            {
                "text": "First",
                "speaker": "John",
                "absolute_start_time": "2025-01-15T10:30:00Z",
                "absolute_end_time": "2025-01-15T10:30:01Z"
            },
            {
                "text": "Second",
                "speaker": "John",
                "absolute_start_time": "2025-01-15T10:30:01Z",
                "absolute_end_time": "2025-01-15T10:30:02Z"
            },
            {
                "text": "Third",
                "speaker": "John",
                "absolute_start_time": "2025-01-15T10:30:02Z",
                "absolute_end_time": "2025-01-15T10:30:03Z"
            }
        ]

        groups = grouper.group_by_speaker(segments)

        assert len(groups) == 1
        assert groups[0]["start_time"] == "2025-01-15T10:30:00Z"  # First segment's start
        assert groups[0]["end_time"] == "2025-01-15T10:30:03Z"    # Last segment's end

        print(f"   Start time: {groups[0]['start_time']} (from first segment)")
        print(f"   End time: {groups[0]['end_time']} (from last segment)")
        print(f"✅ Timing preservation working")

    @pytest.mark.integration
    def test_empty_segment_handling(self):
        """
        Test that empty segments are skipped during grouping
        """
        print("\n[SPEAKER GROUPING] Testing empty segment handling...")

        from speaker_grouper import SpeakerGrouper

        grouper = SpeakerGrouper()

        segments = [
            {
                "text": "Hello",
                "speaker": "John",
                "absolute_start_time": "2025-01-15T10:30:00Z",
                "absolute_end_time": "2025-01-15T10:30:01Z"
            },
            {
                "text": "",  # Empty - should be skipped
                "speaker": "John",
                "absolute_start_time": "2025-01-15T10:30:01Z",
                "absolute_end_time": "2025-01-15T10:30:02Z"
            },
            {
                "text": "World",
                "speaker": "John",
                "absolute_start_time": "2025-01-15T10:30:02Z",
                "absolute_end_time": "2025-01-15T10:30:03Z"
            }
        ]

        groups = grouper.group_by_speaker(segments)

        assert len(groups) == 1
        assert groups[0]["text"] == "Hello World"  # Empty segment ignored

        print(f"   Input: 3 segments (1 empty)")
        print(f"   Output: 1 group with merged text")
        print(f"   Merged: '{groups[0]['text']}'")
        print(f"✅ Empty segment handling working")

    @pytest.mark.integration
    def test_alternating_speakers(self):
        """
        Test grouping with alternating speakers

        John → Jane → John → Jane
        Should create 4 groups (no merging across different speakers)
        """
        print("\n[SPEAKER GROUPING] Testing alternating speakers...")

        from speaker_grouper import SpeakerGrouper

        grouper = SpeakerGrouper()

        segments = [
            {"text": "Hello", "speaker": "John",
             "absolute_start_time": "2025-01-15T10:30:00Z",
             "absolute_end_time": "2025-01-15T10:30:01Z"},
            {"text": "Hi", "speaker": "Jane",
             "absolute_start_time": "2025-01-15T10:30:02Z",
             "absolute_end_time": "2025-01-15T10:30:03Z"},
            {"text": "How are you", "speaker": "John",
             "absolute_start_time": "2025-01-15T10:30:04Z",
             "absolute_end_time": "2025-01-15T10:30:05Z"},
            {"text": "Good", "speaker": "Jane",
             "absolute_start_time": "2025-01-15T10:30:06Z",
             "absolute_end_time": "2025-01-15T10:30:07Z"}
        ]

        groups = grouper.group_by_speaker(segments)

        assert len(groups) == 4  # No merging across speakers
        assert groups[0]["speaker"] == "John"
        assert groups[1]["speaker"] == "Jane"
        assert groups[2]["speaker"] == "John"
        assert groups[3]["speaker"] == "Jane"

        print(f"   Input: 4 segments (alternating John/Jane)")
        print(f"   Output: {len(groups)} groups (no cross-speaker merging)")
        print(f"✅ Alternating speakers working")


class TestDisplayFormatting:
    """
    Integration tests for display formatting
    """

    @pytest.mark.integration
    def test_format_group_for_display(self):
        """
        Test formatting a speaker group for terminal display

        Format: [HH:MM:SS - HH:MM:SS] Speaker: Text
        """
        print("\n[DISPLAY FORMATTING] Testing group formatting...")

        from speaker_grouper import SpeakerGrouper

        grouper = SpeakerGrouper()

        group = {
            "speaker": "John Doe",
            "text": "Hello everyone, how are you today?",
            "start_time": "2025-01-15T10:30:00Z",
            "end_time": "2025-01-15T10:30:05Z"
        }

        formatted = grouper.format_group_for_display(group)

        # Should contain speaker and text
        assert "John Doe" in formatted
        assert "Hello everyone, how are you today?" in formatted

        # Should have time format
        assert "10:30:00" in formatted or "[" in formatted

        print(f"   Formatted: {formatted}")
        print(f"✅ Display formatting working")


class TestSpeakerGroupingWithRealData:
    """
    Integration tests with realistic transcript data
    """

    @pytest.mark.integration
    def test_realistic_conversation(self):
        """
        Test grouping with realistic multi-speaker conversation
        """
        print("\n[REAL DATA] Testing realistic conversation...")

        from speaker_grouper import SpeakerGrouper

        grouper = SpeakerGrouper()

        # Realistic conversation: Interview
        segments = [
            {"text": "Welcome", "speaker": "Interviewer",
             "absolute_start_time": "2025-01-15T14:00:00Z",
             "absolute_end_time": "2025-01-15T14:00:01Z"},
            {"text": "to", "speaker": "Interviewer",
             "absolute_start_time": "2025-01-15T14:00:01Z",
             "absolute_end_time": "2025-01-15T14:00:02Z"},
            {"text": "our", "speaker": "Interviewer",
             "absolute_start_time": "2025-01-15T14:00:02Z",
             "absolute_end_time": "2025-01-15T14:00:03Z"},
            {"text": "show", "speaker": "Interviewer",
             "absolute_start_time": "2025-01-15T14:00:03Z",
             "absolute_end_time": "2025-01-15T14:00:04Z"},

            {"text": "Thank", "speaker": "Guest",
             "absolute_start_time": "2025-01-15T14:00:05Z",
             "absolute_end_time": "2025-01-15T14:00:06Z"},
            {"text": "you", "speaker": "Guest",
             "absolute_start_time": "2025-01-15T14:00:06Z",
             "absolute_end_time": "2025-01-15T14:00:07Z"},
            {"text": "for", "speaker": "Guest",
             "absolute_start_time": "2025-01-15T14:00:07Z",
             "absolute_end_time": "2025-01-15T14:00:08Z"},
            {"text": "having", "speaker": "Guest",
             "absolute_start_time": "2025-01-15T14:00:08Z",
             "absolute_end_time": "2025-01-15T14:00:09Z"},
            {"text": "me", "speaker": "Guest",
             "absolute_start_time": "2025-01-15T14:00:09Z",
             "absolute_end_time": "2025-01-15T14:00:10Z"},

            {"text": "Let's", "speaker": "Interviewer",
             "absolute_start_time": "2025-01-15T14:00:11Z",
             "absolute_end_time": "2025-01-15T14:00:12Z"},
            {"text": "get", "speaker": "Interviewer",
             "absolute_start_time": "2025-01-15T14:00:12Z",
             "absolute_end_time": "2025-01-15T14:00:13Z"},
            {"text": "started", "speaker": "Interviewer",
             "absolute_start_time": "2025-01-15T14:00:13Z",
             "absolute_end_time": "2025-01-15T14:00:14Z"}
        ]

        groups = grouper.group_by_speaker(segments)

        assert len(groups) == 3  # Interviewer → Guest → Interviewer
        assert groups[0]["speaker"] == "Interviewer"
        assert groups[0]["text"] == "Welcome to our show"
        assert groups[1]["speaker"] == "Guest"
        assert groups[1]["text"] == "Thank you for having me"
        assert groups[2]["speaker"] == "Interviewer"
        assert groups[2]["text"] == "Let's get started"

        print(f"   Input: 13 segments (conversation)")
        print(f"   Output: {len(groups)} groups")
        for i, group in enumerate(groups, 1):
            print(f"   {i}. {group['speaker']}: '{group['text']}'")
        print(f"✅ Realistic conversation grouping working")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
