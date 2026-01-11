#!/usr/bin/env python3
"""
COMPREHENSIVE INTEGRATION TESTS: Segment Deduplication (Orchestration Service)

Tests the segment deduplicator that merges transcript segments by absolute_start_time.

Following Vexa reference pattern (vexa/docs/websocket.md):
- Deduplicate by absolute_start_time (unique key)
- When both have updated_at, keep the newer one
- Discard segments with empty/whitespace-only text
- Maintain sorted order

NO MOCKS - Only real deduplication with real segment data!

Architecture:
    Whisper → Orchestration (segments) → Deduplicator → Frontend
"""

import pytest
import sys
from pathlib import Path

# Add src directory to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


class TestSegmentDeduplication:
    """
    Integration tests for segment deduplication by absolute_start_time
    """

    @pytest.mark.integration
    def test_deduplication_by_absolute_start_time(self):
        """
        Test basic deduplication using absolute_start_time as key

        Same absolute_start_time = same segment (should replace, not duplicate)
        """
        print("\n[DEDUPLICATION] Testing by absolute_start_time...")

        from segment_deduplicator import SegmentDeduplicator

        dedup = SegmentDeduplicator()

        # Initial segments
        initial = [
            {
                "text": "Hello",
                "speaker": "SPEAKER_00",
                "absolute_start_time": "2025-01-15T10:30:00Z",
                "absolute_end_time": "2025-01-15T10:30:01Z",
            },
            {
                "text": "World",
                "speaker": "SPEAKER_00",
                "absolute_start_time": "2025-01-15T10:30:02Z",
                "absolute_end_time": "2025-01-15T10:30:03Z",
            },
        ]

        dedup.merge_segments(initial)
        assert dedup.get_segment_count() == 2

        # Update first segment (same absolute_start_time)
        updated = [
            {
                "text": "Hello everyone",  # Updated text
                "speaker": "SPEAKER_00",
                "absolute_start_time": "2025-01-15T10:30:00Z",  # SAME KEY
                "absolute_end_time": "2025-01-15T10:30:01Z",
            }
        ]

        dedup.merge_segments(updated)

        # Should still have 2 segments (not 3)
        all_segments = dedup.get_all_segments()
        assert len(all_segments) == 2
        assert all_segments[0]["text"] == "Hello everyone"

        print("   Initial: 2 segments")
        print(f"   After update: {len(all_segments)} segments (deduplicated)")
        print(f"   Updated text: '{all_segments[0]['text']}'")
        print("✅ Deduplication by absolute_start_time working")

    @pytest.mark.integration
    def test_updated_at_precedence(self):
        """
        Test that updated_at timestamp determines which version to keep

        When both have updated_at: keep the newer one
        """
        print("\n[DEDUPLICATION] Testing updated_at precedence...")

        from segment_deduplicator import SegmentDeduplicator

        dedup = SegmentDeduplicator()

        # Initial segment with updated_at
        older = {
            "text": "Version 1",
            "speaker": "SPEAKER_00",
            "absolute_start_time": "2025-01-15T10:30:00Z",
            "absolute_end_time": "2025-01-15T10:30:01Z",
            "updated_at": "2025-01-15T10:30:01.000Z",
        }

        dedup.merge_segments([older])

        # Try to update with OLDER timestamp (should be ignored)
        even_older = {
            "text": "Version 0 (older)",
            "speaker": "SPEAKER_00",
            "absolute_start_time": "2025-01-15T10:30:00Z",
            "absolute_end_time": "2025-01-15T10:30:01Z",
            "updated_at": "2025-01-15T10:30:00.500Z",  # Older!
        }

        dedup.merge_segments([even_older])
        all_segments = dedup.get_all_segments()
        assert all_segments[0]["text"] == "Version 1"  # Should keep newer

        # Now update with NEWER timestamp (should replace)
        newer = {
            "text": "Version 2 (newest)",
            "speaker": "SPEAKER_00",
            "absolute_start_time": "2025-01-15T10:30:00Z",
            "absolute_end_time": "2025-01-15T10:30:01Z",
            "updated_at": "2025-01-15T10:30:02.000Z",  # Newer!
        }

        dedup.merge_segments([newer])
        all_segments = dedup.get_all_segments()
        assert all_segments[0]["text"] == "Version 2 (newest)"

        print("   Kept version with newest updated_at")
        print(f"   Final text: '{all_segments[0]['text']}'")
        print("✅ updated_at precedence working")

    @pytest.mark.integration
    def test_empty_segment_filtering(self):
        """
        Test that empty/whitespace-only segments are discarded

        Vexa pattern: Ignore segments without meaningful text
        """
        print("\n[DEDUPLICATION] Testing empty segment filtering...")

        from segment_deduplicator import SegmentDeduplicator

        dedup = SegmentDeduplicator()

        segments = [
            {
                "text": "Hello",
                "absolute_start_time": "2025-01-15T10:30:00Z",
                "absolute_end_time": "2025-01-15T10:30:01Z",
            },
            {
                "text": "",  # Empty - should be discarded
                "absolute_start_time": "2025-01-15T10:30:02Z",
                "absolute_end_time": "2025-01-15T10:30:03Z",
            },
            {
                "text": "   ",  # Whitespace only - should be discarded
                "absolute_start_time": "2025-01-15T10:30:04Z",
                "absolute_end_time": "2025-01-15T10:30:05Z",
            },
            {
                "text": "World",
                "absolute_start_time": "2025-01-15T10:30:06Z",
                "absolute_end_time": "2025-01-15T10:30:07Z",
            },
        ]

        dedup.merge_segments(segments)
        all_segments = dedup.get_all_segments()

        # Should only have "Hello" and "World"
        assert len(all_segments) == 2
        assert all_segments[0]["text"] == "Hello"
        assert all_segments[1]["text"] == "World"

        print("   Input: 4 segments (2 empty)")
        print(f"   Output: {len(all_segments)} segments (filtered)")
        print("✅ Empty segment filtering working")

    @pytest.mark.integration
    def test_sorted_output(self):
        """
        Test that segments are returned sorted by absolute_start_time
        """
        print("\n[DEDUPLICATION] Testing sorted output...")

        from segment_deduplicator import SegmentDeduplicator

        dedup = SegmentDeduplicator()

        # Insert in random order
        segments = [
            {
                "text": "Third",
                "absolute_start_time": "2025-01-15T10:30:06Z",
                "absolute_end_time": "2025-01-15T10:30:07Z",
            },
            {
                "text": "First",
                "absolute_start_time": "2025-01-15T10:30:00Z",
                "absolute_end_time": "2025-01-15T10:30:01Z",
            },
            {
                "text": "Second",
                "absolute_start_time": "2025-01-15T10:30:03Z",
                "absolute_end_time": "2025-01-15T10:30:04Z",
            },
        ]

        dedup.merge_segments(segments)
        sorted_segments = dedup.get_all_segments(sorted=True)

        # Verify order
        assert sorted_segments[0]["text"] == "First"
        assert sorted_segments[1]["text"] == "Second"
        assert sorted_segments[2]["text"] == "Third"

        print("   Insertion order: Third, First, Second")
        print(
            f"   Output order: {sorted_segments[0]['text']}, {sorted_segments[1]['text']}, {sorted_segments[2]['text']}"
        )
        print("✅ Sorted output working")


class TestRESTBootstrapPattern:
    """
    Integration tests for REST bootstrap + WebSocket updates pattern

    Tests the complete flow from Vexa docs
    """

    @pytest.mark.integration
    def test_rest_bootstrap_then_websocket_updates(self):
        """
        Test complete Vexa pattern:
        1. REST bootstrap: Initial transcript
        2. WebSocket updates: New segments + updates

        This verifies the entire deduplication workflow
        """
        print("\n[REST BOOTSTRAP] Testing REST + WebSocket pattern...")

        from segment_deduplicator import SegmentDeduplicator

        dedup = SegmentDeduplicator()

        # 1. REST Bootstrap: Fetch initial transcript
        rest_response = [
            {
                "text": "Hello",
                "speaker": "John",
                "absolute_start_time": "2025-01-15T10:30:00Z",
                "absolute_end_time": "2025-01-15T10:30:01Z",
            },
            {
                "text": "World",
                "speaker": "John",
                "absolute_start_time": "2025-01-15T10:30:02Z",
                "absolute_end_time": "2025-01-15T10:30:03Z",
            },
        ]

        dedup.merge_segments(rest_response)
        assert dedup.get_segment_count() == 2
        print(f"   REST bootstrap: {dedup.get_segment_count()} segments")

        # 2. WebSocket Update: Mutable transcript
        ws_update_1 = [
            {
                "text": "Hello everyone",  # Updated first segment
                "speaker": "John",
                "absolute_start_time": "2025-01-15T10:30:00Z",
                "absolute_end_time": "2025-01-15T10:30:01Z",
                "updated_at": "2025-01-15T10:30:05Z",
            }
        ]

        dedup.merge_segments(ws_update_1)
        assert dedup.get_segment_count() == 2  # Still 2 (updated, not added)

        # 3. WebSocket Update: New segment
        ws_update_2 = [
            {
                "text": "Nice to meet you",  # New segment
                "speaker": "Jane",
                "absolute_start_time": "2025-01-15T10:30:04Z",
                "absolute_end_time": "2025-01-15T10:30:06Z",
            }
        ]

        dedup.merge_segments(ws_update_2)
        assert dedup.get_segment_count() == 3  # Now 3 (new added)

        # Verify final state
        all_segments = dedup.get_all_segments()
        assert all_segments[0]["text"] == "Hello everyone"  # Updated
        assert all_segments[2]["text"] == "Nice to meet you"  # New

        print(f"   After WS update 1: {dedup.get_segment_count()} segments (1 updated)")
        print(f"   After WS update 2: {dedup.get_segment_count()} segments (1 new)")
        print("✅ REST + WebSocket pattern working")


class TestDeduplicationPerformance:
    """
    Integration tests for deduplication performance

    Tests efficiency with large numbers of segments
    """

    @pytest.mark.integration
    def test_large_segment_set_performance(self):
        """
        Test performance with 1000+ segments

        Should handle efficiently (O(1) lookups with dict)
        """
        print("\n[PERFORMANCE] Testing large segment set...")

        import time
        from segment_deduplicator import SegmentDeduplicator

        dedup = SegmentDeduplicator()

        # Generate 1000 segments
        segments = []
        for i in range(1000):
            segments.append(
                {
                    "text": f"Segment {i}",
                    "speaker": f"SPEAKER_{i % 5}",
                    "absolute_start_time": f"2025-01-15T10:{(i // 60):02d}:{(i % 60):02d}Z",
                    "absolute_end_time": f"2025-01-15T10:{(i // 60):02d}:{((i % 60) + 1):02d}Z",
                }
            )

        start_time = time.time()
        dedup.merge_segments(segments)
        end_time = time.time()

        processing_time = (end_time - start_time) * 1000  # milliseconds

        assert dedup.get_segment_count() == 1000
        assert processing_time < 100  # Should be fast (<100ms)

        print("   Segments: 1000")
        print(f"   Processing time: {processing_time:.2f}ms")
        print("   Target: <100ms")
        print("✅ Large segment set performance good")

    @pytest.mark.integration
    def test_incremental_updates_performance(self):
        """
        Test performance of many small incremental updates

        Simulates real-time streaming scenario
        """
        print("\n[PERFORMANCE] Testing incremental updates...")

        import time
        from segment_deduplicator import SegmentDeduplicator

        dedup = SegmentDeduplicator()

        # Add 100 segments one at a time (simulating streaming)
        total_time = 0
        for i in range(100):
            segment = {
                "text": f"Segment {i}",
                "absolute_start_time": f"2025-01-15T10:30:{i:02d}Z",
                "absolute_end_time": f"2025-01-15T10:30:{i + 1:02d}Z",
            }

            start_time = time.time()
            dedup.merge_segments([segment])
            end_time = time.time()

            total_time += (end_time - start_time) * 1000

        avg_time = total_time / 100

        assert dedup.get_segment_count() == 100
        assert avg_time < 1.0  # Each update should be <1ms

        print("   Updates: 100")
        print(f"   Average time per update: {avg_time:.3f}ms")
        print("   Target: <1ms")
        print("✅ Incremental updates performance excellent")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
