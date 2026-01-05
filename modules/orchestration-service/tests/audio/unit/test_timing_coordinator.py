#!/usr/bin/env python3
"""
Unit Tests for TimingCoordinator

Tests focused on timestamp alignment, drift detection, and database correlation
across all bot_sessions tables. Validates precise timing coordination.
"""

import unittest
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta

# Add src to path for imports
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(orchestration_root))
sys.path.insert(0, str(src_path))

from src.audio.timing_coordinator import (
    TimingCoordinator,
    TimeWindow,
    TimingCorrelation,
    SynchronizationQuality,
    TimingAccuracy,
    CorrelationScope,
    create_timing_coordinator,
)


class TestTimeWindow(unittest.TestCase):
    """Test TimeWindow data class."""

    def test_time_window_creation(self):
        """Test creating time windows."""
        window = TimeWindow(
            start_timestamp=10.0,
            end_timestamp=15.0,
            confidence=0.9,
            source_table="audio_files",
            source_id="chunk_001",
        )

        self.assertEqual(window.start_timestamp, 10.0)
        self.assertEqual(window.end_timestamp, 15.0)
        self.assertEqual(window.confidence, 0.9)
        self.assertEqual(window.source_table, "audio_files")
        self.assertEqual(window.source_id, "chunk_001")
        self.assertEqual(window.duration, 5.0)

    def test_time_window_overlap_detection(self):
        """Test overlap detection between time windows."""
        window1 = TimeWindow(10.0, 15.0)
        window2 = TimeWindow(12.0, 18.0)  # Overlaps
        window3 = TimeWindow(20.0, 25.0)  # No overlap

        self.assertTrue(window1.overlaps_with(window2))
        self.assertTrue(window2.overlaps_with(window1))
        self.assertFalse(window1.overlaps_with(window3))
        self.assertFalse(window3.overlaps_with(window1))

    def test_time_window_intersection(self):
        """Test intersection calculation between time windows."""
        window1 = TimeWindow(10.0, 15.0)
        window2 = TimeWindow(12.0, 18.0)
        window3 = TimeWindow(20.0, 25.0)

        # Test overlap intersection
        intersection = window1.intersection_with(window2)
        self.assertIsNotNone(intersection)
        self.assertEqual(intersection.start_timestamp, 12.0)
        self.assertEqual(intersection.end_timestamp, 15.0)
        self.assertEqual(intersection.duration, 3.0)

        # Test no overlap
        no_intersection = window1.intersection_with(window3)
        self.assertIsNone(no_intersection)

    def test_time_window_edge_cases(self):
        """Test edge cases for time windows."""
        # Exact boundaries
        window1 = TimeWindow(10.0, 15.0)
        window2 = TimeWindow(15.0, 20.0)  # Touching but not overlapping

        # Touching windows should not overlap by default
        self.assertFalse(window1.overlaps_with(window2))

        # With tolerance, touching windows should overlap
        self.assertTrue(window1.overlaps_with(window2, tolerance=0.1))


class TestTimingCoordinatorInitialization(unittest.TestCase):
    """Test TimingCoordinator initialization."""

    def test_coordinator_creation_defaults(self):
        """Test creating coordinator with default settings."""
        coordinator = TimingCoordinator()

        self.assertIsNone(coordinator.database_adapter)
        self.assertEqual(coordinator.timing_tolerance_ms, 100.0)
        self.assertTrue(coordinator.drift_detection_enabled)
        self.assertTrue(coordinator.auto_correction_enabled)
        self.assertTrue(coordinator.quality_monitoring_enabled)
        self.assertEqual(coordinator.correlation_cache_size, 1000)

    def test_coordinator_creation_custom_settings(self):
        """Test creating coordinator with custom settings."""
        mock_db = MagicMock()

        coordinator = TimingCoordinator(
            database_adapter=mock_db,
            timing_tolerance_ms=50.0,
            drift_detection_enabled=False,
            auto_correction_enabled=False,
            quality_monitoring_enabled=False,
            correlation_cache_size=500,
        )

        self.assertEqual(coordinator.database_adapter, mock_db)
        self.assertEqual(coordinator.timing_tolerance_ms, 50.0)
        self.assertFalse(coordinator.drift_detection_enabled)
        self.assertFalse(coordinator.auto_correction_enabled)
        self.assertFalse(coordinator.quality_monitoring_enabled)
        self.assertEqual(coordinator.correlation_cache_size, 500)

    def test_factory_function(self):
        """Test factory function for creating coordinator."""
        coordinator = create_timing_coordinator(
            timing_tolerance_ms=75.0, drift_detection_enabled=True
        )

        self.assertIsInstance(coordinator, TimingCoordinator)
        self.assertEqual(coordinator.timing_tolerance_ms, 75.0)
        self.assertTrue(coordinator.drift_detection_enabled)


class TestAudioChunkTimestampCorrelation(unittest.TestCase):
    """Test audio chunk timestamp correlation functionality."""

    def setUp(self):
        """Set up test coordinator with mock database."""
        self.mock_db = AsyncMock()
        self.coordinator = TimingCoordinator(
            database_adapter=self.mock_db, timing_tolerance_ms=100.0
        )

    def test_audio_chunk_correlation_with_transcripts(self):
        """Test correlating audio chunks with transcripts."""
        session_id = "test_correlation_session"
        chunk_ids = ["chunk_001", "chunk_002"]

        # Mock audio chunks
        audio_chunks = [
            {"chunk_id": "chunk_001", "chunk_start_time": 0.0, "chunk_end_time": 3.0},
            {"chunk_id": "chunk_002", "chunk_start_time": 2.5, "chunk_end_time": 5.5},
        ]

        # Mock transcripts that overlap with chunks
        transcripts = [
            {
                "transcript_id": "trans_001",
                "start_timestamp": 0.5,
                "end_timestamp": 2.8,
            },
            {
                "transcript_id": "trans_002",
                "start_timestamp": 2.7,
                "end_timestamp": 5.2,
            },
        ]

        # Mock database responses
        self.coordinator._get_audio_chunks = AsyncMock(return_value=audio_chunks)
        self.coordinator._get_session_transcripts = AsyncMock(return_value=transcripts)
        self.coordinator._get_session_translations = AsyncMock(return_value=[])
        self.coordinator._get_session_speaker_correlations = AsyncMock(return_value=[])
        self.coordinator._store_timing_correlations = AsyncMock(return_value=True)

        async def run_test():
            correlations = await self.coordinator.correlate_audio_chunk_timestamps(
                session_id=session_id, audio_chunk_ids=chunk_ids
            )
            return correlations

        correlations = asyncio.run(run_test())

        # Verify correlations were created
        self.assertTrue(len(correlations) > 0)

        # Verify correlation properties
        for correlation in correlations:
            self.assertEqual(correlation.session_id, session_id)
            self.assertEqual(correlation.source_table, "audio_files")
            self.assertEqual(correlation.target_table, "transcripts")
            self.assertIsInstance(correlation.confidence_score, float)
            self.assertGreaterEqual(correlation.confidence_score, 0.0)
            self.assertLessEqual(correlation.confidence_score, 1.0)

    def test_audio_chunk_correlation_without_database(self):
        """Test correlation works gracefully without database."""
        coordinator = TimingCoordinator(database_adapter=None)

        async def run_test():
            correlations = await coordinator.correlate_audio_chunk_timestamps(
                session_id="test_session", audio_chunk_ids=["chunk_001"]
            )
            return correlations

        correlations = asyncio.run(run_test())

        # Should return empty list but not fail
        self.assertEqual(len(correlations), 0)

    def test_correlation_confidence_calculation(self):
        """Test correlation confidence calculation."""
        # Create test windows
        window1 = TimeWindow(0.0, 3.0, confidence=1.0)
        window2 = TimeWindow(1.0, 4.0, confidence=0.9)  # Partial overlap
        window3 = TimeWindow(0.0, 3.0, confidence=0.8)  # Exact overlap

        # Test confidence calculation
        confidence1 = self.coordinator._calculate_temporal_confidence(window1, window2)
        confidence2 = self.coordinator._calculate_temporal_confidence(window1, window3)

        # Exact overlap should have higher confidence
        self.assertGreater(confidence2, confidence1)
        self.assertGreaterEqual(confidence1, 0.0)
        self.assertLessEqual(confidence1, 1.0)
        self.assertGreaterEqual(confidence2, 0.0)
        self.assertLessEqual(confidence2, 1.0)

    def test_timing_accuracy_determination(self):
        """Test timing accuracy level determination."""
        # Test different offset values
        exact_accuracy = self.coordinator._determine_timing_accuracy(5.0)  # 5ms
        high_accuracy = self.coordinator._determine_timing_accuracy(30.0)  # 30ms
        medium_accuracy = self.coordinator._determine_timing_accuracy(150.0)  # 150ms
        low_accuracy = self.coordinator._determine_timing_accuracy(400.0)  # 400ms
        poor_accuracy = self.coordinator._determine_timing_accuracy(1000.0)  # 1000ms

        self.assertEqual(exact_accuracy, TimingAccuracy.EXACT)
        self.assertEqual(high_accuracy, TimingAccuracy.HIGH)
        self.assertEqual(medium_accuracy, TimingAccuracy.MEDIUM)
        self.assertEqual(low_accuracy, TimingAccuracy.LOW)
        self.assertEqual(poor_accuracy, TimingAccuracy.POOR)


class TestTimeDriftDetection(unittest.TestCase):
    """Test time drift detection functionality."""

    def setUp(self):
        """Set up test coordinator."""
        self.mock_db = AsyncMock()
        self.coordinator = TimingCoordinator(
            database_adapter=self.mock_db,
            timing_tolerance_ms=100.0,
            drift_detection_enabled=True,
            auto_correction_enabled=True,
        )

    def test_drift_detection_between_sources(self):
        """Test drift detection between different data sources."""
        session_id = "test_drift_session"

        # Mock timestamp retrieval
        self.coordinator._get_timestamps_for_source = AsyncMock()
        self.coordinator._get_timestamps_for_source.side_effect = [
            [10.0, 20.0, 30.0],  # Reference timestamps (audio_files)
            [10.05, 20.1, 30.15],  # Comparison timestamps (transcripts) - slight drift
            [9.9, 19.85, 29.8],  # Another comparison (translations) - negative drift
        ]

        self.coordinator._apply_drift_correction = AsyncMock(return_value=True)

        async def run_test():
            drift_measurements = await self.coordinator.detect_time_drift(
                session_id=session_id,
                reference_source="audio_files",
                comparison_sources=["transcripts", "translations"],
            )
            return drift_measurements

        drift_measurements = asyncio.run(run_test())

        # Verify drift was detected
        self.assertIn("transcripts", drift_measurements)
        self.assertIn("translations", drift_measurements)

        # Verify drift values are reasonable
        for source, drift in drift_measurements.items():
            self.assertIsInstance(drift, float)

    def test_drift_correction_application(self):
        """Test automatic drift correction application."""
        session_id = "test_correction_session"

        async def run_test():
            # Apply correction for significant drift
            result = await self.coordinator._apply_drift_correction(
                session_id=session_id,
                source="transcripts",
                drift=150.0,  # 150ms drift
            )
            return result

        result = asyncio.run(run_test())

        self.assertTrue(result)

        # Verify correction was stored
        correction_key = f"{session_id}_transcripts"
        self.assertIn(correction_key, self.coordinator.correction_offsets)
        self.assertEqual(
            self.coordinator.correction_offsets[correction_key], 0.15
        )  # 150ms = 0.15s

    def test_drift_statistics_tracking(self):
        """Test drift statistics tracking."""
        initial_corrections = self.coordinator.correlation_stats[
            "drift_corrections_applied"
        ]

        async def run_test():
            await self.coordinator._apply_drift_correction(
                "session_1", "transcripts", 200.0
            )
            await self.coordinator._apply_drift_correction(
                "session_2", "translations", 150.0
            )

        asyncio.run(run_test())

        # Verify statistics were updated
        final_corrections = self.coordinator.correlation_stats[
            "drift_corrections_applied"
        ]
        self.assertEqual(final_corrections, initial_corrections + 2)


class TestTimestampAlignment(unittest.TestCase):
    """Test timestamp alignment functionality."""

    def setUp(self):
        """Set up test coordinator."""
        self.mock_db = AsyncMock()
        self.coordinator = TimingCoordinator(
            database_adapter=self.mock_db, timing_tolerance_ms=100.0
        )

    def test_session_timestamp_alignment(self):
        """Test aligning timestamps for an entire session."""
        session_id = "test_alignment_session"

        # Mock quality assessment - poor quality that needs alignment
        poor_quality = SynchronizationQuality(
            session_id=session_id,
            overall_score=0.3,  # Poor quality
            timing_accuracy=TimingAccuracy.POOR,
            correlation_count=5,
            drift_detected=True,
            max_drift_ms=500.0,
            alignment_consistency=0.2,
        )

        # Mock improved quality after alignment
        good_quality = SynchronizationQuality(
            session_id=session_id,
            overall_score=0.9,  # Good quality
            timing_accuracy=TimingAccuracy.HIGH,
            correlation_count=5,
            drift_detected=False,
            max_drift_ms=50.0,
            alignment_consistency=0.95,
        )

        # Mock methods
        self.coordinator._assess_synchronization_quality = AsyncMock()
        self.coordinator._assess_synchronization_quality.side_effect = [
            poor_quality,
            good_quality,
        ]
        self.coordinator._get_complete_session_data = AsyncMock(
            return_value={"mock": "data"}
        )
        self.coordinator._calculate_optimal_baseline = AsyncMock(return_value=0.0)
        self.coordinator._apply_timestamp_alignment = AsyncMock(return_value=True)

        async def run_test():
            result = await self.coordinator.align_session_timestamps(
                session_id=session_id, force_realignment=False
            )
            return result

        result = asyncio.run(run_test())

        self.assertTrue(result)

        # Verify quality was stored
        self.assertIn(session_id, self.coordinator.synchronization_quality)
        stored_quality = self.coordinator.synchronization_quality[session_id]
        self.assertEqual(stored_quality.overall_score, 0.9)

    def test_alignment_skipped_for_good_quality(self):
        """Test that alignment is skipped for sessions with good quality."""
        session_id = "test_good_quality_session"

        # Mock high quality assessment
        high_quality = SynchronizationQuality(
            session_id=session_id,
            overall_score=0.95,  # Excellent quality
            timing_accuracy=TimingAccuracy.EXACT,
            correlation_count=10,
            drift_detected=False,
            max_drift_ms=10.0,
            alignment_consistency=0.98,
        )

        self.coordinator._assess_synchronization_quality = AsyncMock(
            return_value=high_quality
        )

        async def run_test():
            result = await self.coordinator.align_session_timestamps(
                session_id=session_id, force_realignment=False
            )
            return result

        result = asyncio.run(run_test())

        self.assertTrue(result)

        # Verify alignment was successful without triggering full alignment process


class TestSynchronizedDataRetrieval(unittest.TestCase):
    """Test synchronized data retrieval functionality."""

    def setUp(self):
        """Set up test coordinator with correlations."""
        self.coordinator = TimingCoordinator()

        # Create test correlations
        test_correlations = [
            TimingCorrelation(
                correlation_id="corr_001",
                session_id="test_session",
                source_table="audio_files",
                source_id="chunk_001",
                target_table="transcripts",
                target_id="trans_001",
                time_offset=0.05,
                confidence_score=0.9,
                accuracy_level=TimingAccuracy.HIGH,
                correlation_method="temporal_overlap",
            ),
            TimingCorrelation(
                correlation_id="corr_002",
                session_id="test_session",
                source_table="audio_files",
                source_id="chunk_002",
                target_table="translations",
                target_id="trans_002",
                time_offset=0.02,
                confidence_score=0.95,
                accuracy_level=TimingAccuracy.EXACT,
                correlation_method="temporal_overlap",
            ),
        ]

        self.coordinator.timing_correlations["test_session"] = test_correlations

        # Create test quality
        test_quality = SynchronizationQuality(
            session_id="test_session",
            overall_score=0.92,
            timing_accuracy=TimingAccuracy.HIGH,
            correlation_count=2,
            drift_detected=False,
            max_drift_ms=50.0,
            alignment_consistency=0.9,
        )

        self.coordinator.synchronization_quality["test_session"] = test_quality

    def test_get_synchronized_data_with_correlations(self):
        """Test getting synchronized data with existing correlations."""
        session_id = "test_session"

        async def run_test():
            data = await self.coordinator.get_synchronized_data(
                session_id=session_id, include_correlations=True
            )
            return data

        data = asyncio.run(run_test())

        # Verify data structure
        self.assertEqual(data["session_id"], session_id)
        self.assertIn("correlations", data)
        self.assertEqual(len(data["correlations"]), 2)
        self.assertIn("synchronization_quality", data)

        # Verify correlation data
        correlation = data["correlations"][0]
        self.assertIn("source_table", correlation)
        self.assertIn("target_table", correlation)
        self.assertIn("confidence", correlation)
        self.assertIn("accuracy", correlation)

        # Verify quality data
        quality = data["synchronization_quality"]
        self.assertEqual(quality["overall_score"], 0.92)
        self.assertEqual(quality["correlation_count"], 2)

    def test_get_synchronized_data_with_time_range(self):
        """Test getting synchronized data with time range filtering."""
        session_id = "test_session"
        time_range = (0.0, 5.0)

        async def run_test():
            data = await self.coordinator.get_synchronized_data(
                session_id=session_id, time_range=time_range
            )
            return data

        data = asyncio.run(run_test())

        # Verify time range is included when provided
        self.assertIn("time_range", data)
        self.assertEqual(data["time_range"], time_range)


class TestStatisticsAndMonitoring(unittest.TestCase):
    """Test statistics and monitoring functionality."""

    def test_timing_statistics_initial_state(self):
        """Test initial timing statistics."""
        coordinator = TimingCoordinator()

        stats = coordinator.get_timing_statistics()

        self.assertEqual(stats["total_correlations_attempted"], 0)
        self.assertEqual(stats["successful_correlations"], 0)
        self.assertEqual(stats["failed_correlations"], 0)
        self.assertEqual(stats["drift_corrections_applied"], 0)
        self.assertEqual(stats["active_sessions"], 0)
        self.assertEqual(stats["quality_assessments"], 0)

    def test_timing_statistics_after_operations(self):
        """Test timing statistics after operations."""
        coordinator = TimingCoordinator()

        # Simulate some operations
        coordinator.correlation_stats["total_correlations_attempted"] = 5
        coordinator.correlation_stats["successful_correlations"] = 4
        coordinator.correlation_stats["failed_correlations"] = 1
        coordinator.correlation_stats["drift_corrections_applied"] = 2

        # Add some session data
        coordinator.timing_correlations["session_1"] = []
        coordinator.timing_correlations["session_2"] = []
        coordinator.synchronization_quality["session_1"] = SynchronizationQuality(
            session_id="session_1",
            overall_score=0.8,
            timing_accuracy=TimingAccuracy.HIGH,
            correlation_count=3,
            drift_detected=False,
            max_drift_ms=25.0,
            alignment_consistency=0.9,
        )

        stats = coordinator.get_timing_statistics()

        self.assertEqual(stats["total_correlations_attempted"], 5)
        self.assertEqual(stats["successful_correlations"], 4)
        self.assertEqual(stats["failed_correlations"], 1)
        self.assertEqual(stats["drift_corrections_applied"], 2)
        self.assertEqual(stats["active_sessions"], 2)
        self.assertEqual(stats["quality_assessments"], 1)


class TestSessionManagement(unittest.TestCase):
    """Test session-specific timing management."""

    def setUp(self):
        """Set up test coordinator with session data."""
        self.coordinator = TimingCoordinator()

        # Add test session data
        self.coordinator.timing_correlations["test_session"] = [
            TimingCorrelation(
                correlation_id="corr_001",
                session_id="test_session",
                source_table="audio_files",
                source_id="chunk_001",
                target_table="transcripts",
                target_id="trans_001",
                time_offset=0.05,
                confidence_score=0.9,
                accuracy_level=TimingAccuracy.HIGH,
                correlation_method="temporal_overlap",
            )
        ]

        self.coordinator.synchronization_quality["test_session"] = (
            SynchronizationQuality(
                session_id="test_session",
                overall_score=0.85,
                timing_accuracy=TimingAccuracy.HIGH,
                correlation_count=1,
                drift_detected=False,
                max_drift_ms=50.0,
                alignment_consistency=0.9,
            )
        )

        self.coordinator.detected_drifts["test_session"] = [25.0, 30.0]
        self.coordinator.correction_offsets["test_session_transcripts"] = 0.025

    def test_clear_session_timing_data(self):
        """Test clearing timing data for a session."""
        session_id = "test_session"

        # Verify data exists before clearing
        self.assertIn(session_id, self.coordinator.timing_correlations)
        self.assertIn(session_id, self.coordinator.synchronization_quality)
        self.assertIn(session_id, self.coordinator.detected_drifts)
        self.assertIn(f"{session_id}_transcripts", self.coordinator.correction_offsets)

        async def run_test():
            result = await self.coordinator.clear_session_timing_data(session_id)
            return result

        result = asyncio.run(run_test())

        self.assertTrue(result)

        # Verify data was cleared
        self.assertNotIn(session_id, self.coordinator.timing_correlations)
        self.assertNotIn(session_id, self.coordinator.synchronization_quality)
        self.assertNotIn(session_id, self.coordinator.detected_drifts)
        self.assertNotIn(
            f"{session_id}_transcripts", self.coordinator.correction_offsets
        )


class TestTimingCoordinatorIntegration(unittest.TestCase):
    """Test end-to-end timing coordinator integration."""

    def test_complete_timing_workflow(self):
        """Test complete timing coordination workflow."""
        mock_db = AsyncMock()
        coordinator = create_timing_coordinator(
            database_adapter=mock_db,
            timing_tolerance_ms=100.0,
            drift_detection_enabled=True,
            auto_correction_enabled=True,
        )

        session_id = "complete_workflow_test"

        # Mock comprehensive session data
        audio_chunks = [
            {"chunk_id": "chunk_001", "chunk_start_time": 0.0, "chunk_end_time": 3.0},
            {"chunk_id": "chunk_002", "chunk_start_time": 2.5, "chunk_end_time": 5.5},
        ]

        transcripts = [
            {
                "transcript_id": "trans_001",
                "start_timestamp": 0.1,
                "end_timestamp": 2.9,
            },
            {
                "transcript_id": "trans_002",
                "start_timestamp": 2.6,
                "end_timestamp": 5.4,
            },
        ]

        # Mock database methods
        coordinator._get_audio_chunks = AsyncMock(return_value=audio_chunks)
        coordinator._get_session_transcripts = AsyncMock(return_value=transcripts)
        coordinator._get_session_translations = AsyncMock(return_value=[])
        coordinator._get_session_speaker_correlations = AsyncMock(return_value=[])
        coordinator._store_timing_correlations = AsyncMock(return_value=True)

        async def run_test():
            # 1. Correlate audio chunk timestamps
            correlations = await coordinator.correlate_audio_chunk_timestamps(
                session_id=session_id, audio_chunk_ids=["chunk_001", "chunk_002"]
            )

            # 2. Detect time drift (mock with no significant drift)
            coordinator._get_timestamps_for_source = AsyncMock(return_value=[0.0, 2.5])
            drift = await coordinator.detect_time_drift(session_id)

            # 3. Get synchronized data
            sync_data = await coordinator.get_synchronized_data(session_id)

            # 4. Get timing statistics
            stats = coordinator.get_timing_statistics()

            return correlations, drift, sync_data, stats

        correlations, drift, sync_data, stats = asyncio.run(run_test())

        # Verify complete workflow success
        self.assertTrue(len(correlations) > 0)
        self.assertIsInstance(drift, dict)
        self.assertIn("session_id", sync_data)
        self.assertIn("correlations", sync_data)
        self.assertGreater(stats["total_correlations_attempted"], 0)

        # Verify correlations were cached
        self.assertIn(session_id, coordinator.timing_correlations)

        # Verify quality assessment was created
        if coordinator.quality_monitoring_enabled:
            self.assertIn(session_id, coordinator.synchronization_quality)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
