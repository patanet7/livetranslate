#!/usr/bin/env python3
"""
Unit Tests for SpeakerCorrelationManager

Tests focused on manual testing workflow, loopback audio, and graceful fallbacks.
These tests verify the system works properly for your testing scenarios.
"""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add src to path for imports
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(orchestration_root))
sys.path.insert(0, str(src_path))

from src.audio.speaker_correlator import (
    CorrelationMethod,
    LoopbackSpeakerInfo,
    ManualSpeakerMapping,
    SpeakerCorrelationManager,
    create_speaker_correlation_manager,
)


class TestManualSpeakerMapping(unittest.TestCase):
    """Test ManualSpeakerMapping data class."""

    def test_manual_mapping_creation(self):
        """Test creating manual speaker mappings."""
        mapping = ManualSpeakerMapping(
            whisper_speaker_id="speaker_0",
            display_name="Test User",
            real_name="John Doe",
            notes="Primary speaker for testing",
            confidence=1.0,
        )

        self.assertEqual(mapping.whisper_speaker_id, "speaker_0")
        self.assertEqual(mapping.display_name, "Test User")
        self.assertEqual(mapping.real_name, "John Doe")
        self.assertEqual(mapping.notes, "Primary speaker for testing")
        self.assertEqual(mapping.confidence, 1.0)

    def test_manual_mapping_defaults(self):
        """Test manual mapping with default values."""
        mapping = ManualSpeakerMapping(whisper_speaker_id="speaker_1", display_name="Test User 2")

        self.assertEqual(mapping.whisper_speaker_id, "speaker_1")
        self.assertEqual(mapping.display_name, "Test User 2")
        self.assertIsNone(mapping.real_name)
        self.assertIsNone(mapping.notes)
        self.assertEqual(mapping.confidence, 1.0)  # Default confidence

    def test_mapping_to_dict(self):
        """Test converting mapping to dictionary."""
        mapping = ManualSpeakerMapping(whisper_speaker_id="speaker_0", display_name="Test User")

        result = mapping.to_dict()

        self.assertIsInstance(result, dict)
        self.assertEqual(result["whisper_speaker_id"], "speaker_0")
        self.assertEqual(result["display_name"], "Test User")
        self.assertEqual(result["confidence"], 1.0)


class TestLoopbackSpeakerInfo(unittest.TestCase):
    """Test LoopbackSpeakerInfo data class."""

    def test_loopback_info_creation(self):
        """Test creating loopback speaker info."""
        info = LoopbackSpeakerInfo(
            estimated_speaker_count=2,
            primary_speaker_name="Main Speaker",
            secondary_speakers=["Secondary Speaker"],
            audio_source_description="System Audio Loopback",
            mixing_detected=True,
        )

        self.assertEqual(info.estimated_speaker_count, 2)
        self.assertEqual(info.primary_speaker_name, "Main Speaker")
        self.assertEqual(info.secondary_speakers, ["Secondary Speaker"])
        self.assertEqual(info.audio_source_description, "System Audio Loopback")
        self.assertTrue(info.mixing_detected)

    def test_loopback_info_defaults(self):
        """Test loopback info with default values."""
        info = LoopbackSpeakerInfo()

        self.assertEqual(info.estimated_speaker_count, 1)
        self.assertEqual(info.primary_speaker_name, "Primary Speaker")
        self.assertEqual(info.secondary_speakers, [])
        self.assertEqual(info.audio_source_description, "Loopback Audio")
        self.assertFalse(info.mixing_detected)

    def test_loopback_to_dict(self):
        """Test converting loopback info to dictionary."""
        info = LoopbackSpeakerInfo(estimated_speaker_count=3, primary_speaker_name="Test Primary")

        result = info.to_dict()

        self.assertIsInstance(result, dict)
        self.assertEqual(result["estimated_speaker_count"], 3)
        self.assertEqual(result["primary_speaker_name"], "Test Primary")


class TestSpeakerCorrelationManagerInitialization(unittest.TestCase):
    """Test SpeakerCorrelationManager initialization."""

    def test_manager_creation_defaults(self):
        """Test creating manager with default settings."""
        manager = SpeakerCorrelationManager()

        self.assertIsNone(manager.database_adapter)
        self.assertFalse(manager.enable_google_meet_correlation)
        self.assertTrue(manager.enable_manual_correlation)
        self.assertTrue(manager.enable_fallback_correlation)
        self.assertEqual(manager.correlation_timeout, 5.0)
        self.assertEqual(manager.min_confidence_threshold, 0.3)

    def test_manager_creation_custom_settings(self):
        """Test creating manager with custom settings."""
        mock_db = MagicMock()

        manager = SpeakerCorrelationManager(
            database_adapter=mock_db,
            enable_google_meet_correlation=True,
            enable_manual_correlation=False,
            enable_fallback_correlation=False,
            correlation_timeout=10.0,
            min_confidence_threshold=0.7,
        )

        self.assertEqual(manager.database_adapter, mock_db)
        self.assertTrue(manager.enable_google_meet_correlation)
        self.assertFalse(manager.enable_manual_correlation)
        self.assertFalse(manager.enable_fallback_correlation)
        self.assertEqual(manager.correlation_timeout, 10.0)
        self.assertEqual(manager.min_confidence_threshold, 0.7)

    def test_factory_function(self):
        """Test factory function for creating manager."""
        manager = create_speaker_correlation_manager(
            enable_google_meet_correlation=False, enable_manual_correlation=True
        )

        self.assertIsInstance(manager, SpeakerCorrelationManager)
        self.assertFalse(manager.enable_google_meet_correlation)
        self.assertTrue(manager.enable_manual_correlation)


class TestManualSpeakerCorrelation(unittest.TestCase):
    """Test manual speaker correlation functionality."""

    def setUp(self):
        """Set up test manager."""
        self.manager = SpeakerCorrelationManager(
            enable_manual_correlation=True,
            enable_google_meet_correlation=False,
            enable_fallback_correlation=True,
        )

    def test_set_manual_mapping_success(self):
        """Test setting manual speaker mappings."""
        session_id = "test_session_manual"
        mappings = [
            ManualSpeakerMapping("speaker_0", "Alice", "Alice Johnson"),
            ManualSpeakerMapping("speaker_1", "Bob", "Bob Smith"),
        ]

        async def run_test():
            result = await self.manager.set_manual_speaker_mapping(session_id, mappings)
            return result

        result = asyncio.run(run_test())

        self.assertTrue(result)
        self.assertIn(session_id, self.manager.manual_mappings)
        self.assertEqual(len(self.manager.manual_mappings[session_id]), 2)
        self.assertEqual(self.manager.manual_mappings[session_id][0].display_name, "Alice")
        self.assertEqual(self.manager.manual_mappings[session_id][1].display_name, "Bob")

    def test_manual_correlation_with_mappings(self):
        """Test correlation using manual mappings."""
        session_id = "test_manual_correlation"

        # Sample Whisper speakers
        whisper_speakers = [
            {"speaker_id": "speaker_0", "confidence": 0.9},
            {"speaker_id": "speaker_1", "confidence": 0.8},
        ]

        # Manual mappings
        mappings = [
            ManualSpeakerMapping("speaker_0", "Test User 1", "John Doe"),
            ManualSpeakerMapping("speaker_1", "Test User 2", "Jane Smith"),
        ]

        async def run_test():
            # Set manual mappings
            await self.manager.set_manual_speaker_mapping(session_id, mappings)

            # Perform correlation
            result = await self.manager.correlate_speakers(
                session_id=session_id,
                whisper_speakers=whisper_speakers,
                start_timestamp=0.0,
                end_timestamp=10.0,
            )

            return result

        result = asyncio.run(run_test())

        self.assertTrue(result.success)
        self.assertEqual(result.method_used, CorrelationMethod.MANUAL)
        self.assertEqual(len(result.correlations), 2)
        self.assertEqual(result.confidence_score, 1.0)  # Manual mappings have full confidence

        # Check individual correlations
        correlations = result.correlations
        self.assertEqual(correlations[0].whisper_speaker_id, "speaker_0")
        self.assertEqual(correlations[0].external_speaker_name, "Test User 1")
        self.assertEqual(correlations[0].correlation_confidence, 1.0)

        self.assertEqual(correlations[1].whisper_speaker_id, "speaker_1")
        self.assertEqual(correlations[1].external_speaker_name, "Test User 2")
        self.assertEqual(correlations[1].correlation_confidence, 1.0)

    def test_manual_correlation_partial_mappings(self):
        """Test correlation with partial manual mappings."""
        session_id = "test_partial_manual"

        whisper_speakers = [
            {"speaker_id": "speaker_0", "confidence": 0.9},
            {"speaker_id": "speaker_1", "confidence": 0.8},
            {
                "speaker_id": "speaker_2",
                "confidence": 0.7,
            },  # No manual mapping for this one
        ]

        # Only map two speakers manually
        mappings = [
            ManualSpeakerMapping("speaker_0", "Known User 1"),
            ManualSpeakerMapping("speaker_1", "Known User 2"),
        ]

        async def run_test():
            await self.manager.set_manual_speaker_mapping(session_id, mappings)

            result = await self.manager.correlate_speakers(
                session_id=session_id,
                whisper_speakers=whisper_speakers,
                start_timestamp=0.0,
                end_timestamp=10.0,
            )

            return result

        result = asyncio.run(run_test())

        self.assertTrue(result.success)
        self.assertEqual(result.method_used, CorrelationMethod.MANUAL)
        self.assertEqual(len(result.correlations), 2)  # Only mapped speakers

        # Verify only mapped speakers are correlated
        speaker_ids = [c.whisper_speaker_id for c in result.correlations]
        self.assertIn("speaker_0", speaker_ids)
        self.assertIn("speaker_1", speaker_ids)
        self.assertNotIn("speaker_2", speaker_ids)


class TestLoopbackAudioCorrelation(unittest.TestCase):
    """Test loopback audio correlation functionality."""

    def setUp(self):
        """Set up test manager."""
        self.manager = SpeakerCorrelationManager(
            enable_manual_correlation=True, enable_fallback_correlation=True
        )

    def test_set_loopback_config(self):
        """Test setting loopback configuration."""
        session_id = "test_loopback_config"
        config = LoopbackSpeakerInfo(
            estimated_speaker_count=2,
            primary_speaker_name="Loopback Primary",
            secondary_speakers=["Loopback Secondary"],
            audio_source_description="Test System Audio",
        )

        async def run_test():
            result = await self.manager.set_loopback_config(session_id, config)
            return result

        result = asyncio.run(run_test())

        self.assertTrue(result)
        self.assertIn(session_id, self.manager.loopback_configs)
        stored_config = self.manager.loopback_configs[session_id]
        self.assertEqual(stored_config.primary_speaker_name, "Loopback Primary")
        self.assertEqual(stored_config.secondary_speakers, ["Loopback Secondary"])

    def test_fallback_correlation_with_loopback_config(self):
        """Test fallback correlation using loopback configuration."""
        session_id = "test_fallback_loopback"

        whisper_speakers = [
            {"speaker_id": "speaker_0"},
            {"speaker_id": "speaker_1"},
            {"speaker_id": "speaker_2"},
        ]

        loopback_config = LoopbackSpeakerInfo(
            estimated_speaker_count=3,
            primary_speaker_name="Main Loopback Speaker",
            secondary_speakers=["Secondary Speaker A", "Secondary Speaker B"],
            audio_source_description="System Audio Loopback",
        )

        async def run_test():
            await self.manager.set_loopback_config(session_id, loopback_config)

            # Force fallback method
            result = await self.manager.correlate_speakers(
                session_id=session_id,
                whisper_speakers=whisper_speakers,
                start_timestamp=0.0,
                end_timestamp=10.0,
                force_method=CorrelationMethod.FALLBACK,
            )

            return result

        result = asyncio.run(run_test())

        self.assertTrue(result.success)
        self.assertEqual(result.method_used, CorrelationMethod.FALLBACK)
        self.assertTrue(result.fallback_applied)
        self.assertEqual(len(result.correlations), 3)

        # Check speaker names from loopback config
        correlations = result.correlations
        self.assertEqual(correlations[0].external_speaker_name, "Main Loopback Speaker")
        self.assertEqual(correlations[1].external_speaker_name, "Secondary Speaker A")
        self.assertEqual(correlations[2].external_speaker_name, "Secondary Speaker B")

    def test_fallback_correlation_without_loopback_config(self):
        """Test fallback correlation without loopback configuration."""
        session_id = "test_fallback_generic"

        whisper_speakers = [{"speaker_id": "speaker_0"}, {"speaker_id": "speaker_1"}]

        async def run_test():
            result = await self.manager.correlate_speakers(
                session_id=session_id,
                whisper_speakers=whisper_speakers,
                start_timestamp=0.0,
                end_timestamp=10.0,
                force_method=CorrelationMethod.FALLBACK,
            )

            return result

        result = asyncio.run(run_test())

        self.assertTrue(result.success)
        self.assertEqual(result.method_used, CorrelationMethod.FALLBACK)
        self.assertEqual(len(result.correlations), 2)

        # Check generic speaker names
        correlations = result.correlations
        self.assertEqual(correlations[0].external_speaker_name, "Speaker 1")
        self.assertEqual(correlations[1].external_speaker_name, "Speaker 2")


class TestCorrelationMethodSelection(unittest.TestCase):
    """Test correlation method selection logic."""

    def setUp(self):
        """Set up test manager."""
        self.manager = SpeakerCorrelationManager(
            enable_manual_correlation=True,
            enable_google_meet_correlation=True,
            enable_fallback_correlation=True,
        )

    def test_manual_method_priority(self):
        """Test that manual method has highest priority."""
        session_id = "test_method_priority"

        whisper_speakers = [{"speaker_id": "speaker_0"}]
        google_meet_speakers = [{"id": "gmeet_1", "name": "Google Meet User"}]

        # Set manual mapping
        mappings = [ManualSpeakerMapping("speaker_0", "Manual User")]

        async def run_test():
            await self.manager.set_manual_speaker_mapping(session_id, mappings)

            # Even with Google Meet data available, should use manual
            result = await self.manager.correlate_speakers(
                session_id=session_id,
                whisper_speakers=whisper_speakers,
                google_meet_speakers=google_meet_speakers,
                start_timestamp=0.0,
                end_timestamp=10.0,
            )

            return result

        result = asyncio.run(run_test())

        self.assertEqual(result.method_used, CorrelationMethod.MANUAL)
        self.assertEqual(result.correlations[0].external_speaker_name, "Manual User")

    def test_google_meet_method_selection(self):
        """Test Google Meet method when manual not available."""
        session_id = "test_google_meet_method"

        whisper_speakers = [{"speaker_id": "speaker_0"}]
        google_meet_speakers = [{"id": "gmeet_1", "name": "Google Meet User"}]

        async def run_test():
            # No manual mappings set
            result = await self.manager.correlate_speakers(
                session_id=session_id,
                whisper_speakers=whisper_speakers,
                google_meet_speakers=google_meet_speakers,
                start_timestamp=0.0,
                end_timestamp=10.0,
            )

            return result

        result = asyncio.run(run_test())

        self.assertEqual(result.method_used, CorrelationMethod.GOOGLE_MEET_API)
        self.assertEqual(result.correlations[0].external_speaker_name, "Google Meet User")

    def test_fallback_method_selection(self):
        """Test fallback method when others not available."""
        session_id = "test_fallback_method"

        whisper_speakers = [{"speaker_id": "speaker_0"}]

        async def run_test():
            # No manual mappings, no Google Meet data
            result = await self.manager.correlate_speakers(
                session_id=session_id,
                whisper_speakers=whisper_speakers,
                google_meet_speakers=None,
                start_timestamp=0.0,
                end_timestamp=10.0,
            )

            return result

        result = asyncio.run(run_test())

        self.assertEqual(result.method_used, CorrelationMethod.FALLBACK)
        self.assertTrue(result.fallback_applied)


class TestCorrelationWithDatabase(unittest.TestCase):
    """Test correlation with database integration."""

    def setUp(self):
        """Set up test manager with mock database."""
        self.mock_db = AsyncMock()
        self.manager = SpeakerCorrelationManager(
            database_adapter=self.mock_db, enable_manual_correlation=True
        )

    def test_correlation_storage_success(self):
        """Test successful correlation storage in database."""
        session_id = "test_db_storage"

        whisper_speakers = [{"speaker_id": "speaker_0"}]
        mappings = [ManualSpeakerMapping("speaker_0", "DB Test User")]

        async def run_test():
            await self.manager.set_manual_speaker_mapping(session_id, mappings)

            result = await self.manager.correlate_speakers(
                session_id=session_id,
                whisper_speakers=whisper_speakers,
                start_timestamp=0.0,
                end_timestamp=10.0,
            )

            return result

        result = asyncio.run(run_test())

        self.assertTrue(result.success)

        # Verify database storage was called
        self.mock_db.store_speaker_correlation.assert_called()
        self.assertEqual(self.mock_db.store_speaker_correlation.call_count, 1)

    def test_correlation_without_database(self):
        """Test correlation works without database."""
        manager = SpeakerCorrelationManager(
            database_adapter=None,  # No database
            enable_manual_correlation=True,
        )

        session_id = "test_no_db"
        whisper_speakers = [{"speaker_id": "speaker_0"}]
        mappings = [ManualSpeakerMapping("speaker_0", "No DB User")]

        async def run_test():
            await manager.set_manual_speaker_mapping(session_id, mappings)

            result = await manager.correlate_speakers(
                session_id=session_id,
                whisper_speakers=whisper_speakers,
                start_timestamp=0.0,
                end_timestamp=10.0,
            )

            return result

        result = asyncio.run(run_test())

        # Should still work without database
        self.assertTrue(result.success)
        self.assertEqual(len(result.correlations), 1)


class TestCorrelationStatistics(unittest.TestCase):
    """Test correlation statistics and monitoring."""

    def setUp(self):
        """Set up test manager."""
        self.manager = SpeakerCorrelationManager()

    def test_initial_statistics(self):
        """Test initial statistics state."""
        stats = self.manager.get_correlation_statistics()

        self.assertEqual(stats["total_attempts"], 0)
        self.assertEqual(stats["successful_correlations"], 0)
        self.assertEqual(stats["failed_correlations"], 0)
        self.assertEqual(stats["success_rate"], 0.0)

    def test_statistics_after_successful_correlation(self):
        """Test statistics after successful correlations."""
        session_id = "test_stats_success"

        whisper_speakers = [{"speaker_id": "speaker_0"}]
        mappings = [ManualSpeakerMapping("speaker_0", "Stats Test User")]

        async def run_test():
            # Set manual mappings for multiple sessions
            for i in range(3):
                await self.manager.set_manual_speaker_mapping(f"{session_id}_{i}", mappings)

            # Perform multiple correlations
            for i in range(3):
                await self.manager.correlate_speakers(
                    session_id=f"{session_id}_{i}",
                    whisper_speakers=whisper_speakers,
                    start_timestamp=0.0,
                    end_timestamp=10.0,
                )

        asyncio.run(run_test())

        stats = self.manager.get_correlation_statistics()

        self.assertEqual(stats["total_attempts"], 3)
        self.assertEqual(stats["successful_correlations"], 3)
        self.assertEqual(stats["manual_correlations"], 3)
        self.assertEqual(stats["success_rate"], 1.0)

    def test_statistics_with_failures(self):
        """Test statistics with correlation failures."""
        session_id = "test_stats_failure"

        # Create manager that will fail (no fallback enabled)
        failing_manager = SpeakerCorrelationManager(
            enable_manual_correlation=False,
            enable_google_meet_correlation=False,
            enable_fallback_correlation=False,
        )

        whisper_speakers = [{"speaker_id": "speaker_0"}]

        async def run_test():
            result = await failing_manager.correlate_speakers(
                session_id=session_id,
                whisper_speakers=whisper_speakers,
                start_timestamp=0.0,
                end_timestamp=10.0,
            )
            return result

        result = asyncio.run(run_test())

        self.assertFalse(result.success)

        stats = failing_manager.get_correlation_statistics()
        self.assertEqual(stats["total_attempts"], 1)
        self.assertEqual(stats["failed_correlations"], 1)
        self.assertEqual(stats["success_rate"], 0.0)


class TestSessionManagement(unittest.TestCase):
    """Test session-specific correlation management."""

    def setUp(self):
        """Set up test manager."""
        self.manager = SpeakerCorrelationManager()

    def test_get_session_correlations(self):
        """Test getting correlations for a session."""
        session_id = "test_get_correlations"

        whisper_speakers = [{"speaker_id": "speaker_0"}]
        mappings = [ManualSpeakerMapping("speaker_0", "Session Test User")]

        async def run_test():
            await self.manager.set_manual_speaker_mapping(session_id, mappings)

            # Perform correlation
            await self.manager.correlate_speakers(
                session_id=session_id,
                whisper_speakers=whisper_speakers,
                start_timestamp=0.0,
                end_timestamp=10.0,
            )

            # Get cached correlations
            correlations = await self.manager.get_session_correlations(session_id)
            return correlations

        correlations = asyncio.run(run_test())

        self.assertEqual(len(correlations), 1)
        self.assertEqual(correlations[0].external_speaker_name, "Session Test User")

    def test_clear_session_correlations(self):
        """Test clearing correlations for a session."""
        session_id = "test_clear_correlations"

        whisper_speakers = [{"speaker_id": "speaker_0"}]
        mappings = [ManualSpeakerMapping("speaker_0", "Clear Test User")]

        async def run_test():
            await self.manager.set_manual_speaker_mapping(session_id, mappings)
            await self.manager.correlate_speakers(
                session_id=session_id,
                whisper_speakers=whisper_speakers,
                start_timestamp=0.0,
                end_timestamp=10.0,
            )

            # Verify correlations exist
            correlations_before = await self.manager.get_session_correlations(session_id)

            # Clear correlations
            await self.manager.clear_session_correlations(session_id)

            # Verify correlations are cleared
            correlations_after = await self.manager.get_session_correlations(session_id)

            return correlations_before, correlations_after

        correlations_before, correlations_after = asyncio.run(run_test())

        self.assertEqual(len(correlations_before), 1)
        self.assertEqual(len(correlations_after), 0)

        # Verify all session data is cleared
        self.assertNotIn(session_id, self.manager.correlation_cache)
        self.assertNotIn(session_id, self.manager.manual_mappings)
        self.assertNotIn(session_id, self.manager.loopback_configs)


class TestCorrelationIntegration(unittest.TestCase):
    """Test end-to-end correlation integration."""

    def test_complete_manual_workflow(self):
        """Test complete manual testing workflow."""
        manager = create_speaker_correlation_manager(
            enable_google_meet_correlation=False,
            enable_manual_correlation=True,
            enable_fallback_correlation=True,
        )

        session_id = "complete_manual_test"

        async def run_test():
            # 1. Set up manual speaker mappings
            mappings = [
                ManualSpeakerMapping("speaker_0", "Alice", "Alice Johnson", "Primary test speaker"),
                ManualSpeakerMapping("speaker_1", "Bob", "Bob Smith", "Secondary test speaker"),
            ]

            await manager.set_manual_speaker_mapping(session_id, mappings)

            # 2. Set up loopback configuration
            loopback_config = LoopbackSpeakerInfo(
                estimated_speaker_count=2,
                primary_speaker_name="Alice",
                secondary_speakers=["Bob"],
                audio_source_description="Manual Test Loopback Audio",
                mixing_detected=False,
            )

            await manager.set_loopback_config(session_id, loopback_config)

            # 3. Simulate whisper speakers detected
            whisper_speakers = [
                {"speaker_id": "speaker_0", "confidence": 0.95, "speaking_time": 45.2},
                {"speaker_id": "speaker_1", "confidence": 0.88, "speaking_time": 32.1},
            ]

            # 4. Perform correlation
            result = await manager.correlate_speakers(
                session_id=session_id,
                whisper_speakers=whisper_speakers,
                start_timestamp=0.0,
                end_timestamp=120.0,
            )

            # 5. Get final correlations
            correlations = await manager.get_session_correlations(session_id)

            # 6. Get statistics
            stats = manager.get_correlation_statistics()

            return result, correlations, stats

        result, correlations, stats = asyncio.run(run_test())

        # Verify complete workflow success
        self.assertTrue(result.success)
        self.assertEqual(result.method_used, CorrelationMethod.MANUAL)
        self.assertEqual(len(result.correlations), 2)
        self.assertEqual(result.confidence_score, 1.0)

        # Verify correlations are accessible
        self.assertEqual(len(correlations), 2)

        # Verify speaker names
        speaker_names = [c.external_speaker_name for c in correlations]
        self.assertIn("Alice", speaker_names)
        self.assertIn("Bob", speaker_names)

        # Verify statistics
        self.assertEqual(stats["total_attempts"], 1)
        self.assertEqual(stats["successful_correlations"], 1)
        self.assertEqual(stats["manual_correlations"], 1)
        self.assertEqual(stats["success_rate"], 1.0)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
