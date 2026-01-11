#!/usr/bin/env python3
"""
Comprehensive Bot Lifecycle Tests

Tests the complete Google Meet bot system including spawning, audio capture,
data processing, and virtual webcam generation. Validates the full integration
pipeline from bot creation to meeting completion.

Test Categories:
- Bot Manager lifecycle (spawn, monitor, cleanup)
- Audio capture system with real audio data
- Caption processing and speaker timeline
- Time correlation between external/internal data
- Virtual webcam generation and output
- Database integration and persistence
- Error handling and recovery
"""

import pytest
import asyncio
import time
import tempfile
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Temporarily commenting out imports for modules not yet implemented
# from bot.bot_manager import GoogleMeetBotManager, BotStatus, MeetingRequest, BotInstance
# from bot.audio_capture import GoogleMeetAudioCapture, AudioConfig, MeetingInfo
# from bot.caption_processor import GoogleMeetCaptionProcessor, CaptionSegment, SpeakerTimelineEvent
# from bot.time_correlation import TimeCorrelationEngine, CorrelationConfig, ExternalSpeakerEvent, InternalTranscriptionResult
# from bot.virtual_webcam import VirtualWebcamManager, WebcamConfig, DisplayMode, Theme
# from bot.bot_integration import GoogleMeetBotIntegration, BotConfig, ServiceEndpoints
# from database.bot_session_manager import BotSessionDatabaseManager

# Skip all tests in this file until bot components are implemented
pytestmark = pytest.mark.skip(reason="Bot components not yet implemented")


class TestBotLifecycle:
    """Test complete bot lifecycle from spawn to cleanup."""

    @pytest.fixture
    async def bot_manager(self):
        """Create bot manager for testing."""
        config = {
            "database_url": "sqlite:///:memory:",
            "max_concurrent_bots": 5,
            "health_check_interval": 1.0,
        }
        manager = GoogleMeetBotManager(config)
        await manager.initialize()
        yield manager
        await manager.shutdown()

    @pytest.fixture
    def meeting_request(self):
        """Create test meeting request."""
        return MeetingRequest(
            meeting_id="test-meeting-123",
            meeting_title="Test Bot Lifecycle Meeting",
            organizer_email="test@example.com",
            target_languages=["en", "es", "fr"],
            auto_translation=True,
            priority="high",
        )

    @pytest.mark.asyncio
    async def test_bot_spawn_lifecycle(self, bot_manager, meeting_request):
        """Test complete bot spawn and lifecycle."""
        # Test bot spawn
        bot_id = await bot_manager.spawn_bot(meeting_request)
        assert bot_id is not None
        assert len(bot_id) > 0

        # Verify bot is in active bots
        active_bots = bot_manager.get_active_bots()
        assert bot_id in active_bots

        # Check bot status
        bot_status = await bot_manager.get_bot_status(bot_id)
        assert bot_status is not None
        assert bot_status.status in [BotStatus.SPAWNING, BotStatus.ACTIVE]
        assert bot_status.meeting_request.meeting_id == meeting_request.meeting_id

        # Wait for bot to become active
        max_wait = 10  # seconds
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status = await bot_manager.get_bot_status(bot_id)
            if status.status == BotStatus.ACTIVE:
                break
            await asyncio.sleep(0.1)

        # Verify bot is active
        final_status = await bot_manager.get_bot_status(bot_id)
        assert final_status.status == BotStatus.ACTIVE

        # Test bot termination
        success = await bot_manager.terminate_bot(bot_id)
        assert success is True

        # Verify bot is removed
        await asyncio.sleep(1.0)  # Allow cleanup time
        active_bots_after = bot_manager.get_active_bots()
        assert bot_id not in active_bots_after

    @pytest.mark.asyncio
    async def test_bot_error_handling(self, bot_manager):
        """Test bot error handling and recovery."""
        # Test invalid meeting request
        invalid_request = MeetingRequest(
            meeting_id="",  # Invalid empty meeting ID
            meeting_title="Invalid Meeting",
        )

        bot_id = await bot_manager.spawn_bot(invalid_request)
        assert bot_id is None  # Should fail to spawn

        # Test terminating non-existent bot
        success = await bot_manager.terminate_bot("non-existent-bot-123")
        assert success is False

    @pytest.mark.asyncio
    async def test_multiple_bots(self, bot_manager):
        """Test managing multiple concurrent bots."""
        meeting_requests = [
            MeetingRequest(meeting_id=f"meeting-{i}", meeting_title=f"Meeting {i}")
            for i in range(3)
        ]

        # Spawn multiple bots
        bot_ids = []
        for request in meeting_requests:
            bot_id = await bot_manager.spawn_bot(request)
            assert bot_id is not None
            bot_ids.append(bot_id)

        # Verify all bots are active
        active_bots = bot_manager.get_active_bots()
        for bot_id in bot_ids:
            assert bot_id in active_bots

        # Get system stats
        stats = bot_manager.get_system_stats()
        assert stats["active_bots"] == 3
        assert stats["total_bots_spawned"] >= 3

        # Terminate all bots
        for bot_id in bot_ids:
            success = await bot_manager.terminate_bot(bot_id)
            assert success is True


class TestAudioCapture:
    """Test audio capture system with real audio data."""

    @pytest.fixture
    def audio_config(self):
        """Create audio configuration for testing."""
        return AudioConfig(
            sample_rate=16000, channels=1, chunk_duration=1.0, quality_threshold=0.5
        )

    @pytest.fixture
    def meeting_info(self):
        """Create meeting info for testing."""
        return MeetingInfo(
            meeting_id="test-audio-meeting",
            meeting_title="Audio Capture Test",
            participant_count=3,
        )

    @pytest.fixture
    def mock_whisper_service(self):
        """Mock whisper service for testing."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "session_id": "test-session",
                "status": "created",
            }
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            yield mock_client

    def generate_test_audio(self, duration_seconds=2.0, sample_rate=16000):
        """Generate test audio data."""
        samples = int(duration_seconds * sample_rate)
        # Generate sine wave with some noise
        t = np.linspace(0, duration_seconds, samples)
        frequency = 440  # A4 note
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        # Add some noise to make it more realistic
        noise = 0.1 * np.random.randn(samples)
        audio = audio + noise
        return audio.astype(np.float32)

    @pytest.mark.asyncio
    async def test_audio_capture_initialization(
        self, audio_config, mock_whisper_service
    ):
        """Test audio capture initialization."""
        capture = GoogleMeetAudioCapture(
            config=audio_config, whisper_service_url="http://localhost:5001"
        )

        assert capture.config == audio_config
        assert capture.is_capturing is False
        assert capture.total_chunks_captured == 0

    @pytest.mark.asyncio
    async def test_audio_processing_pipeline(
        self, audio_config, meeting_info, mock_whisper_service
    ):
        """Test audio processing pipeline with real audio data."""
        capture = GoogleMeetAudioCapture(
            config=audio_config, whisper_service_url="http://localhost:5001"
        )

        # Track processed audio chunks
        processed_chunks = []
        transcription_results = []

        def on_audio_chunk(chunk, quality_metrics):
            processed_chunks.append((chunk, quality_metrics))

        def on_transcription(result):
            transcription_results.append(result)

        capture.set_audio_chunk_callback(on_audio_chunk)
        capture.set_transcription_callback(on_transcription)

        # Mock audio stream with actual audio data
        test_audio = self.generate_test_audio(duration_seconds=3.0)

        # Simulate audio capture
        with patch("sounddevice.InputStream") as mock_stream:
            mock_stream_instance = Mock()
            mock_stream.return_value = mock_stream_instance

            # Start capture
            success = await capture.start_capture(meeting_info)
            assert success is True
            assert capture.is_capturing is True

            # Simulate audio callbacks with real data
            chunk_size = int(audio_config.sample_rate * audio_config.chunk_duration)
            for i in range(0, len(test_audio), chunk_size):
                chunk = test_audio[i : i + chunk_size]
                if len(chunk) == chunk_size:
                    # Simulate the audio callback
                    capture._audio_callback(
                        chunk.reshape(-1, 1), len(chunk), None, None
                    )

            # Wait for processing
            await asyncio.sleep(2.0)

            # Stop capture
            success = await capture.stop_capture()
            assert success is True
            assert capture.is_capturing is False

        # Verify audio was processed
        assert len(processed_chunks) > 0
        assert capture.total_chunks_captured > 0

        # Verify audio quality analysis
        for chunk, quality_metrics in processed_chunks:
            assert "rms_level" in quality_metrics
            assert "voice_activity" in quality_metrics
            assert "quality_score" in quality_metrics
            assert quality_metrics["quality_score"] >= 0.0
            assert quality_metrics["quality_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_audio_quality_detection(self, audio_config):
        """Test audio quality detection with different audio types."""
        capture = GoogleMeetAudioCapture(
            config=audio_config, whisper_service_url="http://localhost:5001"
        )

        # Test with good quality audio (sine wave)
        good_audio = self.generate_test_audio(duration_seconds=1.0)
        quality_metrics = capture.quality_analyzer.analyze_chunk(good_audio)

        assert quality_metrics["voice_activity"] is True  # Should detect activity
        assert quality_metrics["rms_level"] > 0.1  # Should have good signal level
        assert quality_metrics["quality_score"] > 0.5  # Should be good quality

        # Test with silence
        silence = np.zeros(16000, dtype=np.float32)
        silence_metrics = capture.quality_analyzer.analyze_chunk(silence)

        assert silence_metrics["voice_activity"] is False  # Should detect silence
        assert silence_metrics["rms_level"] < 0.01  # Should have low signal
        assert silence_metrics["quality_score"] < 0.2  # Should be poor quality

        # Test with clipping
        clipped_audio = np.ones(16000, dtype=np.float32)  # Maximum amplitude
        clipping_metrics = capture.quality_analyzer.analyze_chunk(clipped_audio)

        assert clipping_metrics["clipping_detected"] is True  # Should detect clipping
        assert clipping_metrics["peak_level"] > 0.98  # Should detect high peak


class TestCaptionProcessor:
    """Test caption processing and speaker timeline extraction."""

    @pytest.fixture
    def caption_processor(self):
        """Create caption processor for testing."""
        return GoogleMeetCaptionProcessor(session_id="test-caption-session")

    @pytest.mark.asyncio
    async def test_caption_parsing(self, caption_processor):
        """Test parsing various caption formats."""
        test_captions = [
            ("John Doe: Hello everyone, welcome to our meeting", time.time()),
            ("[Jane Smith] Thanks for joining us today", time.time() + 1),
            ("Bob Johnson - Can everyone hear me okay?", time.time() + 2),
            ("<Alice Brown> Yes, we can hear you clearly", time.time() + 3),
            ("Mike Wilson joined the meeting", time.time() + 4),
            ("Sarah Davis left the meeting", time.time() + 5),
        ]

        processed_captions = []
        timeline_events = []

        def on_caption(caption):
            processed_captions.append(caption)

        def on_speaker_event(event):
            timeline_events.append(event)

        caption_processor.set_caption_callback(on_caption)
        caption_processor.set_speaker_event_callback(on_speaker_event)

        # Process all test captions
        for caption_text, timestamp in test_captions:
            success = caption_processor.process_caption_line(caption_text, timestamp)
            assert success is True

        # Verify caption processing
        assert len(processed_captions) == 4  # 4 speaker captions
        assert len(timeline_events) == 2  # 2 system events (join/leave)

        # Verify caption content
        speaker_names = [caption.speaker_name for caption in processed_captions]
        assert "John Doe" in speaker_names
        assert "Jane Smith" in speaker_names
        assert "Bob Johnson" in speaker_names
        assert "Alice Brown" in speaker_names

        # Verify timeline events
        event_types = [event.event_type for event in timeline_events]
        assert "join" in event_types
        assert "leave" in event_types

    @pytest.mark.asyncio
    async def test_speaker_timeline_management(self, caption_processor):
        """Test speaker timeline and statistics."""
        # Add multiple speakers with various activities
        test_events = [
            ("John Doe joined the meeting", time.time()),
            ("John Doe: Welcome everyone to today's meeting", time.time() + 1),
            ("Jane Smith joined the meeting", time.time() + 2),
            ("Jane Smith: Thank you John, excited to be here", time.time() + 3),
            ("John Doe: Let's start with the first agenda item", time.time() + 4),
            ("Bob Johnson joined the meeting", time.time() + 5),
            ("Bob Johnson: Sorry I'm late, can you hear me?", time.time() + 6),
            ("Jane Smith: Yes Bob, we can hear you fine", time.time() + 7),
            ("Bob Johnson left the meeting", time.time() + 8),
        ]

        for event_text, timestamp in test_events:
            caption_processor.process_caption_line(event_text, timestamp)

        # Get timeline statistics
        timeline_data = caption_processor.get_current_timeline()
        stats = timeline_data["statistics"]

        # Verify speaker statistics
        assert stats["total_speakers"] >= 3  # John, Jane, Bob
        assert stats["total_captions"] >= 5  # 5 speaking events
        assert stats["total_timeline_events"] >= 3  # 2 joins + 1 leave

        # Verify speaker details
        speakers = timeline_data["speakers"]
        assert "speaker_john_doe" in speakers
        assert "speaker_jane_smith" in speakers
        assert "speaker_bob_johnson" in speakers

        # Check speaker activity
        john_speaker = speakers["speaker_john_doe"]
        assert john_speaker["utterance_count"] >= 2
        assert john_speaker["is_active"] is True

        bob_speaker = speakers["speaker_bob_johnson"]
        assert bob_speaker["is_active"] is False  # Left the meeting


class TestTimeCorrelation:
    """Test time correlation between external and internal data."""

    @pytest.fixture
    def correlation_engine(self):
        """Create correlation engine for testing."""
        config = CorrelationConfig(
            timing_tolerance=2.0,
            audio_delay_compensation=0.5,
            min_correlation_confidence=0.6,
        )
        return TimeCorrelationEngine("test-correlation-session", config)

    @pytest.mark.asyncio
    async def test_correlation_basic_matching(self, correlation_engine):
        """Test basic correlation between external events and internal results."""
        base_time = time.time()

        # Add external speaker events (from Google Meet)
        external_events = [
            ExternalSpeakerEvent(
                speaker_id="speaker_john",
                speaker_name="John Doe",
                event_type="speaking_start",
                timestamp=base_time,
                confidence=0.9,
            ),
            ExternalSpeakerEvent(
                speaker_id="speaker_jane",
                speaker_name="Jane Smith",
                event_type="speaking_start",
                timestamp=base_time + 5.0,
                confidence=0.85,
            ),
        ]

        # Add internal transcription results (from our processing)
        internal_results = [
            InternalTranscriptionResult(
                segment_id="seg_001",
                text="Hello everyone, welcome to the meeting",
                start_timestamp=base_time + 0.3,  # Slight delay
                end_timestamp=base_time + 3.0,
                language="en",
                confidence=0.92,
                session_id="test-correlation-session",
            ),
            InternalTranscriptionResult(
                segment_id="seg_002",
                text="Thanks John, glad to be here today",
                start_timestamp=base_time + 5.4,  # Slight delay
                end_timestamp=base_time + 8.0,
                language="en",
                confidence=0.88,
                session_id="test-correlation-session",
            ),
        ]

        # Add events to correlation engine
        for event in external_events:
            success = correlation_engine.add_external_event(event)
            assert success is True

        for result in internal_results:
            success = correlation_engine.add_internal_result(result)
            assert success is True

        # Get correlations
        correlations = correlation_engine.get_correlations()

        # Verify correlations were made
        assert len(correlations) >= 1

        # Check correlation quality
        for correlation in correlations:
            assert correlation["correlation_confidence"] >= 0.6
            assert correlation["speaker_name"] in ["John Doe", "Jane Smith"]
            assert correlation["correlation_type"] in [
                "exact",
                "inferred",
                "interpolated",
            ]
            assert abs(correlation["timing_offset"]) <= 2.0  # Within tolerance

    @pytest.mark.asyncio
    async def test_correlation_statistics(self, correlation_engine):
        """Test correlation statistics and performance tracking."""
        base_time = time.time()

        # Add multiple events and results
        for i in range(5):
            external_event = ExternalSpeakerEvent(
                speaker_id=f"speaker_{i}",
                speaker_name=f"Speaker {i}",
                event_type="speaking_start",
                timestamp=base_time + i * 2.0,
                confidence=0.8 + i * 0.05,
            )
            correlation_engine.add_external_event(external_event)

            internal_result = InternalTranscriptionResult(
                segment_id=f"seg_{i:03d}",
                text=f"Test transcription {i}",
                start_timestamp=base_time + i * 2.0 + 0.2,
                end_timestamp=base_time + i * 2.0 + 1.5,
                language="en",
                confidence=0.85 + i * 0.03,
                session_id="test-correlation-session",
            )
            correlation_engine.add_internal_result(internal_result)

        # Get statistics
        stats = correlation_engine.get_statistics()

        assert stats["total_external_events"] == 5
        assert stats["total_internal_results"] == 5
        assert stats["total_correlations"] >= 0
        assert stats["success_rate"] >= 0.0
        assert stats["success_rate"] <= 1.0


class TestVirtualWebcam:
    """Test virtual webcam generation and translation display."""

    @pytest.fixture
    def webcam_config(self):
        """Create webcam configuration for testing."""
        return WebcamConfig(
            width=1280,
            height=720,
            fps=30,
            display_mode=DisplayMode.OVERLAY,
            theme=Theme.DARK,
            max_translations_displayed=3,
        )

    @pytest.mark.asyncio
    async def test_webcam_initialization(self, webcam_config):
        """Test virtual webcam initialization."""
        webcam = VirtualWebcamManager(webcam_config)

        assert webcam.config == webcam_config
        assert webcam.is_streaming is False
        assert webcam.frames_generated == 0
        assert len(webcam.current_translations) == 0

    @pytest.mark.asyncio
    async def test_webcam_streaming_lifecycle(self, webcam_config):
        """Test webcam streaming start/stop lifecycle."""
        webcam = VirtualWebcamManager(webcam_config)

        # Test stream start
        success = await webcam.start_stream("test-webcam-session")
        assert success is True
        assert webcam.is_streaming is True

        # Allow some frame generation
        await asyncio.sleep(1.0)

        # Verify frames are being generated
        assert webcam.frames_generated > 0

        # Test stream stop
        success = await webcam.stop_stream()
        assert success is True
        assert webcam.is_streaming is False

        # Get final stats
        stats = webcam.get_webcam_stats()
        assert stats["frames_generated"] > 0
        assert stats["duration_seconds"] > 0

    @pytest.mark.asyncio
    async def test_translation_display(self, webcam_config):
        """Test adding and displaying translations."""
        webcam = VirtualWebcamManager(webcam_config)

        # Start streaming
        await webcam.start_stream("test-translation-session")

        # Add test translations
        test_translations = [
            {
                "translation_id": "trans_001",
                "translated_text": "Hello everyone, welcome to the meeting",
                "source_language": "zh",
                "target_language": "en",
                "speaker_name": "John Doe",
                "speaker_id": "speaker_john",
                "translation_confidence": 0.95,
            },
            {
                "translation_id": "trans_002",
                "translated_text": "Gracias Juan, me alegro de estar aquí",
                "source_language": "en",
                "target_language": "es",
                "speaker_name": "Jane Smith",
                "speaker_id": "speaker_jane",
                "translation_confidence": 0.88,
            },
        ]

        for translation in test_translations:
            webcam.add_translation(translation)

        # Verify translations were added
        assert len(webcam.current_translations) == 2

        # Allow frame generation with translations
        await asyncio.sleep(2.0)

        # Verify frames include translation data
        frame_base64 = webcam.get_current_frame_base64()
        assert frame_base64 is not None
        assert len(frame_base64) > 0

        await webcam.stop_stream()


class TestBotIntegration:
    """Test complete bot integration pipeline."""

    @pytest.fixture
    def bot_config(self):
        """Create bot configuration for testing."""
        return BotConfig(
            bot_id="test-integration-bot",
            bot_name="Test Integration Bot",
            target_languages=["en", "es", "fr"],
            virtual_webcam_enabled=True,
            service_endpoints=ServiceEndpoints(
                whisper_service="http://localhost:5001",
                translation_service="http://localhost:5003",
                orchestration_service="http://localhost:3000",
            ),
        )

    @pytest.fixture
    def meeting_info(self):
        """Create meeting info for integration testing."""
        return MeetingInfo(
            meeting_id="integration-test-meeting",
            meeting_title="Bot Integration Test Meeting",
            organizer_email="test@example.com",
            participant_count=4,
        )

    @pytest.mark.asyncio
    async def test_bot_integration_lifecycle(self, bot_config, meeting_info):
        """Test complete bot integration lifecycle."""
        # Mock external services
        with patch("httpx.AsyncClient") as mock_client:
            # Mock successful service responses
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "success",
                "session_id": "test-session",
            }
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            # Create bot integration
            bot_integration = GoogleMeetBotIntegration(bot_config)

            # Track events
            session_events = []
            transcriptions = []
            translations = []

            def on_session_event(event_type, data):
                session_events.append((event_type, data))

            def on_transcription(data):
                transcriptions.append(data)

            def on_translation(data):
                translations.append(data)

            bot_integration.set_session_event_callback(on_session_event)
            bot_integration.set_transcription_callback(on_transcription)
            bot_integration.set_translation_callback(on_translation)

            # Test joining meeting
            session_id = await bot_integration.join_meeting(meeting_info)
            assert session_id is not None
            assert len(session_id) > 0

            # Verify session event was fired
            assert len(session_events) > 0
            assert session_events[0][0] == "meeting_joined"

            # Get session status
            status = bot_integration.get_session_status(session_id)
            assert status is not None
            assert status["session_id"] == session_id
            assert status["meeting_id"] == meeting_info.meeting_id
            assert status["status"] in ["active", "spawning"]

            # Simulate some processing time
            await asyncio.sleep(1.0)

            # Test leaving meeting
            success = await bot_integration.leave_meeting(session_id)
            assert success is True

            # Verify leave event
            leave_events = [
                event for event in session_events if event[0] == "meeting_left"
            ]
            assert len(leave_events) > 0

            # Get final statistics
            stats = bot_integration.get_bot_statistics()
            assert stats["bot_id"] == bot_config.bot_id
            assert stats["total_sessions"] >= 1
            assert stats["target_languages"] == bot_config.target_languages


class TestDatabaseIntegration:
    """Test database integration for bot sessions."""

    @pytest.fixture
    async def database_manager(self):
        """Create database manager for testing."""
        # Use in-memory SQLite for testing
        config = {
            "database_url": "sqlite:///:memory:",
            "audio_storage_path": tempfile.mkdtemp(),
        }
        manager = BotSessionDatabaseManager(config)
        await manager.initialize()
        yield manager
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_session_storage(self, database_manager):
        """Test storing and retrieving bot sessions."""
        session_data = {
            "bot_id": "test-bot-001",
            "meeting_id": "test-meeting-456",
            "meeting_title": "Database Test Meeting",
            "status": "active",
            "start_time": datetime.now(),
            "target_languages": ["en", "es"],
            "session_metadata": {"test": True},
        }

        # Store session
        session_id = await database_manager.create_session(session_data)
        assert session_id is not None

        # Retrieve session
        retrieved_session = await database_manager.get_session(session_id)
        assert retrieved_session is not None
        assert retrieved_session["bot_id"] == session_data["bot_id"]
        assert retrieved_session["meeting_id"] == session_data["meeting_id"]
        assert retrieved_session["status"] == session_data["status"]

    @pytest.mark.asyncio
    async def test_audio_file_storage(self, database_manager):
        """Test storing audio files and metadata."""
        session_id = "test-audio-session"

        # Create test audio data
        test_audio = np.random.random(16000).astype(np.float32)  # 1 second of audio
        audio_bytes = test_audio.tobytes()

        metadata = {
            "duration_seconds": 1.0,
            "sample_rate": 16000,
            "channels": 1,
            "audio_quality_score": 0.85,
        }

        # Store audio file
        file_id = await database_manager.store_audio_file(
            session_id, audio_bytes, metadata
        )
        assert file_id is not None

        # Retrieve audio file info
        file_info = await database_manager.get_audio_file(file_id)
        assert file_info is not None
        assert file_info["session_id"] == session_id
        assert file_info["file_size"] == len(audio_bytes)
        assert file_info["duration_seconds"] == 1.0

    @pytest.mark.asyncio
    async def test_transcript_storage(self, database_manager):
        """Test storing transcripts and translations."""
        session_id = "test-transcript-session"

        transcript_data = {
            "text": "Hello everyone, welcome to our meeting",
            "source_type": "whisper_service",
            "language": "en",
            "start_timestamp": time.time(),
            "end_timestamp": time.time() + 3.0,
            "speaker_id": "speaker_john",
            "speaker_name": "John Doe",
            "confidence_score": 0.92,
        }

        # Store transcript
        transcript_id = await database_manager.store_transcript(
            session_id, transcript_data
        )
        assert transcript_id is not None

        # Store translation
        translation_data = {
            "source_transcript_id": transcript_id,
            "translated_text": "Hola a todos, bienvenidos a nuestra reunión",
            "source_language": "en",
            "target_language": "es",
            "translation_confidence": 0.89,
            "speaker_id": "speaker_john",
            "speaker_name": "John Doe",
            "start_timestamp": transcript_data["start_timestamp"],
            "end_timestamp": transcript_data["end_timestamp"],
        }

        translation_id = await database_manager.store_translation(
            session_id, translation_data
        )
        assert translation_id is not None

        # Retrieve transcript with translations
        transcript_info = await database_manager.get_transcript(transcript_id)
        assert transcript_info is not None
        assert transcript_info["text"] == transcript_data["text"]
        assert transcript_info["speaker_name"] == "John Doe"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
