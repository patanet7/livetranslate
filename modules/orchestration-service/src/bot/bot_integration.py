#!/usr/bin/env python3
"""
Google Meet Bot Services Integration Pipeline - Orchestration Service Integration

Orchestrates the complete flow from Google Meet audio/caption capture through
our transcription and translation services, with time correlation and final
output generation. Now fully integrated into the orchestration service.

Features:
- Complete bot-to-services integration
- Real-time audio and caption processing coordination
- Service communication and error handling
- Speaker-attributed transcription and translation
- Meeting lifecycle management
- Performance monitoring and analytics
- Virtual webcam output generation
- Database integration for persistent storage
"""

import os
import sys
import time
import logging
import asyncio
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import httpx
import uuid
from collections import deque

# Import bot components (now integrated in orchestration service)
from .audio_capture import GoogleMeetAudioCapture, AudioConfig, MeetingInfo
from .caption_processor import GoogleMeetCaptionProcessor
from .time_correlation import (
    TimeCorrelationEngine,
    CorrelationConfig,
    ExternalSpeakerEvent,
    InternalTranscriptionResult,
)
from .virtual_webcam import VirtualWebcamManager, WebcamConfig, DisplayMode, Theme

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ServiceEndpoints:
    """Configuration for service endpoints."""

    whisper_service: str = "http://localhost:5001"
    translation_service: str = "http://localhost:5003"
    orchestration_service: str = "http://localhost:3000"


@dataclass
class BotConfig:
    """Configuration for the Google Meet bot."""

    bot_id: str
    bot_name: str = "LiveTranslate Bot"
    target_languages: List[str] = None
    audio_config: AudioConfig = None
    correlation_config: CorrelationConfig = None
    webcam_config: WebcamConfig = None
    service_endpoints: ServiceEndpoints = None
    auto_join_meetings: bool = True
    recording_enabled: bool = False
    real_time_translation: bool = True
    virtual_webcam_enabled: bool = True

    def __post_init__(self):
        if self.target_languages is None:
            self.target_languages = ["en", "es", "fr", "de", "zh"]
        if self.audio_config is None:
            self.audio_config = AudioConfig()
        if self.correlation_config is None:
            self.correlation_config = CorrelationConfig()
        if self.webcam_config is None:
            self.webcam_config = WebcamConfig()
        if self.service_endpoints is None:
            self.service_endpoints = ServiceEndpoints()


@dataclass
class BotSession:
    """Information about an active bot session."""

    session_id: str
    meeting_info: MeetingInfo
    bot_config: BotConfig
    start_time: float
    end_time: Optional[float] = None
    status: str = "active"  # 'active', 'paused', 'ended', 'error'
    participants_count: int = 0
    messages_processed: int = 0
    errors_count: int = 0


class ServiceClient:
    """Client for communicating with our microservices."""

    def __init__(self, endpoints: ServiceEndpoints):
        self.endpoints = endpoints
        self.session_cache: Dict[str, Dict] = {}

    async def create_whisper_session(
        self, session_id: str, meeting_info: MeetingInfo
    ) -> bool:
        """Create a session with the whisper service."""
        try:
            session_data = {
                "session_id": session_id,
                "meeting_info": asdict(meeting_info),
                "audio_config": {
                    "sample_rate": 16000,
                    "channels": 1,
                    "format": "float32",
                },
                "bot_mode": True,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.endpoints.whisper_service}/api/sessions/create",
                    json=session_data,
                    timeout=10.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    self.session_cache[session_id] = {"whisper": result}
                    logger.info(f"Created whisper session: {session_id}")
                    return True
                else:
                    logger.error(
                        f"Failed to create whisper session: {response.status_code}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Error creating whisper session: {e}")
            return False

    async def create_translation_session(
        self, session_id: str, target_languages: List[str]
    ) -> bool:
        """Create a session with the translation service."""
        try:
            session_data = {
                "session_id": session_id,
                "target_languages": target_languages,
                "source_language": "auto",
                "context_mode": "continuous",
                "bot_mode": True,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.endpoints.translation_service}/api/sessions/create",
                    json=session_data,
                    timeout=10.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    if session_id not in self.session_cache:
                        self.session_cache[session_id] = {}
                    self.session_cache[session_id]["translation"] = result
                    logger.info(f"Created translation session: {session_id}")
                    return True
                else:
                    logger.error(
                        f"Failed to create translation session: {response.status_code}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Error creating translation session: {e}")
            return False

    async def request_translation(
        self, session_id: str, text: str, speaker_id: str, target_language: str
    ) -> Optional[Dict]:
        """Request translation for speaker-attributed text."""
        try:
            translation_data = {
                "text": text,
                "session_id": session_id,
                "speaker_id": speaker_id,
                "target_language": target_language,
                "source_language": "auto",
                "context_mode": "speaker_aware",
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.endpoints.translation_service}/api/translate",
                    json=translation_data,
                    timeout=10.0,
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(
                        f"Translation request failed: {response.status_code}"
                    )
                    return None

        except Exception as e:
            logger.error(f"Error requesting translation: {e}")
            return None

    async def close_all_sessions(self, session_id: str) -> bool:
        """Close all service sessions."""
        success = True

        try:
            # Close whisper session
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.endpoints.whisper_service}/api/sessions/{session_id}/close",
                    timeout=10.0,
                )
                if response.status_code != 200:
                    success = False
                    logger.warning(
                        f"Failed to close whisper session: {response.status_code}"
                    )
        except Exception as e:
            logger.error(f"Error closing whisper session: {e}")
            success = False

        try:
            # Close translation session
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.endpoints.translation_service}/api/sessions/{session_id}/close",
                    timeout=10.0,
                )
                if response.status_code != 200:
                    success = False
                    logger.warning(
                        f"Failed to close translation session: {response.status_code}"
                    )
        except Exception as e:
            logger.error(f"Error closing translation session: {e}")
            success = False

        return success


class GoogleMeetBotIntegration:
    """
    Main integration class that orchestrates all bot functionality.
    Integrated with orchestration service bot management.
    """

    def __init__(self, bot_config: BotConfig, bot_manager=None, database_manager=None):
        self.config = bot_config
        self.bot_manager = bot_manager
        self.database_manager = database_manager

        # Service client
        self.service_client = ServiceClient(bot_config.service_endpoints)

        # Components (will be initialized per session)
        self.audio_capture = None
        self.caption_processor = None
        self.correlation_engine = None
        self.virtual_webcam = None

        # Session management
        self.active_sessions: Dict[str, BotSession] = {}
        self.current_session_id = None

        # Performance tracking
        self.total_sessions = 0
        self.successful_sessions = 0
        self.total_messages_processed = 0

        # Event callbacks
        self.on_transcription_ready = None
        self.on_translation_ready = None
        self.on_speaker_event = None
        self.on_session_event = None
        self.on_error = None

        # Thread safety
        self.lock = threading.RLock()

        logger.info(f"GoogleMeetBotIntegration initialized")
        logger.info(f"  Bot ID: {bot_config.bot_id}")
        logger.info(f"  Target languages: {bot_config.target_languages}")
        logger.info(f"  Virtual webcam enabled: {bot_config.virtual_webcam_enabled}")
        logger.info(f"  Services: {asdict(bot_config.service_endpoints)}")

    def set_transcription_callback(self, callback: Callable[[Dict], None]):
        """Set callback for transcription results."""
        self.on_transcription_ready = callback

    def set_translation_callback(self, callback: Callable[[Dict], None]):
        """Set callback for translation results."""
        self.on_translation_ready = callback

    def set_speaker_event_callback(self, callback: Callable[[Dict], None]):
        """Set callback for speaker events."""
        self.on_speaker_event = callback

    def set_session_event_callback(self, callback: Callable[[str, Dict], None]):
        """Set callback for session events."""
        self.on_session_event = callback

    def set_error_callback(self, callback: Callable[[str], None]):
        """Set callback for error notifications."""
        self.on_error = callback

    async def join_meeting(self, meeting_info: MeetingInfo) -> str:
        """
        Join a Google Meet meeting and start processing.

        Args:
            meeting_info: Information about the meeting to join

        Returns:
            Session ID if successful, None if failed
        """
        session_id = (
            f"bot_{self.config.bot_id}_{meeting_info.meeting_id}_{int(time.time())}"
        )

        try:
            with self.lock:
                # Create bot session
                bot_session = BotSession(
                    session_id=session_id,
                    meeting_info=meeting_info,
                    bot_config=self.config,
                    start_time=time.time(),
                    participants_count=meeting_info.participant_count,
                )

                self.active_sessions[session_id] = bot_session
                self.current_session_id = session_id
                self.total_sessions += 1

            # Initialize service sessions
            success = await self._initialize_service_sessions(session_id, meeting_info)
            if not success:
                await self._cleanup_session(session_id)
                return None

            # Initialize bot components
            success = await self._initialize_bot_components(session_id, meeting_info)
            if not success:
                await self._cleanup_session(session_id)
                return None

            # Start processing
            success = await self._start_processing(session_id)
            if not success:
                await self._cleanup_session(session_id)
                return None

            logger.info(
                f"Successfully joined meeting: {meeting_info.meeting_id} (session: {session_id})"
            )

            if self.on_session_event:
                self.on_session_event(
                    "meeting_joined",
                    {
                        "session_id": session_id,
                        "meeting_info": asdict(meeting_info),
                        "bot_config": asdict(self.config),
                    },
                )

            return session_id

        except Exception as e:
            logger.error(f"Failed to join meeting: {e}")
            if self.on_error:
                self.on_error(f"Meeting join failed: {e}")
            await self._cleanup_session(session_id)
            return None

    async def leave_meeting(self, session_id: str = None) -> bool:
        """Leave a meeting and stop processing."""
        session_id = session_id or self.current_session_id
        if not session_id or session_id not in self.active_sessions:
            logger.warning(f"No active session found: {session_id}")
            return False

        try:
            # Update session status
            with self.lock:
                session = self.active_sessions[session_id]
                session.status = "ending"
                session.end_time = time.time()

            # Stop processing
            await self._stop_processing(session_id)

            # Cleanup session
            await self._cleanup_session(session_id)

            # Update statistics
            with self.lock:
                if session.errors_count == 0:
                    self.successful_sessions += 1

                if session_id == self.current_session_id:
                    self.current_session_id = None

            logger.info(f"Successfully left meeting (session: {session_id})")

            if self.on_session_event:
                self.on_session_event(
                    "meeting_left",
                    {
                        "session_id": session_id,
                        "duration": session.end_time - session.start_time
                        if session.end_time
                        else 0,
                        "messages_processed": session.messages_processed,
                        "errors_count": session.errors_count,
                    },
                )

            return True

        except Exception as e:
            logger.error(f"Error leaving meeting: {e}")
            if self.on_error:
                self.on_error(f"Meeting leave failed: {e}")
            return False

    async def _initialize_service_sessions(
        self, session_id: str, meeting_info: MeetingInfo
    ) -> bool:
        """Initialize sessions with all services."""
        try:
            # Create whisper session
            success = await self.service_client.create_whisper_session(
                session_id, meeting_info
            )
            if not success:
                return False

            # Create translation session
            success = await self.service_client.create_translation_session(
                session_id, self.config.target_languages
            )
            if not success:
                return False

            return True

        except Exception as e:
            logger.error(f"Error initializing service sessions: {e}")
            return False

    async def _initialize_bot_components(
        self, session_id: str, meeting_info: MeetingInfo
    ) -> bool:
        """Initialize bot components for the session."""
        try:
            # Initialize audio capture
            self.audio_capture = GoogleMeetAudioCapture(
                self.config.audio_config,
                self.config.service_endpoints.whisper_service,
                bot_manager=self.bot_manager,
                database_manager=self.database_manager,
            )

            # Set audio capture callbacks
            self.audio_capture.set_transcription_callback(
                self._handle_transcription_result
            )
            self.audio_capture.set_error_callback(self._handle_audio_error)

            # Initialize caption processor
            self.caption_processor = GoogleMeetCaptionProcessor(
                session_id,
                bot_manager=self.bot_manager,
                database_manager=self.database_manager,
            )

            # Set caption processor callbacks
            self.caption_processor.set_caption_callback(self._handle_caption_segment)
            self.caption_processor.set_speaker_event_callback(
                self._handle_speaker_event
            )
            self.caption_processor.set_error_callback(self._handle_caption_error)

            # Initialize correlation engine
            self.correlation_engine = TimeCorrelationEngine(
                session_id,
                self.config.correlation_config,
                bot_manager=self.bot_manager,
                database_manager=self.database_manager,
            )

            # Initialize virtual webcam if enabled
            if self.config.virtual_webcam_enabled:
                self.virtual_webcam = VirtualWebcamManager(
                    self.config.webcam_config, bot_manager=self.bot_manager
                )

                # Set webcam callbacks
                self.virtual_webcam.on_error = self._handle_webcam_error

            return True

        except Exception as e:
            logger.error(f"Error initializing bot components: {e}")
            return False

    async def _start_processing(self, session_id: str) -> bool:
        """Start all processing components."""
        try:
            session = self.active_sessions[session_id]

            # Start audio capture
            success = await self.audio_capture.start_capture(session.meeting_info)
            if not success:
                return False

            # Start caption processing
            success = await self.caption_processor.start_processing()
            if not success:
                return False

            # Start virtual webcam if enabled
            if self.virtual_webcam:
                success = await self.virtual_webcam.start_stream(session_id)
                if not success:
                    logger.warning(
                        "Failed to start virtual webcam, continuing without it"
                    )
                    # Don't fail the whole process if webcam fails

            # Update session status
            with self.lock:
                session.status = "active"

            return True

        except Exception as e:
            logger.error(f"Error starting processing: {e}")
            return False

    async def _stop_processing(self, session_id: str) -> bool:
        """Stop all processing components."""
        try:
            # Stop audio capture
            if self.audio_capture:
                await self.audio_capture.stop_capture()

            # Stop caption processing
            if self.caption_processor:
                await self.caption_processor.stop_processing()

            # Stop virtual webcam
            if self.virtual_webcam:
                await self.virtual_webcam.stop_stream()

            return True

        except Exception as e:
            logger.error(f"Error stopping processing: {e}")
            return False

    async def _cleanup_session(self, session_id: str):
        """Clean up session resources."""
        try:
            # Close service sessions
            await self.service_client.close_all_sessions(session_id)

            # Clean up components
            self.audio_capture = None
            self.caption_processor = None
            self.correlation_engine = None
            self.virtual_webcam = None

            # Remove from active sessions
            with self.lock:
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]

        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")

    def _handle_transcription_result(self, result: Dict):
        """Handle transcription result from audio capture."""
        try:
            if not result.get("clean_text"):
                return

            # Create internal transcription result
            internal_result = InternalTranscriptionResult(
                segment_id=result.get("segment_id", str(uuid.uuid4())),
                text=result["clean_text"],
                start_timestamp=result.get("window_start_time", time.time()),
                end_timestamp=result.get("window_end_time", time.time()),
                language=result.get("language", "unknown"),
                confidence=result.get("confidence", 0.0),
                session_id=self.current_session_id,
                processing_metadata=result,
            )

            # Add to correlation engine
            if self.correlation_engine:
                self.correlation_engine.add_internal_result(internal_result)

            # Try to get correlations and process them
            asyncio.create_task(self._process_correlations())

        except Exception as e:
            logger.error(f"Error handling transcription result: {e}")
            if self.on_error:
                self.on_error(f"Transcription handling error: {e}")

    def _handle_caption_segment(self, caption):
        """Handle caption segment from Google Meet."""
        try:
            # Create external speaker event
            external_event = ExternalSpeakerEvent(
                speaker_id=caption.speaker_id,
                speaker_name=caption.speaker_name,
                event_type="speaking_start",
                timestamp=caption.start_timestamp,
                confidence=caption.confidence,
                source="google_meet",
            )

            # Add to correlation engine
            if self.correlation_engine:
                self.correlation_engine.add_external_event(external_event)

            # Try to get correlations and process them
            asyncio.create_task(self._process_correlations())

        except Exception as e:
            logger.error(f"Error handling caption segment: {e}")
            if self.on_error:
                self.on_error(f"Caption handling error: {e}")

    def _handle_speaker_event(self, event):
        """Handle speaker timeline event."""
        try:
            if self.on_speaker_event:
                self.on_speaker_event(
                    {
                        "event_type": event.event_type,
                        "speaker_id": event.speaker_id,
                        "timestamp": event.timestamp,
                        "metadata": event.metadata,
                    }
                )

        except Exception as e:
            logger.error(f"Error handling speaker event: {e}")

    async def _process_correlations(self):
        """Process new correlations and generate translations."""
        try:
            if not self.correlation_engine:
                return

            # Get recent correlations
            recent_correlations = self.correlation_engine.get_correlations(
                start_time=time.time() - 10  # Last 10 seconds
            )

            for correlation in recent_correlations:
                # Request translations for each target language
                for target_lang in self.config.target_languages:
                    if target_lang == correlation["language"]:
                        continue  # Skip same language

                    translation_result = await self.service_client.request_translation(
                        self.current_session_id,
                        correlation["text"],
                        correlation["speaker_id"],
                        target_lang,
                    )

                    if translation_result:
                        # Create complete correlated result
                        correlated_data = {
                            "session_id": self.current_session_id,
                            "correlation_id": correlation["correlation_id"],
                            "speaker_id": correlation["speaker_id"],
                            "speaker_name": correlation["speaker_name"],
                            "original_text": correlation["text"],
                            "original_language": correlation["language"],
                            "translated_text": translation_result.get(
                                "translated_text"
                            ),
                            "target_language": target_lang,
                            "start_timestamp": correlation["start_timestamp"],
                            "end_timestamp": correlation["end_timestamp"],
                            "correlation_confidence": correlation[
                                "correlation_confidence"
                            ],
                            "translation_confidence": translation_result.get(
                                "confidence", 0.0
                            ),
                            "timestamp": time.time(),
                        }

                        # Add to virtual webcam if enabled
                        if self.virtual_webcam:
                            self.virtual_webcam.add_translation(correlated_data)

                        # Notify callbacks
                        if self.on_translation_ready:
                            self.on_translation_ready(correlated_data)

                        # Update session statistics
                        with self.lock:
                            if self.current_session_id in self.active_sessions:
                                self.active_sessions[
                                    self.current_session_id
                                ].messages_processed += 1
                                self.total_messages_processed += 1

        except Exception as e:
            logger.error(f"Error processing correlations: {e}")
            if self.on_error:
                self.on_error(f"Correlation processing error: {e}")

    def _handle_audio_error(self, error: str):
        """Handle audio capture error."""
        logger.error(f"Audio capture error: {error}")
        if self.current_session_id and self.current_session_id in self.active_sessions:
            with self.lock:
                self.active_sessions[self.current_session_id].errors_count += 1

        if self.on_error:
            self.on_error(f"Audio error: {error}")

    def _handle_caption_error(self, error: str):
        """Handle caption processing error."""
        logger.error(f"Caption processing error: {error}")
        if self.current_session_id and self.current_session_id in self.active_sessions:
            with self.lock:
                self.active_sessions[self.current_session_id].errors_count += 1

        if self.on_error:
            self.on_error(f"Caption error: {error}")

    def _handle_webcam_error(self, error: str):
        """Handle virtual webcam error."""
        logger.error(f"Virtual webcam error: {error}")
        if self.current_session_id and self.current_session_id in self.active_sessions:
            with self.lock:
                self.active_sessions[self.current_session_id].errors_count += 1

        if self.on_error:
            self.on_error(f"Webcam error: {error}")

    def get_session_status(self, session_id: str = None) -> Optional[Dict]:
        """Get status of a bot session."""
        session_id = session_id or self.current_session_id
        if not session_id or session_id not in self.active_sessions:
            return None

        with self.lock:
            session = self.active_sessions[session_id]

            status = {
                "session_id": session.session_id,
                "meeting_id": session.meeting_info.meeting_id,
                "meeting_title": session.meeting_info.meeting_title,
                "status": session.status,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "duration": (session.end_time or time.time()) - session.start_time,
                "participants_count": session.participants_count,
                "messages_processed": session.messages_processed,
                "errors_count": session.errors_count,
                "bot_config": asdict(session.bot_config),
            }

            # Add component statistics if available
            if self.audio_capture:
                status["audio_stats"] = self.audio_capture.get_capture_stats()

            if self.correlation_engine:
                status["correlation_stats"] = self.correlation_engine.get_statistics()

            if self.virtual_webcam:
                status["webcam_stats"] = self.virtual_webcam.get_webcam_stats()

            return status

    def get_bot_statistics(self) -> Dict[str, Any]:
        """Get comprehensive bot statistics."""
        with self.lock:
            return {
                "bot_id": self.config.bot_id,
                "total_sessions": self.total_sessions,
                "successful_sessions": self.successful_sessions,
                "active_sessions": len(self.active_sessions),
                "total_messages_processed": self.total_messages_processed,
                "success_rate": self.successful_sessions / max(1, self.total_sessions),
                "current_session_id": self.current_session_id,
                "target_languages": self.config.target_languages,
                "virtual_webcam_enabled": self.config.virtual_webcam_enabled,
                "service_endpoints": asdict(self.config.service_endpoints),
            }


# Factory functions
def create_bot_integration(
    bot_id: str, bot_manager=None, database_manager=None, **config_kwargs
) -> GoogleMeetBotIntegration:
    """Create a bot integration instance with configuration."""
    bot_config = BotConfig(bot_id=bot_id, **config_kwargs)
    return GoogleMeetBotIntegration(bot_config, bot_manager, database_manager)


# Example usage
async def main():
    """Example usage of the bot integration."""
    # Create bot
    bot = create_bot_integration(
        bot_id="test-bot-001",
        bot_name="Test LiveTranslate Bot",
        target_languages=["en", "es", "fr"],
        virtual_webcam_enabled=True,
    )

    # Set up callbacks
    def on_transcription(data):
        print(f"Transcription: {data['speaker_name']} - {data['original_text']}")

    def on_translation(data):
        print(
            f"Translation ({data['target_language']}): {data['speaker_name']} - {data['translated_text']}"
        )

    def on_session_event(event_type, data):
        print(f"Session event: {event_type} - {data}")

    def on_error(error):
        print(f"Error: {error}")

    bot.set_transcription_callback(on_transcription)
    bot.set_translation_callback(on_translation)
    bot.set_session_event_callback(on_session_event)
    bot.set_error_callback(on_error)

    # Simulate meeting
    meeting = MeetingInfo(
        meeting_id="test-meeting-456",
        meeting_title="Test Integration Meeting",
        organizer_email="test@example.com",
        participant_count=5,
    )

    # Join meeting
    session_id = await bot.join_meeting(meeting)
    if session_id:
        print(f"Joined meeting with session: {session_id}")

        # Run for 60 seconds
        await asyncio.sleep(60)

        # Leave meeting
        await bot.leave_meeting(session_id)

        # Print final statistics
        stats = bot.get_bot_statistics()
        print(f"Final bot statistics: {json.dumps(stats, indent=2, default=str)}")
    else:
        print("Failed to join meeting")


if __name__ == "__main__":
    asyncio.run(main())
