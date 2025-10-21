#!/usr/bin/env python3
"""
Streaming Coordinator - Orchestrates the Complete Real-Time Pipeline

This coordinator ties together:
1. Frontend WebSocket Handler (receives audio from browser)
2. Whisper WebSocket Client (streams audio to Whisper service)
3. Segment Deduplication (deduplicates incoming segments)
4. Speaker Grouping (groups segments by speaker)
5. Translation Service (translates segments)

Complete Flow:
    Frontend (Browser)
        ↓ audio chunks
    Orchestration (Frontend Handler)
        ↓ audio chunks
    Orchestration (Whisper Client)
        ↓ audio chunks
    Whisper Service
        ↓ segments
    Orchestration (Deduplicator)
        ↓ deduplicated segments
    Orchestration (Speaker Grouper)
        ↓ grouped segments
    Translation Service
        ↓ translations
    Orchestration (Frontend Handler)
        ↓ segments + translations
    Frontend (Browser)
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, field

from websocket_frontend_handler import WebSocketFrontendHandler
from websocket_whisper_client import WebSocketWhisperClient
from segment_deduplicator import SegmentDeduplicator
from speaker_grouper import SpeakerGrouper

logger = logging.getLogger(__name__)


@dataclass
class StreamingSessionState:
    """State for an active streaming session"""
    session_id: str
    frontend_connection_id: str
    whisper_session_id: str
    config: Dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Processing components
    deduplicator: SegmentDeduplicator = field(default_factory=SegmentDeduplicator)
    speaker_grouper: SpeakerGrouper = field(default_factory=SpeakerGrouper)

    # Statistics
    audio_chunks_sent: int = 0
    segments_received: int = 0
    segments_deduplicated: int = 0
    translations_generated: int = 0

    def update_stats(
        self,
        audio_chunks: int = 0,
        segments: int = 0,
        deduplicated: int = 0,
        translations: int = 0
    ):
        """Update session statistics"""
        self.audio_chunks_sent += audio_chunks
        self.segments_received += segments
        self.segments_deduplicated += deduplicated
        self.translations_generated += translations


class StreamingCoordinator:
    """
    Coordinates the complete real-time streaming pipeline

    Usage:
        coordinator = StreamingCoordinator(
            whisper_host="localhost",
            whisper_port=5001
        )

        await coordinator.start()

        # WebSocket server is now running on localhost:8000
        # Frontend connects and streams audio
        # Coordinator handles the entire pipeline automatically
    """

    def __init__(
        self,
        whisper_host: str = "localhost",
        whisper_port: int = 5001,
        frontend_host: str = "0.0.0.0",
        frontend_port: int = 8000,
        enable_translation: bool = False,
        translation_service_url: Optional[str] = None
    ):
        """
        Initialize streaming coordinator

        Args:
            whisper_host: Whisper service hostname
            whisper_port: Whisper service WebSocket port
            frontend_host: Host to bind frontend WebSocket server
            frontend_port: Port for frontend WebSocket server
            enable_translation: Enable translation service
            translation_service_url: Translation service URL
        """
        self.whisper_host = whisper_host
        self.whisper_port = whisper_port
        self.frontend_host = frontend_host
        self.frontend_port = frontend_port
        self.enable_translation = enable_translation
        self.translation_service_url = translation_service_url

        # Components
        self.frontend_handler = WebSocketFrontendHandler()
        self.whisper_client = WebSocketWhisperClient(
            whisper_host=whisper_host,
            whisper_port=whisper_port,
            auto_reconnect=True
        )

        # Session tracking
        self.sessions: Dict[str, StreamingSessionState] = {}
        self.connection_to_session: Dict[str, str] = {}  # frontend_connection_id -> session_id

        # Setup callbacks
        self._setup_callbacks()

    def _setup_callbacks(self):
        """Setup callbacks between components"""

        # Frontend Handler Callbacks
        self.frontend_handler.on_session_start(self._handle_session_start)
        self.frontend_handler.on_audio_chunk(self._handle_audio_chunk)
        self.frontend_handler.on_session_end(self._handle_session_end)
        self.frontend_handler.on_connection_change(self._handle_connection_change)

        # Whisper Client Callbacks
        self.whisper_client.on_segment(self._handle_whisper_segment)
        self.whisper_client.on_error(self._handle_whisper_error)

    async def start(self):
        """Start the streaming coordinator"""
        logger.info("🚀 Starting Streaming Coordinator...")

        # Connect to Whisper service
        logger.info(f"Connecting to Whisper at {self.whisper_host}:{self.whisper_port}")
        connected = await self.whisper_client.connect()

        if not connected:
            raise RuntimeError("Failed to connect to Whisper service")

        logger.info("✅ Connected to Whisper service")

        # Start frontend WebSocket server
        import websockets
        logger.info(f"Starting frontend WebSocket server on {self.frontend_host}:{self.frontend_port}")

        async with websockets.serve(
            self.frontend_handler.handle_connection,
            self.frontend_host,
            self.frontend_port
        ):
            logger.info(f"✅ Frontend WebSocket server running on ws://{self.frontend_host}:{self.frontend_port}")
            await asyncio.Future()  # Run forever

    async def stop(self):
        """Stop the streaming coordinator"""
        logger.info("Stopping Streaming Coordinator...")

        # Close all sessions
        for session_id in list(self.sessions.keys()):
            await self._cleanup_session(session_id)

        # Disconnect from Whisper
        await self.whisper_client.close()

        logger.info("✅ Streaming Coordinator stopped")

    # Session Management

    async def _handle_session_start(
        self,
        frontend_connection_id: str,
        session_id: str,
        config: Dict[str, Any]
    ):
        """Handle session start from frontend"""
        logger.info(f"📝 Starting session: {session_id} (frontend: {frontend_connection_id})")

        # Create Whisper session ID (unique per frontend connection)
        whisper_session_id = f"whisper-{session_id}-{frontend_connection_id}"

        # Start Whisper streaming session
        await self.whisper_client.start_stream(
            session_id=whisper_session_id,
            config=config
        )

        # Create session state
        session_state = StreamingSessionState(
            session_id=session_id,
            frontend_connection_id=frontend_connection_id,
            whisper_session_id=whisper_session_id,
            config=config
        )

        self.sessions[session_id] = session_state
        self.connection_to_session[frontend_connection_id] = session_id

        logger.info(f"✅ Session started: {session_id}")

    async def _handle_audio_chunk(
        self,
        frontend_connection_id: str,
        audio_data: bytes,
        timestamp: datetime
    ):
        """Handle audio chunk from frontend"""
        # Get session for this connection
        session_id = self.connection_to_session.get(frontend_connection_id)
        if not session_id:
            logger.warning(f"No session for connection: {frontend_connection_id}")
            return

        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return

        logger.debug(f"🎵 Audio chunk for session {session_id}: {len(audio_data)} bytes")

        # Forward audio to Whisper
        try:
            await self.whisper_client.send_audio_chunk(
                session_id=session.whisper_session_id,
                audio_data=audio_data,
                timestamp=timestamp
            )

            # Update stats
            session.update_stats(audio_chunks=1)

        except Exception as e:
            logger.error(f"Error sending audio to Whisper: {e}")
            await self.frontend_handler.send_segment(
                frontend_connection_id,
                {"type": "error", "error": f"Whisper error: {e}"}
            )

    async def _handle_session_end(
        self,
        frontend_connection_id: str,
        session_id: str
    ):
        """Handle session end from frontend"""
        logger.info(f"⏹️ Ending session: {session_id}")

        await self._cleanup_session(session_id)

    async def _handle_connection_change(
        self,
        frontend_connection_id: str,
        connected: bool
    ):
        """Handle frontend connection state change"""
        if not connected:
            # Connection closed - cleanup any active session
            session_id = self.connection_to_session.get(frontend_connection_id)
            if session_id:
                logger.info(f"Connection {frontend_connection_id} closed - cleaning up session {session_id}")
                await self._cleanup_session(session_id)

    # Segment Processing

    async def _handle_whisper_segment(self, segment: Dict[str, Any]):
        """Handle segment from Whisper service"""
        whisper_session_id = segment.get("session_id")

        if not whisper_session_id:
            logger.warning("Segment missing session_id")
            return

        # Find session by whisper_session_id
        session = None
        for s in self.sessions.values():
            if s.whisper_session_id == whisper_session_id:
                session = s
                break

        if not session:
            logger.warning(f"Session not found for Whisper session: {whisper_session_id}")
            return

        logger.debug(f"📄 Segment for session {session.session_id}: {segment.get('text', '')[:50]}")

        # Update stats
        session.update_stats(segments=1)

        # Step 1: Deduplicate segment
        deduplicated_segments = session.deduplicator.merge_segments([segment])
        session.update_stats(deduplicated=len(deduplicated_segments))

        # Step 2: Group by speaker
        grouped_segments = session.speaker_grouper.group_by_speaker(deduplicated_segments)

        # Step 3: Send segments to frontend
        for grouped_segment in grouped_segments:
            await self.frontend_handler.send_segment(
                session.frontend_connection_id,
                grouped_segment
            )

        # Step 4: Translate if enabled
        if self.enable_translation and self.translation_service_url:
            for grouped_segment in grouped_segments:
                translation = await self._translate_segment(grouped_segment)
                if translation:
                    session.update_stats(translations=1)
                    await self.frontend_handler.send_translation(
                        session.frontend_connection_id,
                        translation
                    )

    async def _translate_segment(self, segment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Translate a segment

        TODO: Implement actual translation service call

        Args:
            segment: Segment to translate

        Returns:
            Translation data or None
        """
        # Placeholder - implement when translation service is ready
        logger.debug(f"Translation requested for: {segment.get('text', '')[:50]}")
        return None

    async def _handle_whisper_error(self, error: str):
        """Handle error from Whisper service"""
        logger.error(f"❌ Whisper error: {error}")

        # Notify all active sessions
        for session in self.sessions.values():
            await self.frontend_handler.send_segment(
                session.frontend_connection_id,
                {
                    "type": "error",
                    "error": f"Whisper service error: {error}",
                    "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                }
            )

    # Cleanup

    async def _cleanup_session(self, session_id: str):
        """Cleanup session resources"""
        session = self.sessions.get(session_id)
        if not session:
            return

        logger.info(f"🧹 Cleaning up session: {session_id}")

        # Close Whisper session
        try:
            await self.whisper_client.close_stream(session.whisper_session_id)
        except Exception as e:
            logger.error(f"Error closing Whisper session: {e}")

        # Log session statistics
        logger.info(
            f"Session {session_id} stats: "
            f"audio_chunks={session.audio_chunks_sent}, "
            f"segments={session.segments_received}, "
            f"deduplicated={session.segments_deduplicated}, "
            f"translations={session.translations_generated}"
        )

        # Remove from tracking
        self.connection_to_session.pop(session.frontend_connection_id, None)
        del self.sessions[session_id]

        logger.info(f"✅ Session cleaned up: {session_id}")

    # Status Methods

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        session = self.sessions.get(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "frontend_connection_id": session.frontend_connection_id,
            "whisper_session_id": session.whisper_session_id,
            "config": session.config,
            "created_at": session.created_at.isoformat(),
            "stats": {
                "audio_chunks_sent": session.audio_chunks_sent,
                "segments_received": session.segments_received,
                "segments_deduplicated": session.segments_deduplicated,
                "translations_generated": session.translations_generated
            }
        }

    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all session information"""
        return {
            session_id: self.get_session_info(session_id)
            for session_id in self.sessions
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        frontend_stats = self.frontend_handler.get_stats()
        whisper_stats = self.whisper_client.get_connection_stats()

        return {
            "active_sessions": len(self.sessions),
            "frontend": frontend_stats,
            "whisper": whisper_stats,
            "total_audio_chunks": sum(s.audio_chunks_sent for s in self.sessions.values()),
            "total_segments": sum(s.segments_received for s in self.sessions.values()),
            "total_translations": sum(s.translations_generated for s in self.sessions.values())
        }


# Main entry point
async def main():
    """Run the streaming coordinator"""
    import os

    # Get configuration from environment
    whisper_host = os.getenv("WHISPER_HOST", "localhost")
    whisper_port = int(os.getenv("WHISPER_PORT", "5001"))
    frontend_host = os.getenv("FRONTEND_HOST", "0.0.0.0")
    frontend_port = int(os.getenv("FRONTEND_PORT", "8000"))

    # Create coordinator
    coordinator = StreamingCoordinator(
        whisper_host=whisper_host,
        whisper_port=whisper_port,
        frontend_host=frontend_host,
        frontend_port=frontend_port
    )

    try:
        # Start coordinator
        await coordinator.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await coordinator.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    asyncio.run(main())
