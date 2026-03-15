#!/usr/bin/env python3
"""
Service Integration for Triton-based Translation Service

This module handles integration with other LiveTranslate services:
- WebSocket communication with frontend
- Real-time translation pipeline with whisper service
- Speaker diarization integration
- Session management and state synchronization
"""

import json
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import aiohttp
import websockets
from livetranslate_common.logging import get_logger
from translation_service import TranslationRequest, TranslationService

logger = get_logger()


@dataclass
class ServiceEndpoints:
    """Configuration for other LiveTranslate services"""

    whisper_service: str = "http://whisper-service:5001"
    speaker_service: str = "http://speaker-service:5002"
    frontend_service: str = "http://frontend-service:3000"
    websocket_service: str = "ws://websocket-service:8765"

    # Triton-specific endpoints
    triton_server: str = "http://localhost:8000"
    triton_metrics: str = "http://localhost:8002/metrics"


@dataclass
class TranslationSession:
    """Translation session with real-time capabilities"""

    session_id: str
    source_language: str = "auto"
    target_language: str = "en"
    speaker_id: str | None = None
    created_at: datetime = None
    last_activity: datetime = None

    # Real-time state
    active_transcription: bool = False
    translation_buffer: list[str] = None
    confidence_threshold: float = 0.8

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_activity is None:
            self.last_activity = datetime.now()
        if self.translation_buffer is None:
            self.translation_buffer = []


class TritonTranslationIntegration:
    """
    Integration service for Triton-based translation with other LiveTranslate components
    """

    def __init__(
        self, translation_service: TranslationService, endpoints: ServiceEndpoints | None = None
    ):
        """Initialize integration service"""
        self.translation_service = translation_service
        self.endpoints = endpoints or ServiceEndpoints()
        self.sessions: dict[str, TranslationSession] = {}
        self.websocket_connections: dict[str, websockets.WebSocketServerProtocol] = {}

        # Service health tracking
        self.service_health: dict[str, bool] = {}
        self.last_health_check = datetime.now()

        logger.info("Triton Translation Integration initialized")

    async def initialize(self):
        """Initialize integration with other services"""
        try:
            # Check health of dependent services
            await self._check_service_health()

            # Initialize WebSocket server for real-time communication
            await self._start_websocket_server()

            # Register with service discovery (if available)
            await self._register_with_services()

            logger.info("Translation integration fully initialized")

        except Exception as e:
            logger.error(f"Failed to initialize translation integration: {e}")
            raise

    async def handle_whisper_transcription(
        self, session_id: str, transcription_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle real-time transcription from Whisper service

        Args:
            session_id: Session identifier
            transcription_data: Transcription result from Whisper

        Returns:
            Translation result with timing and confidence
        """
        try:
            # Extract transcription details
            text = transcription_data.get("text", "")
            timestamp = transcription_data.get("timestamp", time.time())
            speaker_id = transcription_data.get("speaker_id")

            if not text.strip():
                return {"error": "Empty transcription text"}

            # Get or create session
            session = await self._get_or_create_session(session_id)
            session.last_activity = datetime.now()

            # Update speaker if provided
            if speaker_id:
                session.speaker_id = speaker_id

            # Create translation request
            translation_request = TranslationRequest(
                text=text,
                source_language=session.source_language,
                target_language=session.target_language,
                session_id=session_id,
                confidence_threshold=session.confidence_threshold,
                context=self._build_context(session),
            )

            # Perform translation
            start_time = time.time()
            translation_result = await self.translation_service.translate(translation_request)
            processing_time = time.time() - start_time

            # Build response
            response = {
                "session_id": session_id,
                "original_text": text,
                "translated_text": translation_result.translated_text,
                "source_language": translation_result.source_language,
                "target_language": translation_result.target_language,
                "confidence_score": translation_result.confidence_score,
                "processing_time": processing_time,
                "timestamp": timestamp,
                "speaker_id": speaker_id,
                "backend_used": translation_result.backend_used,
            }

            # Update session buffer
            session.translation_buffer.append(translation_result.translated_text)
            if len(session.translation_buffer) > 50:  # Keep last 50 translations
                session.translation_buffer = session.translation_buffer[-50:]

            # Broadcast to WebSocket clients
            await self._broadcast_translation(session_id, response)

            # Send to speaker service for context (if available)
            await self._update_speaker_context(session_id, response)

            return response

        except Exception as e:
            logger.error(f"Error handling Whisper transcription: {e}")
            return {"error": str(e), "session_id": session_id}

    async def handle_streaming_translation(
        self, session_id: str, text_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Handle streaming translation for real-time use cases

        Args:
            session_id: Session identifier
            text_stream: Async generator of text chunks

        Yields:
            Streaming translation results
        """
        session = await self._get_or_create_session(session_id)
        accumulated_text = ""

        async for text_chunk in text_stream:
            accumulated_text += text_chunk

            # Translate when we have enough text or at sentence boundaries
            if len(accumulated_text) > 50 or text_chunk.endswith((".", "!", "?", "\n")):
                try:
                    translation_request = TranslationRequest(
                        text=accumulated_text,
                        source_language=session.source_language,
                        target_language=session.target_language,
                        session_id=session_id,
                        streaming=True,
                    )

                    # Stream translation
                    async for translation_chunk in self.translation_service.translate_stream(
                        translation_request
                    ):
                        result = {
                            "session_id": session_id,
                            "text_chunk": translation_chunk,
                            "is_partial": True,
                            "timestamp": time.time(),
                        }

                        # Broadcast to WebSocket clients
                        await self._broadcast_translation(session_id, result)

                        yield result

                    # Reset accumulator
                    accumulated_text = ""

                except Exception as e:
                    logger.error(f"Error in streaming translation: {e}")
                    yield {"error": str(e), "session_id": session_id}

    async def create_translation_session(
        self, session_config: dict[str, Any]
    ) -> TranslationSession:
        """
        Create a new translation session

        Args:
            session_config: Session configuration

        Returns:
            Created translation session
        """
        session_id = session_config.get("session_id") or f"session_{int(time.time())}"

        session = TranslationSession(
            session_id=session_id,
            source_language=session_config.get("source_language", "auto"),
            target_language=session_config.get("target_language", "en"),
            confidence_threshold=session_config.get("confidence_threshold", 0.8),
        )

        self.sessions[session_id] = session

        # Notify other services about new session
        await self._notify_session_created(session)

        logger.info(f"Created translation session: {session_id}")
        return session

    async def get_session_status(self, session_id: str) -> dict[str, Any] | None:
        """Get status of a translation session"""
        session = self.sessions.get(session_id)
        if not session:
            return None

        # Get translation service stats
        service_stats = await self.translation_service.get_session(session_id)

        return {
            "session_id": session_id,
            "source_language": session.source_language,
            "target_language": session.target_language,
            "speaker_id": session.speaker_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "active_transcription": session.active_transcription,
            "translation_count": len(session.translation_buffer),
            "service_stats": service_stats,
        }

    async def close_session(self, session_id: str) -> dict[str, Any]:
        """Close a translation session and return final stats"""
        session = self.sessions.pop(session_id, None)
        if not session:
            return {"error": "Session not found"}

        # Get final stats from translation service
        final_stats = await self.translation_service.close_session(session_id)

        # Close WebSocket connection if exists
        if session_id in self.websocket_connections:
            await self.websocket_connections[session_id].close()
            del self.websocket_connections[session_id]

        # Notify other services
        await self._notify_session_closed(session_id)

        result = {
            "session_id": session_id,
            "duration": (datetime.now() - session.created_at).total_seconds(),
            "translation_count": len(session.translation_buffer),
            "final_stats": final_stats,
        }

        logger.info(f"Closed translation session: {session_id}")
        return result

    async def get_service_health(self) -> dict[str, Any]:
        """Get health status of all integrated services"""
        # Update health check if needed
        if (datetime.now() - self.last_health_check).seconds > 30:
            await self._check_service_health()

        # Get translation service status
        translation_status = await self.translation_service.get_service_status()

        return {
            "translation_service": translation_status,
            "integrated_services": self.service_health,
            "active_sessions": len(self.sessions),
            "websocket_connections": len(self.websocket_connections),
            "last_health_check": self.last_health_check.isoformat(),
        }

    # Private helper methods

    async def _get_or_create_session(self, session_id: str) -> TranslationSession:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            await self.create_translation_session({"session_id": session_id})
        return self.sessions[session_id]

    def _build_context(self, session: TranslationSession) -> str:
        """Build context from recent translations"""
        if not session.translation_buffer:
            return ""

        # Use last few translations as context
        recent_translations = session.translation_buffer[-5:]
        return " ".join(recent_translations)

    async def _broadcast_translation(self, session_id: str, data: dict[str, Any]):
        """Broadcast translation result to WebSocket clients"""
        if session_id in self.websocket_connections:
            try:
                websocket = self.websocket_connections[session_id]
                await websocket.send(json.dumps(data))
            except Exception as e:
                logger.warning(f"Failed to broadcast to WebSocket: {e}")
                # Remove dead connection
                if session_id in self.websocket_connections:
                    del self.websocket_connections[session_id]

    async def _check_service_health(self):
        """Check health of dependent services"""
        services_to_check = {
            "whisper": f"{self.endpoints.whisper_service}/api/health",
            "speaker": f"{self.endpoints.speaker_service}/api/health",
            "frontend": f"{self.endpoints.frontend_service}/api/health",
            "triton": f"{self.endpoints.triton_server}/v2/health",
        }

        async with aiohttp.ClientSession() as session:
            for service_name, health_url in services_to_check.items():
                try:
                    async with session.get(
                        health_url, timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        self.service_health[service_name] = response.status == 200
                except Exception:
                    self.service_health[service_name] = False

        self.last_health_check = datetime.now()

    async def _start_websocket_server(self):
        """Start WebSocket server for real-time communication"""
        # This would typically be handled by the websocket-service
        # For now, we'll just prepare for WebSocket connections
        logger.info("WebSocket integration prepared")

    async def _register_with_services(self):
        """Register this service with other LiveTranslate services"""
        # This would typically involve service discovery registration
        # with registration data containing service info and endpoints
        logger.info("Service registration prepared")

    async def _update_speaker_context(self, session_id: str, translation_data: dict[str, Any]):
        """Update speaker service with translation context"""
        if not self.service_health.get("speaker", False):
            return

        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"{self.endpoints.speaker_service}/api/context",
                    json={
                        "session_id": session_id,
                        "speaker_id": translation_data.get("speaker_id"),
                        "translated_text": translation_data.get("translated_text"),
                        "timestamp": translation_data.get("timestamp"),
                    },
                    timeout=aiohttp.ClientTimeout(total=5),
                )
        except Exception as e:
            logger.warning(f"Failed to update speaker context: {e}")

    async def _notify_session_created(self, session: TranslationSession):
        """Notify other services about new session"""
        notification = {
            "event": "session_created",
            "session_id": session.session_id,
            "source_language": session.source_language,
            "target_language": session.target_language,
        }

        # Send to other services as needed
        logger.debug(f"Session created notification: {notification}")

    async def _notify_session_closed(self, session_id: str):
        """Notify other services about closed session"""
        notification = {"event": "session_closed", "session_id": session_id}

        # Send to other services as needed
        logger.debug(f"Session closed notification: {notification}")


# Factory function for easy integration setup
async def create_triton_integration(
    translation_service: TranslationService, endpoints: ServiceEndpoints | None = None
) -> TritonTranslationIntegration:
    """
    Create and initialize Triton translation integration

    Args:
        translation_service: Initialized translation service
        endpoints: Service endpoints configuration

    Returns:
        Initialized integration service
    """
    integration = TritonTranslationIntegration(translation_service, endpoints)
    await integration.initialize()
    return integration
