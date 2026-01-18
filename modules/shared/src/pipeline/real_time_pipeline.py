#!/usr/bin/env python3
"""
Real-time Pipeline Orchestrator for LiveTranslate

This module orchestrates the complete pipeline:
Audio -> Whisper (Transcription) -> Speaker (Diarization) -> Translation -> Frontend

Key Features:
- End-to-end pipeline coordination
- WebSocket-based real-time streaming
- Session state management across services
- Error handling and recovery
- Performance monitoring
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import aiohttp
import websockets

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the real-time pipeline"""

    # Service endpoints
    whisper_url: str = "http://whisper:5001"
    speaker_url: str = "http://speaker:5002"
    translation_url: str = "http://translation:5003"
    frontend_url: str = "http://frontend:3000"

    # WebSocket endpoints
    whisper_ws: str = "ws://whisper:5001/ws"
    speaker_ws: str = "ws://speaker:5002/ws"
    translation_ws: str = "ws://translation:5003/ws"
    frontend_ws: str = "ws://frontend:3000/ws"

    # Pipeline settings
    enable_speaker_diarization: bool = True
    enable_translation: bool = True
    buffer_duration: float = 3.0
    confidence_threshold: float = 0.7

    # Language settings
    source_language: str = "auto"
    target_language: str = "en"


@dataclass
class PipelineEvent:
    """Event in the pipeline processing"""

    event_id: str
    session_id: str
    event_type: str  # audio_chunk, transcription, speaker_update, translation, error
    timestamp: str
    data: dict[Any, Any]
    processing_time: float | None = None

    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class RealTimePipeline:
    """
    Orchestrates the complete real-time processing pipeline
    """

    def __init__(self, config: PipelineConfig | None = None):
        """Initialize the pipeline orchestrator"""
        self.config = config or PipelineConfig()
        self.active_sessions: dict[str, dict] = {}
        self.websocket_connections: dict[str, websockets.WebSocketServerProtocol] = {}

        # Pipeline state
        self.is_running = False
        self.event_queue = asyncio.Queue()
        self.processing_tasks: list[asyncio.Task] = []

        logger.info("RealTimePipeline initialized")

    async def start_session(self, session_id: str, config: dict | None = None) -> dict:
        """
        Start a new real-time processing session

        Args:
            session_id: Unique session identifier
            config: Session-specific configuration

        Returns:
            Session information
        """
        try:
            session_config = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "config": config or {},
                "state": {
                    "audio_chunks_processed": 0,
                    "transcriptions_generated": 0,
                    "translations_generated": 0,
                    "last_activity": datetime.now().isoformat(),
                },
                "pipeline_stages": {
                    "whisper": {"status": "initializing", "last_update": None},
                    "speaker": {"status": "initializing", "last_update": None},
                    "translation": {"status": "initializing", "last_update": None},
                },
            }

            # Initialize sessions in all services
            await self._initialize_service_sessions(session_id, config)

            self.active_sessions[session_id] = session_config

            # Start processing task for this session
            task = asyncio.create_task(self._process_session_events(session_id))
            self.processing_tasks.append(task)

            logger.info(f"Started real-time session: {session_id}")
            return session_config

        except Exception as e:
            logger.error(f"Failed to start session {session_id}: {e}")
            raise

    async def process_audio_chunk(
        self, session_id: str, audio_data: bytes, metadata: dict | None = None
    ) -> dict:
        """
        Process an audio chunk through the complete pipeline

        Args:
            session_id: Session identifier
            audio_data: Raw audio data
            metadata: Additional metadata (sample_rate, format, etc.)

        Returns:
            Processing result with all pipeline outputs
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        start_time = time.time()

        try:
            # Create pipeline event
            event = PipelineEvent(
                event_id=str(uuid.uuid4()),
                session_id=session_id,
                event_type="audio_chunk",
                data={"audio_size": len(audio_data), "metadata": metadata or {}},
                timestamp=datetime.now().isoformat(),
            )

            # Step 1: Send to Whisper for transcription
            transcription_result = await self._process_whisper(session_id, audio_data, metadata)

            if not transcription_result or not transcription_result.get("text"):
                return {"status": "no_speech", "event_id": event.event_id}

            # Step 2: Send to Speaker service for diarization (if enabled)
            speaker_result = None
            if self.config.enable_speaker_diarization:
                speaker_result = await self._process_speaker(
                    session_id, audio_data, transcription_result
                )

            # Step 3: Send to Translation service (if enabled)
            translation_result = None
            if self.config.enable_translation:
                translation_result = await self._process_translation(
                    session_id, transcription_result, speaker_result
                )

            # Step 4: Combine results and send to frontend
            combined_result = {
                "event_id": event.event_id,
                "session_id": session_id,
                "timestamp": event.timestamp,
                "processing_time": time.time() - start_time,
                "transcription": transcription_result,
                "speaker": speaker_result,
                "translation": translation_result,
            }

            # Send to frontend
            await self._send_to_frontend(session_id, combined_result)

            # Update session state
            await self._update_session_state(session_id, combined_result)

            return combined_result

        except Exception as e:
            logger.error(f"Pipeline processing failed for session {session_id}: {e}")

            # Send error to frontend
            error_result = {
                "event_id": str(uuid.uuid4()),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

            await self._send_to_frontend(session_id, error_result)
            raise

    async def _process_whisper(
        self, session_id: str, audio_data: bytes, metadata: dict | None
    ) -> dict | None:
        """Process audio through Whisper service"""
        try:
            async with aiohttp.ClientSession() as session:
                # Prepare form data for file upload
                data = aiohttp.FormData()
                data.add_field("file", audio_data, filename="audio.wav", content_type="audio/wav")
                data.add_field("session_id", session_id)

                if metadata:
                    data.add_field("metadata", json.dumps(metadata))

                async with session.post(
                    f"{self.config.whisper_url}/transcribe", data=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()

                        # Update pipeline stage
                        if session_id in self.active_sessions:
                            self.active_sessions[session_id]["pipeline_stages"]["whisper"] = {
                                "status": "success",
                                "last_update": datetime.now().isoformat(),
                            }

                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"Whisper service error: {response.status} - {error_text}")
                        return None

        except Exception as e:
            logger.error(f"Whisper processing failed: {e}")

            # Update pipeline stage
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["pipeline_stages"]["whisper"] = {
                    "status": "error",
                    "error": str(e),
                    "last_update": datetime.now().isoformat(),
                }

            return None

    async def _process_speaker(
        self, session_id: str, audio_data: bytes, transcription: dict
    ) -> dict | None:
        """Process audio through Speaker service"""
        try:
            async with aiohttp.ClientSession() as session:
                # Prepare form data
                data = aiohttp.FormData()
                data.add_field("file", audio_data, filename="audio.wav", content_type="audio/wav")
                data.add_field("session_id", session_id)
                data.add_field("transcription", json.dumps(transcription))

                async with session.post(
                    f"{self.config.speaker_url}/diarize", data=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()

                        # Update pipeline stage
                        if session_id in self.active_sessions:
                            self.active_sessions[session_id]["pipeline_stages"]["speaker"] = {
                                "status": "success",
                                "last_update": datetime.now().isoformat(),
                            }

                        return result
                    else:
                        error_text = await response.text()
                        logger.warning(f"Speaker service error: {response.status} - {error_text}")
                        return None

        except Exception as e:
            logger.warning(f"Speaker processing failed: {e}")

            # Update pipeline stage
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["pipeline_stages"]["speaker"] = {
                    "status": "error",
                    "error": str(e),
                    "last_update": datetime.now().isoformat(),
                }

            return None

    async def _process_translation(
        self, session_id: str, transcription: dict, speaker: dict | None
    ) -> dict | None:
        """Process transcription through Translation service"""
        try:
            # Prepare translation request
            translation_request = {
                "text": transcription.get("text", ""),
                "source_language": self.config.source_language,
                "target_language": self.config.target_language,
                "session_id": session_id,
                "context": {"transcription": transcription, "speaker": speaker},
            }

            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"{self.config.translation_url}/translate", json=translation_request
                ) as response,
            ):
                if response.status == 200:
                    result = await response.json()

                    # Update pipeline stage
                    if session_id in self.active_sessions:
                        self.active_sessions[session_id]["pipeline_stages"]["translation"] = {
                            "status": "success",
                            "last_update": datetime.now().isoformat(),
                        }

                    return result
                else:
                    error_text = await response.text()
                    logger.warning(f"Translation service error: {response.status} - {error_text}")
                    return None

        except Exception as e:
            logger.warning(f"Translation processing failed: {e}")

            # Update pipeline stage
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["pipeline_stages"]["translation"] = {
                    "status": "error",
                    "error": str(e),
                    "last_update": datetime.now().isoformat(),
                }

            return None

    async def _send_to_frontend(self, session_id: str, data: dict):
        """Send processed data to frontend via WebSocket"""
        try:
            # If we have a direct WebSocket connection to frontend
            if session_id in self.websocket_connections:
                websocket = self.websocket_connections[session_id]
                await websocket.send(json.dumps({"type": "pipeline_result", "data": data}))
            else:
                # Fallback: HTTP POST to frontend
                async with aiohttp.ClientSession() as session:
                    await session.post(f"{self.config.frontend_url}/api/pipeline/result", json=data)

        except Exception as e:
            logger.warning(f"Failed to send data to frontend: {e}")

    async def _update_session_state(self, session_id: str, result: dict):
        """Update session state with processing results"""
        if session_id not in self.active_sessions:
            return

        session = self.active_sessions[session_id]
        state = session["state"]

        # Update counters
        state["audio_chunks_processed"] += 1

        if result.get("transcription"):
            state["transcriptions_generated"] += 1

        if result.get("translation"):
            state["translations_generated"] += 1

        state["last_activity"] = datetime.now().isoformat()

    async def _initialize_service_sessions(self, session_id: str, config: dict | None):
        """Initialize sessions in all backend services"""
        services = [
            (self.config.whisper_url, "whisper"),
            (self.config.speaker_url, "speaker"),
            (self.config.translation_url, "translation"),
        ]

        for service_url, service_name in services:
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        f"{service_url}/api/sessions",
                        json={"session_id": session_id, "config": config},
                    )
                logger.debug(f"Initialized session in {service_name} service")

            except Exception as e:
                logger.warning(f"Failed to initialize session in {service_name}: {e}")

    async def _process_session_events(self, session_id: str):
        """Process events for a specific session"""
        try:
            while session_id in self.active_sessions:
                # Process any queued events for this session
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Session event processing failed for {session_id}: {e}")

    async def stop_session(self, session_id: str) -> dict:
        """
        Stop a real-time processing session

        Args:
            session_id: Session identifier

        Returns:
            Final session statistics
        """
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        try:
            session = self.active_sessions.pop(session_id)

            # Close WebSocket connection if exists
            if session_id in self.websocket_connections:
                await self.websocket_connections[session_id].close()
                del self.websocket_connections[session_id]

            # Stop processing tasks
            for task in self.processing_tasks:
                if not task.done():
                    task.cancel()

            # Close sessions in backend services
            await self._close_service_sessions(session_id)

            # Calculate final statistics
            duration = (
                datetime.now() - datetime.fromisoformat(session["created_at"])
            ).total_seconds()

            result = {
                "session_id": session_id,
                "duration": duration,
                "final_state": session["state"],
                "pipeline_stages": session["pipeline_stages"],
            }

            logger.info(f"Stopped real-time session: {session_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to stop session {session_id}: {e}")
            return {"error": str(e)}

    async def _close_service_sessions(self, session_id: str):
        """Close sessions in all backend services"""
        services = [
            (self.config.whisper_url, "whisper"),
            (self.config.speaker_url, "speaker"),
            (self.config.translation_url, "translation"),
        ]

        for service_url, service_name in services:
            try:
                async with aiohttp.ClientSession() as session:
                    await session.delete(f"{service_url}/api/sessions/{session_id}")
                logger.debug(f"Closed session in {service_name} service")

            except Exception as e:
                logger.warning(f"Failed to close session in {service_name}: {e}")

    async def get_session_status(self, session_id: str) -> dict | None:
        """Get status of a processing session"""
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]

        # Calculate session duration
        duration = (datetime.now() - datetime.fromisoformat(session["created_at"])).total_seconds()

        return {
            "session_id": session_id,
            "duration": duration,
            "state": session["state"],
            "pipeline_stages": session["pipeline_stages"],
            "config": session["config"],
        }

    async def get_all_sessions(self) -> dict[str, dict]:
        """Get status of all active sessions"""
        sessions_status = {}

        for session_id in self.active_sessions:
            sessions_status[session_id] = await self.get_session_status(session_id)

        return sessions_status

    def add_websocket_connection(
        self, session_id: str, websocket: websockets.WebSocketServerProtocol
    ):
        """Add a WebSocket connection for a session"""
        self.websocket_connections[session_id] = websocket
        logger.info(f"Added WebSocket connection for session: {session_id}")

    def remove_websocket_connection(self, session_id: str):
        """Remove a WebSocket connection for a session"""
        if session_id in self.websocket_connections:
            del self.websocket_connections[session_id]
            logger.info(f"Removed WebSocket connection for session: {session_id}")


# Factory function
async def create_pipeline(config: PipelineConfig | None = None) -> RealTimePipeline:
    """Create and initialize a real-time pipeline"""
    pipeline = RealTimePipeline(config)
    return pipeline
