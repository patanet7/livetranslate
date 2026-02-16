#!/usr/bin/env python3
"""
Whisper Service Integration for Translation Service

This module handles real-time integration with the Whisper service for
continuous transcription -> translation pipeline.
"""

import json
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

import aiohttp
import websockets
from flask import jsonify, request
from livetranslate_common.logging import get_logger
from translation_service import TranslationRequest, TranslationService

logger = get_logger()


class WhisperTranslationBridge:
    """
    Bridge between Whisper service and Translation service for real-time processing
    """

    def __init__(
        self,
        translation_service: TranslationService,
        whisper_url: str = "http://whisper-service:5001",
    ):
        """Initialize the bridge"""
        self.translation_service = translation_service
        self.whisper_url = whisper_url
        self.whisper_ws_url = (
            whisper_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        )

        # Active translation sessions
        self.active_sessions: dict[str, dict[str, Any]] = {}

        logger.info(f"Whisper-Translation bridge initialized for {whisper_url}")

    async def start_real_time_translation(
        self, session_id: str, config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Start real-time translation for a Whisper session

        Args:
            session_id: Session identifier
            config: Translation configuration

        Returns:
            Session configuration
        """
        try:
            # Create translation session
            translation_session = await self.translation_service.create_session(session_id, config)

            # Store session configuration
            self.active_sessions[session_id] = {
                "translation_config": translation_session,
                "source_language": config.get("source_language", "auto"),
                "target_language": config.get("target_language", "en"),
                "created_at": datetime.now().isoformat(),
                "transcription_count": 0,
                "translation_count": 0,
            }

            logger.info(f"Started real-time translation for session {session_id}")
            return translation_session

        except Exception as e:
            logger.error(f"Failed to start real-time translation: {e}")
            raise

    async def process_transcription(
        self, session_id: str, transcription_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Process transcription from Whisper and return translation

        Args:
            session_id: Session identifier
            transcription_data: Transcription result from Whisper

        Returns:
            Translation result
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"No active translation session for {session_id}")

            session = self.active_sessions[session_id]

            # Extract transcription text
            text = transcription_data.get("text", "").strip()
            if not text:
                return {"error": "Empty transcription text"}

            # Update session stats
            session["transcription_count"] += 1

            # Create translation request
            translation_request = TranslationRequest(
                text=text,
                source_language=session["source_language"],
                target_language=session["target_language"],
                session_id=session_id,
                confidence_threshold=0.7,  # Lower threshold for real-time
                context=self._build_context(session_id),
            )

            # Perform translation
            translation_result = await self.translation_service.translate(translation_request)

            # Update session stats
            session["translation_count"] += 1
            session["last_translation"] = datetime.now().isoformat()

            # Build response with both transcription and translation
            response = {
                "session_id": session_id,
                "transcription": {
                    "text": text,
                    "confidence": transcription_data.get("confidence", 0.0),
                    "timestamp": transcription_data.get("timestamp"),
                    "speaker_id": transcription_data.get("speaker_id"),
                    "language": transcription_data.get("language"),
                },
                "translation": {
                    "text": translation_result.translated_text,
                    "source_language": translation_result.source_language,
                    "target_language": translation_result.target_language,
                    "confidence": translation_result.confidence_score,
                    "backend": translation_result.backend_used,
                    "processing_time": translation_result.processing_time,
                },
                "timestamp": datetime.now().isoformat(),
            }

            return response

        except Exception as e:
            logger.error(f"Failed to process transcription: {e}")
            return {"error": str(e), "session_id": session_id}

    async def stream_translations(
        self, session_id: str, transcription_stream: AsyncGenerator[dict[str, Any], None]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream translations from a transcription stream

        Args:
            session_id: Session identifier
            transcription_stream: Async generator of transcription results

        Yields:
            Translation results
        """
        try:
            async for transcription_data in transcription_stream:
                translation_result = await self.process_transcription(
                    session_id, transcription_data
                )
                yield translation_result

        except Exception as e:
            logger.error(f"Error in translation streaming: {e}")
            yield {"error": str(e), "session_id": session_id}

    async def connect_to_whisper_websocket(self, session_id: str, on_translation: callable):
        """
        Connect to Whisper WebSocket and process translations in real-time

        Args:
            session_id: Session identifier
            on_translation: Callback function for translation results
        """
        try:
            async with websockets.connect(f"{self.whisper_ws_url}/{session_id}") as websocket:
                logger.info(f"Connected to Whisper WebSocket for session {session_id}")

                async for message in websocket:
                    try:
                        transcription_data = json.loads(message)

                        # Process transcription and get translation
                        translation_result = await self.process_transcription(
                            session_id, transcription_data
                        )

                        # Call the callback with the result
                        await on_translation(translation_result)

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from Whisper WebSocket: {message}")
                    except Exception as e:
                        logger.error(f"Error processing Whisper message: {e}")

        except Exception as e:
            logger.error(f"Whisper WebSocket connection failed: {e}")
            raise

    async def stop_real_time_translation(self, session_id: str) -> dict[str, Any]:
        """
        Stop real-time translation for a session

        Args:
            session_id: Session identifier

        Returns:
            Final session statistics
        """
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}

            # Get session stats
            session = self.active_sessions.pop(session_id)

            # Close translation service session
            final_stats = await self.translation_service.close_session(session_id)

            # Build final response
            result = {
                "session_id": session_id,
                "duration": (
                    datetime.now() - datetime.fromisoformat(session["created_at"])
                ).total_seconds(),
                "transcription_count": session["transcription_count"],
                "translation_count": session["translation_count"],
                "final_stats": final_stats,
            }

            logger.info(f"Stopped real-time translation for session {session_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to stop real-time translation: {e}")
            return {"error": str(e)}

    async def get_session_status(self, session_id: str) -> dict[str, Any] | None:
        """Get status of a translation session"""
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]
        translation_stats = await self.translation_service.get_session(session_id)

        return {
            "session_id": session_id,
            "active": True,
            "source_language": session["source_language"],
            "target_language": session["target_language"],
            "created_at": session["created_at"],
            "transcription_count": session["transcription_count"],
            "translation_count": session["translation_count"],
            "translation_stats": translation_stats,
        }

    async def check_whisper_health(self) -> bool:
        """Check if Whisper service is healthy"""
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    f"{self.whisper_url}/api/health", timeout=aiohttp.ClientTimeout(total=5)
                ) as response,
            ):
                return response.status == 200
        except Exception:
            return False

    def _build_context(self, session_id: str) -> str:
        """Build context from recent translations"""
        # This could be enhanced to maintain translation history
        # For now, return empty context
        return ""


# Integration endpoints for API server
def add_whisper_integration_routes(app, translation_service: TranslationService):
    """
    Add Whisper integration routes to the Flask app

    Args:
        app: Flask application
        translation_service: Translation service instance
    """

    # Create bridge instance
    whisper_bridge = WhisperTranslationBridge(translation_service)

    @app.route("/api/whisper/start", methods=["POST"])
    async def start_whisper_translation():
        """Start real-time Whisper -> Translation pipeline"""
        try:
            data = request.get_json()
            session_id = data.get("session_id")
            if not session_id:
                return jsonify({"error": "Missing session_id"}), 400

            config = data.get("config", {})
            result = await whisper_bridge.start_real_time_translation(session_id, config)

            return jsonify({"status": "started", "session_id": session_id, "config": result})

        except Exception as e:
            logger.error(f"Failed to start Whisper translation: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/whisper/stop/<session_id>", methods=["POST"])
    async def stop_whisper_translation(session_id: str):
        """Stop real-time Whisper -> Translation pipeline"""
        try:
            result = await whisper_bridge.stop_real_time_translation(session_id)
            return jsonify(result)

        except Exception as e:
            logger.error(f"Failed to stop Whisper translation: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/whisper/process", methods=["POST"])
    async def process_whisper_transcription():
        """Process a single transcription from Whisper"""
        try:
            data = request.get_json()
            session_id = data.get("session_id")
            transcription_data = data.get("transcription")

            if not session_id or not transcription_data:
                return jsonify({"error": "Missing session_id or transcription"}), 400

            result = await whisper_bridge.process_transcription(session_id, transcription_data)
            return jsonify(result)

        except Exception as e:
            logger.error(f"Failed to process transcription: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/whisper/status/<session_id>", methods=["GET"])
    async def get_whisper_session_status(session_id: str):
        """Get status of Whisper translation session"""
        try:
            status = await whisper_bridge.get_session_status(session_id)
            if not status:
                return jsonify({"error": "Session not found"}), 404

            return jsonify(status)

        except Exception as e:
            logger.error(f"Failed to get session status: {e}")
            return jsonify({"error": str(e)}), 500

    return whisper_bridge
