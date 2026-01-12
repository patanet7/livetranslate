#!/usr/bin/env python3
"""
WebSocket Audio Streaming Router

Provides real-time audio streaming from frontend → orchestration → Whisper → frontend.
Follows the same pattern as bot containers for consistency.

Message Flow:
1. Frontend connects via WebSocket
2. Frontend sends: authenticate, start_session, audio_chunk (base64)
3. Orchestration forwards audio to Whisper service
4. Whisper returns segments
5. Orchestration sends segments back to frontend

This matches the bot pattern:
- Bot: Container → WebSocket → Orchestration → Whisper
- Frontend: Browser → WebSocket → Orchestration → Whisper

Moved from standalone routers/websocket_audio.py for package consolidation.
"""

import logging
import base64
import asyncio
from typing import Dict, Set
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from socketio_whisper_client import SocketIOWhisperClient


logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Track active connections and sessions
_active_connections: Set[str] = set()
_session_to_connections: Dict[str, Set[str]] = {}

# Create singleton Socket.IO Whisper client
whisper_client = SocketIOWhisperClient(
    whisper_host="localhost",  # Use localhost for dev (or "whisper" for Docker)
    whisper_port=5001,
    auto_reconnect=True,
)


@router.websocket("/stream")
async def websocket_audio_stream(websocket: WebSocket):
    """
    WebSocket endpoint for frontend audio streaming

    Endpoint: ws://orchestration:3000/api/audio/stream

    Incoming Messages (from frontend):
    1. authenticate: {"type": "authenticate", "user_id": "user-123", "token": "..."}
    2. start_session: {"type": "start_session", "session_id": "session-xyz", "config": {...}}
    3. audio_chunk: {"type": "audio_chunk", "audio": "base64...", "timestamp": "ISO8601"}
    4. end_session: {"type": "end_session", "session_id": "session-xyz"}

    Outgoing Messages (to frontend):
    1. authenticated: {"type": "authenticated", "connection_id": "...", "user_id": "..."}
    2. session_started: {"type": "session_started", "session_id": "...", "timestamp": "..."}
    3. segment: {"type": "segment", "text": "...", "speaker": "...", "confidence": 0.95, ...}
    4. translation: {"type": "translation", "text": "...", "source_lang": "...", "target_lang": "..."}
    5. error: {"type": "error", "error": "...", "timestamp": "..."}
    """
    connection_id = f"frontend-{id(websocket)}"
    session_id = None
    user_id = None

    try:
        # Accept WebSocket connection
        await websocket.accept()
        _active_connections.add(connection_id)
        logger.info(f"Frontend WebSocket connection accepted: {connection_id}")

        # Send welcome message
        await websocket.send_json(
            {
                "type": "connected",
                "connection_id": connection_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Message handling loop
        while True:
            # Receive message from frontend
            message = await websocket.receive_json()

            msg_type = message.get("type")
            logger.debug(f"Received message type: {msg_type}")

            # Handle authenticate
            if msg_type == "authenticate":
                user_id = message.get("user_id")
                token = message.get("token")

                # TODO: Implement actual authentication
                logger.info(f"Authenticated user: {user_id}")

                await websocket.send_json(
                    {
                        "type": "authenticated",
                        "connection_id": connection_id,
                        "user_id": user_id,
                    }
                )

            # Handle start_session
            elif msg_type == "start_session":
                session_id = message.get("session_id")
                config = message.get("config", {})

                logger.info(f"Starting session: {session_id}")

                # Track session to connection mapping
                if session_id not in _session_to_connections:
                    _session_to_connections[session_id] = set()
                _session_to_connections[session_id].add(connection_id)

                # Start Whisper WebSocket session
                try:
                    # Ensure whisper client is connected
                    if not whisper_client.connected:
                        logger.info("Connecting to Whisper service...")
                        await whisper_client.connect()

                    # Set up callback to forward segments to frontend
                    def segment_callback(segment: dict):
                        """Forward segments from Whisper to frontend"""
                        asyncio.create_task(
                            websocket.send_json({"type": "segment", **segment})
                        )

                    whisper_client.on_segment(segment_callback)

                    await whisper_client.start_stream(
                        session_id=session_id, config=config
                    )

                    logger.info(f"Whisper session started: {session_id}")

                    await websocket.send_json(
                        {
                            "type": "session_started",
                            "session_id": session_id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

                except Exception as e:
                    logger.error(f"Failed to start Whisper session: {e}")
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error": f"Failed to start session: {e}",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

            # Handle audio_chunk
            elif msg_type == "audio_chunk":
                if not session_id:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error": "No active session. Please send start_session first.",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    continue

                audio_base64 = message.get("audio")
                timestamp_str = message.get("timestamp")

                if not audio_base64:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error": "Missing audio data",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    continue

                try:
                    audio_data = base64.b64decode(audio_base64)
                    logger.debug(
                        f"Received audio chunk: {len(audio_data)} bytes, forwarding to Whisper"
                    )

                    # Forward to Whisper service
                    # Segments will come back via the callback registered in start_session
                    await whisper_client.send_audio_chunk(
                        session_id=session_id,
                        audio_data=audio_data,
                        timestamp=timestamp_str,
                    )

                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}")
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error": f"Audio processing failed: {e}",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

            # Handle end_session
            elif msg_type == "end_session":
                end_session_id = message.get("session_id") or session_id

                logger.info(f"Ending session: {end_session_id}")

                # End Whisper session
                if end_session_id:
                    await whisper_client.close_stream(end_session_id)
                    # Clean up session tracking
                    if end_session_id in _session_to_connections:
                        _session_to_connections[end_session_id].discard(connection_id)
                        if not _session_to_connections[end_session_id]:
                            del _session_to_connections[end_session_id]

                session_id = None

                await websocket.send_json(
                    {
                        "type": "session_ended",
                        "session_id": end_session_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

            # Handle ping
            elif msg_type == "ping":
                await websocket.send_json(
                    {"type": "pong", "timestamp": datetime.now(timezone.utc).isoformat()}
                )

            else:
                logger.warning(f"Unknown message type: {msg_type}")
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": f"Unknown message type: {msg_type}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

    except WebSocketDisconnect:
        logger.info(f"Frontend disconnected: {connection_id}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)

    finally:
        # Cleanup connection tracking
        _active_connections.discard(connection_id)

        # Cleanup session if active
        if session_id:
            try:
                await whisper_client.end_session(session_id)
                if session_id in _session_to_connections:
                    _session_to_connections[session_id].discard(connection_id)
                    if not _session_to_connections[session_id]:
                        del _session_to_connections[session_id]
            except Exception as e:
                logger.error(f"Error ending session during cleanup: {e}")

        logger.info(f"Cleaned up connection: {connection_id}")


# Health check endpoint
@router.get("/health")
async def websocket_audio_health():
    """
    Health check for WebSocket audio streaming

    Returns:
        status, active_connections, whisper_status
    """
    return {
        "status": "healthy",
        "active_connections": len(_active_connections),
        "whisper_connected": whisper_client.is_connected()
        if hasattr(whisper_client, "is_connected")
        else False,
        "sessions": list(_session_to_connections.keys()),
    }
