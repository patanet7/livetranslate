"""
Caption Router

FastAPI router for caption streaming via WebSocket.
Provides real-time caption broadcasting for Fireflies translation output.

Endpoints:
- WebSocket /api/captions/stream/{session_id} - Real-time caption stream
- GET /api/captions/{session_id} - Get current captions (REST)
- GET /api/captions/{session_id}/stats - Get session statistics

Integrates with CaptionBuffer for caption management.
"""

import asyncio
import json
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status
from livetranslate_common.logging import get_logger
from models.fireflies import CaptionEntry
from pydantic import BaseModel, Field
from services.caption_buffer import CaptionBuffer, create_caption_buffer

logger = get_logger()

router = APIRouter(prefix="/captions", tags=["captions"])


# =============================================================================
# WebSocket Connection Manager
# =============================================================================


class ConnectionManager:
    """
    Manages WebSocket connections for caption streaming.

    Handles multiple clients per session and broadcasts
    caption updates to all connected clients.
    """

    def __init__(self):
        # session_id -> set of WebSocket connections
        self._connections: dict[str, set[WebSocket]] = {}
        # WebSocket -> session_id (reverse lookup)
        self._session_lookup: dict[WebSocket, str] = {}
        # WebSocket -> target_language filter (optional)
        self._language_filters: dict[WebSocket, str | None] = {}

    async def connect(
        self,
        websocket: WebSocket,
        session_id: str,
        target_language: str | None = None,
    ) -> None:
        """Accept WebSocket connection and register it."""
        await websocket.accept()

        if session_id not in self._connections:
            self._connections[session_id] = set()

        self._connections[session_id].add(websocket)
        self._session_lookup[websocket] = session_id
        self._language_filters[websocket] = target_language

        logger.info(
            f"WebSocket connected: session={session_id}, "
            f"language={target_language}, "
            f"total_connections={len(self._connections[session_id])}"
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove WebSocket connection."""
        session_id = self._session_lookup.get(websocket)
        if session_id and session_id in self._connections:
            self._connections[session_id].discard(websocket)
            if not self._connections[session_id]:
                del self._connections[session_id]

        self._session_lookup.pop(websocket, None)
        self._language_filters.pop(websocket, None)

        logger.info(f"WebSocket disconnected: session={session_id}")

    async def broadcast_to_session(
        self,
        session_id: str,
        message: dict,
        target_language: str | None = None,
    ) -> None:
        """
        Broadcast message to all connections for a session.

        Args:
            session_id: Session to broadcast to
            message: Message to send (will be JSON encoded)
            target_language: If set, only send to clients with matching filter
        """
        connections = self._connections.get(session_id, set())
        event_type = message.get("event", "unknown")

        logger.debug(
            f"Broadcasting {event_type} to session {session_id}: {len(connections)} connections"
        )

        sent_count = 0
        for websocket in connections.copy():
            try:
                # Check language filter
                client_lang = self._language_filters.get(websocket)
                if client_lang and target_language and client_lang != target_language:
                    continue

                await websocket.send_json(message)
                sent_count += 1
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                self.disconnect(websocket)

        if sent_count > 0:
            logger.info(
                f"Broadcast {event_type} to {sent_count} client(s) for session {session_id}"
            )

    def get_connection_count(self, session_id: str) -> int:
        """Get number of connections for a session."""
        return len(self._connections.get(session_id, set()))

    def get_total_connections(self) -> int:
        """Get total number of connections across all sessions."""
        return sum(len(conns) for conns in self._connections.values())


# Global connection manager
_connection_manager = ConnectionManager()


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager."""
    return _connection_manager


# =============================================================================
# Caption Buffer Manager (Singleton)
# =============================================================================


_caption_buffers: dict[str, CaptionBuffer] = {}


def get_caption_buffer(session_id: str) -> CaptionBuffer:
    """
    Get or create a CaptionBuffer for a session.

    Registers callbacks to broadcast captions to WebSocket clients.
    """
    if session_id not in _caption_buffers:
        # Uses defaults from caption_buffer.py (DRY)
        # DEFAULT_CAPTION_DURATION_SECONDS = 4.0
        # DEFAULT_MAX_AGGREGATION_TIME = 3.0
        buffer = create_caption_buffer(
            config={
                "max_captions": 20,
                "show_original": True,
            }
        )

        # Register callback for caption broadcasting
        async def on_caption_added(caption: CaptionEntry):
            await _connection_manager.broadcast_to_session(
                session_id,
                {
                    "event": "caption_added",
                    "caption": caption.model_dump(mode="json"),
                },
                target_language=caption.target_language,
            )

        async def on_caption_expired(caption_id: str):
            await _connection_manager.broadcast_to_session(
                session_id,
                {
                    "event": "caption_expired",
                    "caption_id": caption_id,
                },
            )

        # Store for session
        _caption_buffers[session_id] = buffer

        logger.info(f"Created caption buffer for session: {session_id}")

    return _caption_buffers[session_id]


def remove_caption_buffer(session_id: str) -> bool:
    """Remove a caption buffer for a session."""
    if session_id in _caption_buffers:
        del _caption_buffers[session_id]
        logger.info(f"Removed caption buffer for session: {session_id}")
        return True
    return False


# =============================================================================
# Response Models
# =============================================================================


class CaptionsResponse(BaseModel):
    """Response with current captions."""

    session_id: str
    captions: list[CaptionEntry]
    count: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CaptionStatsResponse(BaseModel):
    """Response with caption statistics."""

    session_id: str
    captions_added: int
    captions_expired: int
    current_count: int
    unique_speakers: int
    connection_count: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WebSocketMessage(BaseModel):
    """WebSocket message format."""

    event: str
    data: dict | None = None


# =============================================================================
# WebSocket Endpoint
# =============================================================================


@router.websocket("/stream/{session_id}")
async def caption_stream(
    websocket: WebSocket,
    session_id: str,
    target_language: str | None = None,
):
    """
    WebSocket endpoint for real-time caption streaming.

    Connect to receive captions as they are produced by the translation pipeline.
    Supports optional target_language filter to receive only specific translations.

    Message Format:
    - caption_added: New caption available
    - caption_expired: Caption has expired
    - caption_updated: Caption was modified
    - error: An error occurred

    Query Parameters:
    - target_language: Optional filter for specific language (e.g., "es")
    """
    manager = get_connection_manager()

    await manager.connect(websocket, session_id, target_language)

    try:
        # Send initial state
        buffer = get_caption_buffer(session_id)
        current_captions = buffer.get_active_captions()

        await websocket.send_json(
            {
                "event": "connected",
                "session_id": session_id,
                "current_captions": [c.to_dict() for c in current_captions],
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        # Keep connection open and handle messages
        while True:
            try:
                # Wait for client messages (ping/pong, filter updates, etc.)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,  # Ping every 30 seconds
                )

                try:
                    message = json.loads(data)
                    event = message.get("event")

                    if event == "ping":
                        await websocket.send_json({"event": "pong"})
                    elif event == "set_language":
                        # Update language filter
                        new_lang = message.get("language")
                        manager._language_filters[websocket] = new_lang
                        await websocket.send_json(
                            {
                                "event": "language_updated",
                                "language": new_lang,
                            }
                        )
                    elif event == "get_captions":
                        # Request current captions
                        captions = buffer.get_active_captions()
                        await websocket.send_json(
                            {
                                "event": "captions",
                                "captions": [c.to_dict() for c in captions],
                            }
                        )
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client: {data[:100]}")
                    await websocket.send_json(
                        {
                            "event": "error",
                            "message": "Invalid JSON",
                        }
                    )

            except TimeoutError:
                # Send ping on timeout
                await websocket.send_json({"event": "ping"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session={session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)


# =============================================================================
# REST Endpoints
# =============================================================================


@router.get("/{session_id}")
async def get_current_captions(session_id: str):
    """
    Get currently active captions for a session.

    Returns all captions that have not yet expired.
    """
    if session_id not in _caption_buffers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    buffer = _caption_buffers[session_id]
    captions = buffer.get_active_captions()

    return {
        "session_id": session_id,
        "captions": [c.to_dict() for c in captions],
        "count": len(captions),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/{session_id}/stats")
async def get_caption_stats(session_id: str):
    """
    Get caption statistics for a session.

    Returns counts and metrics about caption processing.
    """
    if session_id not in _caption_buffers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    buffer = _caption_buffers[session_id]
    stats = buffer.get_stats()
    manager = get_connection_manager()

    return {
        "session_id": session_id,
        "captions_added": stats.get("total_captions_added", 0),
        "captions_expired": stats.get("total_captions_expired", 0),
        "current_count": stats.get("current_caption_count", 0),
        "unique_speakers": stats.get("speakers_seen", 0),
        "connection_count": manager.get_connection_count(session_id),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_session_captions(session_id: str):
    """
    Clear all captions for a session.

    Removes the caption buffer and disconnects all WebSocket clients.
    """
    if session_id not in _caption_buffers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Notify clients
    manager = get_connection_manager()
    await manager.broadcast_to_session(
        session_id,
        {"event": "session_cleared"},
    )

    # Remove buffer
    remove_caption_buffer(session_id)


# =============================================================================
# Test/Debug Endpoint - POST captions directly
# =============================================================================


class AddCaptionRequest(BaseModel):
    """Request to add a caption."""

    text: str = Field(..., description="Translated text to display")
    original_text: str | None = Field(None, description="Original text")
    speaker_name: str = Field("Speaker", description="Speaker name")
    speaker_color: str = Field("#4CAF50", description="Speaker color hex")
    target_language: str = Field("es", description="Target language code")
    duration_seconds: float | None = Field(
        None, description="Display duration (uses buffer default if not set)"
    )
    confidence: float = Field(0.95, description="Translation confidence")


@router.post("/{session_id}", status_code=status.HTTP_201_CREATED)
async def add_caption(session_id: str, request: AddCaptionRequest):
    """
    Add a caption to a session (for testing/debugging).

    This endpoint allows direct injection of captions for testing the
    caption overlay without the full translation pipeline.

    If the same speaker is still active, text is appended to their
    existing caption instead of creating a new one.
    """
    # Get or create buffer
    buffer = get_caption_buffer(session_id)

    # Set speaker color if provided
    if request.speaker_color:
        buffer.set_speaker_color(request.speaker_name, request.speaker_color)

    # Add caption to buffer - returns (Caption, was_updated)
    caption, was_updated = buffer.add_caption(
        translated_text=request.text,
        speaker_name=request.speaker_name,
        original_text=request.original_text,
        target_language=request.target_language,
        confidence=request.confidence,
        duration=request.duration_seconds,
    )

    # Broadcast to WebSocket clients - use appropriate event
    manager = get_connection_manager()
    event_type = "caption_updated" if was_updated else "caption_added"
    await manager.broadcast_to_session(
        session_id,
        {
            "event": event_type,
            "caption": caption.to_dict(),
        },
    )

    action = "updated" if was_updated else "added"
    logger.info(f"Caption {action} for session {session_id}: {caption.translated_text[:50]}...")

    return {
        "status": "updated" if was_updated else "created",
        "caption_id": caption.id,
        "session_id": session_id,
        "was_aggregated": was_updated,
    }


# =============================================================================
# Utility Functions
# =============================================================================


async def broadcast_caption(
    session_id: str,
    caption: CaptionEntry,
) -> None:
    """
    Broadcast a caption to all connected clients.

    Call this when adding a new caption from the translation pipeline.
    """
    manager = get_connection_manager()
    await manager.broadcast_to_session(
        session_id,
        {
            "event": "caption_added",
            "caption": caption.model_dump(mode="json"),
        },
        target_language=caption.target_language,
    )


async def broadcast_expiration(
    session_id: str,
    caption_id: str,
) -> None:
    """
    Broadcast caption expiration to all connected clients.
    """
    manager = get_connection_manager()
    await manager.broadcast_to_session(
        session_id,
        {
            "event": "caption_expired",
            "caption_id": caption_id,
        },
    )
