"""
WebSocket API router for real-time communication

Enhanced WebSocket endpoints with FastAPI for real-time audio processing,
bot management, and system notifications.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
    status,
)
from fastapi.websockets import WebSocketState

from src.models.websocket import (
    WebSocketMessage,
    WebSocketResponse,
    ConnectionInfo,
    SessionInfo,
    ConnectionStats,
    MessageType,
    BroadcastMessage,
)
from src.dependencies import get_websocket_manager
from src.utils.rate_limiting import RateLimiter

logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket rate limiting
ws_rate_limiter = RateLimiter()


@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    websocket_manager=Depends(get_websocket_manager),
):
    """
    Main WebSocket endpoint for real-time communication

    - **token**: Optional authentication token
    - **session_id**: Session identifier to join
    - **user_id**: User identifier
    """
    connection_id = None

    try:
        # Accept WebSocket connection
        await websocket.accept()
        logger.info(f"WebSocket connection attempt from {websocket.client}")

        # Authenticate if token provided
        authenticated_user = None
        if token:
            try:
                # TODO: Implement WebSocket token verification
                authenticated_user = {"user_id": user_id or "anonymous"}
                user_id = authenticated_user.get("user_id", user_id)
            except Exception as e:
                logger.warning(f"WebSocket authentication failed: {e}")
                await websocket.close(code=4001, reason="Authentication failed")
                return

        # Register connection
        client_ip = str(websocket.client.host) if websocket.client else "unknown"
        user_agent = websocket.headers.get("user-agent", "unknown")
        connection_id = await websocket_manager.connect(
            websocket, client_ip, user_agent
        )

        logger.info(f"WebSocket connection registered: {connection_id}")

        # Send welcome message
        welcome_message = WebSocketResponse(
            type=MessageType.CONNECT,
            message="Connected successfully",
            data={
                "connection_id": connection_id,
                "session_id": session_id,
                "user_id": user_id,
                "authenticated": authenticated_user is not None,
                "server_time": datetime.utcnow().isoformat(),
            },
        )

        await websocket.send_text(welcome_message.json())

        # Message handling loop
        while True:
            try:
                # Receive message
                raw_message = await websocket.receive_text()

                # Rate limiting check
                client_ip = (
                    str(websocket.client.host) if websocket.client else "unknown"
                )
                if not await ws_rate_limiter.is_allowed(
                    client_ip, "websocket", 100, 60
                ):
                    await _send_error(websocket, "Rate limit exceeded", "RATE_LIMIT")
                    continue

                # Parse message
                try:
                    message_data = json.loads(raw_message)
                    message = WebSocketMessage(**message_data)
                except (json.JSONDecodeError, ValueError) as e:
                    await _send_error(
                        websocket, f"Invalid message format: {e}", "INVALID_FORMAT"
                    )
                    continue

                # Process message
                await _handle_websocket_message(
                    websocket, message, connection_id, websocket_manager
                )

            except WebSocketDisconnect:
                logger.info(f"WebSocket client disconnected: {connection_id}")
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await _send_error(websocket, "Internal error", "INTERNAL_ERROR")

    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")

    finally:
        # Unregister connection
        if connection_id:
            await websocket_manager.disconnect(connection_id)
            logger.info(f"WebSocket connection unregistered: {connection_id}")


@router.websocket("/session/{session_id}")
async def session_websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    user_id: Optional[str] = None,
    role: Optional[str] = "participant",
    websocket_manager=Depends(get_websocket_manager),
):
    """
    WebSocket endpoint for joining a specific session

    - **session_id**: Session to join
    - **user_id**: User identifier
    - **role**: User role in session (owner, participant, observer)
    """
    connection_id = None

    try:
        await websocket.accept()
        logger.info(f"Session WebSocket connection for session {session_id}")

        # Register connection to session
        client_ip = str(websocket.client.host) if websocket.client else "unknown"
        user_agent = websocket.headers.get("user-agent", "unknown")
        connection_id = await websocket_manager.connect(
            websocket, client_ip, user_agent
        )

        # Join session
        await websocket_manager._join_session(connection_id, session_id, {"role": role})

        # Send session info
        session_info = await websocket_manager.get_session_info(session_id)
        welcome_message = WebSocketResponse(
            type=MessageType.JOIN_SESSION,
            message="Joined session successfully",
            data={
                "connection_id": connection_id,
                "session_info": session_info,
                "role": role,
            },
        )

        await websocket.send_text(welcome_message.json())

        # Notify other session participants
        notification = WebSocketMessage(
            type=MessageType.SESSION_UPDATE,
            data={"event": "user_joined", "user_id": user_id, "role": role},
            session_id=session_id,
        )

        await websocket_manager.broadcast_to_session(
            session_id, notification, exclude_connection=connection_id
        )

        # Message handling loop
        while True:
            try:
                raw_message = await websocket.receive_text()

                # Rate limiting
                client_ip = (
                    str(websocket.client.host) if websocket.client else "unknown"
                )
                if not await ws_rate_limiter.is_allowed(
                    client_ip, "websocket_session", 100, 60
                ):
                    await _send_error(websocket, "Rate limit exceeded", "RATE_LIMIT")
                    continue

                # Parse and handle message
                try:
                    message_data = json.loads(raw_message)
                    message = WebSocketMessage(**message_data)
                    message.session_id = session_id  # Ensure session context

                    await _handle_session_message(
                        websocket, message, connection_id, session_id, websocket_manager
                    )

                except (json.JSONDecodeError, ValueError) as e:
                    await _send_error(
                        websocket, f"Invalid message format: {e}", "INVALID_FORMAT"
                    )

            except WebSocketDisconnect:
                logger.info(f"Session WebSocket disconnected: {connection_id}")
                break

    except Exception as e:
        logger.error(f"Session WebSocket error: {e}")

    finally:
        if connection_id:
            # Leave session and notify participants
            if session_id:
                await websocket_manager._leave_session(connection_id, session_id)

                notification = WebSocketMessage(
                    type=MessageType.SESSION_UPDATE,
                    data={"event": "user_left", "user_id": user_id},
                    session_id=session_id,
                )

                await websocket_manager.broadcast_to_session(session_id, notification)

            await websocket_manager.disconnect(connection_id)


@router.get("/connections", response_model=List[ConnectionInfo])
async def get_active_connections(
    websocket_manager=Depends(get_websocket_manager),
) -> List[ConnectionInfo]:
    """Get list of active WebSocket connections"""
    try:
        connections = websocket_manager.get_all_connections()
        return [ConnectionInfo(**conn) for conn in connections]
    except Exception as e:
        logger.error(f"Failed to get active connections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve active connections",
        )


@router.get("/sessions", response_model=List[SessionInfo])
async def get_active_sessions(
    websocket_manager=Depends(get_websocket_manager),
) -> List[SessionInfo]:
    """Get list of active sessions"""
    try:
        sessions = websocket_manager.get_all_sessions()
        return [SessionInfo(**session) for session in sessions]
    except Exception as e:
        logger.error(f"Failed to get active sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve active sessions",
        )


@router.get("/stats", response_model=ConnectionStats)
async def get_websocket_stats(
    websocket_manager=Depends(get_websocket_manager),
) -> ConnectionStats:
    """Get WebSocket connection statistics"""
    try:
        stats = websocket_manager.get_stats()
        return ConnectionStats(**stats)
    except Exception as e:
        logger.error(f"Failed to get WebSocket stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve WebSocket statistics",
        )


@router.post("/broadcast")
async def broadcast_message(
    broadcast: BroadcastMessage, websocket_manager=Depends(get_websocket_manager)
) -> Dict[str, Any]:
    """Broadcast message to multiple connections"""
    try:
        await websocket_manager.broadcast_to_all(broadcast.dict())
        result = {"targets_reached": 1, "total_targets": 1}  # Simplified for now
        return {
            "status": "success",
            "message": "Broadcast sent successfully",
            "targets_reached": result.get("targets_reached", 0),
            "total_targets": result.get("total_targets", 0),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to broadcast message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to broadcast message",
        )


@router.post("/sessions/{session_id}/message")
async def send_session_message(
    session_id: str,
    message: WebSocketMessage,
    exclude_user_id: Optional[str] = None,
    websocket_manager=Depends(get_websocket_manager),
) -> Dict[str, Any]:
    """Send message to all participants in a session"""
    try:
        message.session_id = session_id

        result = await websocket_manager.broadcast_to_session(
            session_id, message, exclude_user_id=exclude_user_id
        )

        return {
            "status": "success",
            "message": "Session message sent successfully",
            "participants_reached": result.get("participants_reached", 0),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to send session message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send session message",
        )


@router.delete("/connections/{connection_id}")
async def disconnect_connection(
    connection_id: str,
    reason: Optional[str] = "Disconnected by admin",
    websocket_manager=Depends(get_websocket_manager),
) -> Dict[str, Any]:
    """Forcefully disconnect a WebSocket connection"""
    try:
        success = await websocket_manager.disconnect_connection(connection_id, reason)

        if success:
            return {
                "status": "success",
                "message": f"Connection {connection_id} disconnected",
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Connection {connection_id} not found",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disconnect connection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to disconnect connection",
        )


# Helper functions


async def _handle_websocket_message(
    websocket: WebSocket,
    message: WebSocketMessage,
    connection_id: str,
    websocket_manager,
):
    """Handle incoming WebSocket message"""

    try:
        if message.type == MessageType.PING:
            # Handle ping/pong
            pong_response = WebSocketResponse(
                type=MessageType.PONG, data={"timestamp": datetime.utcnow().isoformat()}
            )
            await websocket.send_text(pong_response.json())

        elif message.type == MessageType.CREATE_SESSION:
            # Create new session
            session_id = await websocket_manager.create_session(
                connection_id, message.data.get("config", {})
            )

            response = WebSocketResponse(
                type=MessageType.CREATE_SESSION,
                message="Session created successfully",
                data={"session_id": session_id},
            )
            await websocket.send_text(response.json())

        elif message.type == MessageType.JOIN_SESSION:
            # Join existing session
            session_id = message.data.get("session_id")
            role = message.data.get("role", "participant")

            if session_id:
                success = await websocket_manager.join_session(
                    connection_id, session_id, role
                )
                if success:
                    response = WebSocketResponse(
                        type=MessageType.JOIN_SESSION,
                        message="Joined session successfully",
                        data={"session_id": session_id, "role": role},
                    )
                else:
                    response = WebSocketResponse(
                        type=MessageType.ERROR,
                        success=False,
                        message="Failed to join session",
                        data={"session_id": session_id},
                    )
                await websocket.send_text(response.json())

        elif message.type == MessageType.AUDIO_CHUNK:
            # Handle audio chunk for processing
            await _handle_audio_chunk(
                websocket, message, connection_id, websocket_manager
            )

        elif message.type == MessageType.BOT_SPAWN:
            # Handle bot spawn request
            await _handle_bot_spawn(
                websocket, message, connection_id, websocket_manager
            )

        else:
            # Forward message to session if applicable
            if message.session_id:
                await websocket_manager.broadcast_to_session(
                    message.session_id, message, exclude_connection=connection_id
                )

    except Exception as e:
        logger.error(f"Error handling WebSocket message: {e}")
        await _send_error(websocket, "Failed to process message", "PROCESSING_ERROR")


async def _handle_session_message(
    websocket: WebSocket,
    message: WebSocketMessage,
    connection_id: str,
    session_id: str,
    websocket_manager,
):
    """Handle messages within a session context"""

    try:
        if message.type == MessageType.AUDIO_CHUNK:
            # Process audio in session context
            await _handle_audio_chunk(
                websocket, message, connection_id, websocket_manager
            )

            # Broadcast to other session participants
            await websocket_manager.broadcast_to_session(
                session_id, message, exclude_connection=connection_id
            )

        elif message.type == MessageType.USER_MESSAGE:
            # Broadcast user message to session
            await websocket_manager.broadcast_to_session(
                session_id, message, exclude_connection=connection_id
            )

        elif message.type == MessageType.BOT_STATUS:
            # Broadcast bot status updates
            await websocket_manager.broadcast_to_session(session_id, message)

        else:
            # Default: broadcast to session
            await websocket_manager.broadcast_to_session(
                session_id, message, exclude_connection=connection_id
            )

    except Exception as e:
        logger.error(f"Error handling session message: {e}")
        await _send_error(
            websocket, "Failed to process session message", "SESSION_ERROR"
        )


async def _handle_audio_chunk(
    websocket: WebSocket,
    message: WebSocketMessage,
    connection_id: str,
    websocket_manager,
):
    """Handle audio chunk processing"""

    try:
        # Extract audio data
        audio_data = message.data.get("audio_data")
        chunk_id = message.data.get("chunk_id", 0)

        if not audio_data:
            await _send_error(websocket, "Missing audio data", "MISSING_DATA")
            return

        # Process audio chunk (this would integrate with audio service)
        # For now, echo back with processing confirmation
        response = WebSocketResponse(
            type=MessageType.AUDIO_RESULT,
            message="Audio chunk processed",
            data={
                "chunk_id": chunk_id,
                "processing_time_ms": 50,  # Simulated
                "status": "processed",
            },
        )

        await websocket.send_text(response.json())

    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")
        await _send_error(websocket, "Audio processing failed", "AUDIO_ERROR")


async def _handle_bot_spawn(
    websocket: WebSocket,
    message: WebSocketMessage,
    connection_id: str,
    websocket_manager,
):
    """Handle bot spawn request"""

    try:
        bot_config = message.data.get("config", {})

        if not bot_config:
            await _send_error(websocket, "Missing bot configuration", "MISSING_CONFIG")
            return

        # This would integrate with bot manager
        bot_id = f"bot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        response = WebSocketResponse(
            type=MessageType.BOT_STATUS,
            message="Bot spawn initiated",
            data={"bot_id": bot_id, "status": "spawning", "config": bot_config},
        )

        await websocket.send_text(response.json())

        # Broadcast to session if applicable
        if message.session_id:
            await websocket_manager.broadcast_to_session(
                message.session_id,
                WebSocketMessage(
                    type=MessageType.BOT_STATUS,
                    data={
                        "bot_id": bot_id,
                        "status": "spawning",
                        "event": "bot_spawn_initiated",
                    },
                    session_id=message.session_id,
                ),
                exclude_connection=connection_id,
            )

    except Exception as e:
        logger.error(f"Error handling bot spawn: {e}")
        await _send_error(websocket, "Bot spawn failed", "BOT_ERROR")


async def _send_error(websocket: WebSocket, error_message: str, error_code: str):
    """Send error message to WebSocket client"""

    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            error_response = WebSocketResponse(
                type=MessageType.ERROR,
                success=False,
                message=error_message,
                error_code=error_code,
                data={"timestamp": datetime.utcnow().isoformat()},
            )
            await websocket.send_text(error_response.json())
    except Exception as e:
        logger.error(f"Failed to send error message: {e}")
