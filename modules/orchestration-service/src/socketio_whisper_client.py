#!/usr/bin/env python3
"""
Socket.IO Whisper Client

Connects to Whisper service via Socket.IO for real-time streaming transcription.
This replaces WebSocketWhisperClient with proper Socket.IO protocol support.
"""

import logging
import socketio
import asyncio
import base64
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WhisperSessionState:
    """State tracking for a Whisper streaming session"""
    session_id: str
    config: Dict[str, Any]
    chunks_sent: int = 0
    segments_received: int = 0
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now(timezone.utc)


class SocketIOWhisperClient:
    """
    Socket.IO client for Whisper service streaming transcription.

    Provides real-time audio streaming to Whisper service using Socket.IO protocol.
    """

    def __init__(
        self,
        whisper_host: str = "localhost",
        whisper_port: int = 5001,
        auto_reconnect: bool = True
    ):
        """
        Initialize Socket.IO Whisper client

        Args:
            whisper_host: Whisper service hostname
            whisper_port: Whisper service port
            auto_reconnect: Enable automatic reconnection
        """
        self.whisper_host = whisper_host
        self.whisper_port = whisper_port
        self.whisper_url = f"http://{whisper_host}:{whisper_port}"

        # Create Socket.IO client
        self.sio = socketio.AsyncClient(
            reconnection=auto_reconnect,
            reconnection_attempts=5,
            reconnection_delay=1,
            reconnection_delay_max=5,
            logger=False,
            engineio_logger=False
        )

        # Session tracking
        self.sessions: Dict[str, WhisperSessionState] = {}
        self.connected = False

        # Callbacks
        self._segment_callbacks = []
        self._error_callbacks = []
        self._connection_callbacks = []

        # Register Socket.IO event handlers
        self._register_handlers()

        logger.info(f"📡 SocketIOWhisperClient initialized for {self.whisper_url}")

    def _register_handlers(self):
        """Register Socket.IO event handlers"""

        @self.sio.event
        async def connect():
            """Handle connection to Whisper service"""
            self.connected = True
            logger.info(f"✅ Connected to Whisper service at {self.whisper_url}")
            self._notify_connection(True)

        @self.sio.event
        async def disconnect():
            """Handle disconnection from Whisper service"""
            self.connected = False
            logger.warning(f"🔌 Disconnected from Whisper service")
            self._notify_connection(False)

        @self.sio.event
        async def connect_error(data):
            """Handle connection error"""
            logger.error(f"❌ Connection error: {data}")
            self._notify_error(f"Connection error: {data}")

        @self.sio.on('transcription_result')
        async def on_transcription_result(data):
            """Handle transcription segment from Whisper"""
            logger.debug(f"📄 Received transcription result")

            # Update session activity
            session_id = data.get('session_id')
            if session_id and session_id in self.sessions:
                self.sessions[session_id].segments_received += 1
                self.sessions[session_id].update_activity()

            # Notify callbacks
            self._notify_segment(data)

        @self.sio.on('error')
        async def on_error(data):
            """Handle error from Whisper"""
            error_msg = data.get('error', 'Unknown error')
            logger.error(f"❌ Whisper error: {error_msg}")
            self._notify_error(error_msg)

        @self.sio.on('session_started')
        async def on_session_started(data):
            """Handle session started confirmation"""
            session_id = data.get('session_id')
            logger.info(f"✅ Session started: {session_id}")

        @self.sio.on('pong')
        async def on_pong(data):
            """Handle pong response"""
            logger.debug("🏓 Received pong")

    async def connect(self) -> bool:
        """
        Connect to Whisper service via Socket.IO

        Returns:
            bool: True if connected successfully
        """
        if self.connected:
            logger.info("⚠️ Already connected to Whisper service")
            return True

        try:
            logger.info(f"🔌 Connecting to Whisper service at {self.whisper_url}")
            await self.sio.connect(self.whisper_url)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Whisper: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Whisper service"""
        if self.sio.connected:
            await self.sio.disconnect()
        self.connected = False
        logger.info("🔌 Disconnected from Whisper service")

    async def start_stream(
        self,
        session_id: str,
        config: Dict[str, Any]
    ) -> str:
        """
        Start a streaming session

        Args:
            session_id: Unique session identifier
            config: Whisper configuration (model, language, etc.)

        Returns:
            str: Session ID

        Raises:
            RuntimeError: If not connected
        """
        if not self.connected:
            raise RuntimeError("Not connected to Whisper service")

        # Create session state
        session_state = WhisperSessionState(
            session_id=session_id,
            config=config
        )
        self.sessions[session_id] = session_state

        # Send join_session event to Whisper
        await self.sio.emit('join_session', {
            'session_id': session_id,
            'config': config
        })

        logger.info(f"🎬 Started stream for session: {session_id}")
        return session_id

    async def send_audio_chunk(
        self,
        session_id: str,
        audio_data: bytes,
        timestamp: Optional[str] = None
    ):
        """
        Send audio chunk to Whisper for processing

        Args:
            session_id: Session identifier
            audio_data: Raw audio data (bytes)
            timestamp: Chunk timestamp (optional)

        Raises:
            RuntimeError: If not connected or session not found
        """
        if not self.connected:
            raise RuntimeError("Not connected to Whisper service")

        if session_id not in self.sessions:
            raise RuntimeError(f"Session not found: {session_id}")

        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        # Use current timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        # Send transcribe_stream event
        await self.sio.emit('transcribe_stream', {
            'session_id': session_id,
            'audio_data': audio_base64,
            'timestamp': timestamp
        })

        # Update session stats
        session = self.sessions[session_id]
        session.chunks_sent += 1
        session.update_activity()

        logger.debug(f"🎵 Sent audio chunk for session {session_id} ({len(audio_data)} bytes)")

    async def close_stream(self, session_id: str):
        """
        Close a streaming session

        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            # Send leave_session event
            await self.sio.emit('leave_session', {
                'session_id': session_id
            })

            # Remove session
            del self.sessions[session_id]
            logger.info(f"⏹️ Closed stream for session: {session_id}")

    def on_segment(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register callback for transcription segments

        Args:
            callback: Function to call with segment data
        """
        self._segment_callbacks.append(callback)

    def on_error(self, callback: Callable[[str], None]):
        """
        Register callback for errors

        Args:
            callback: Function to call with error message
        """
        self._error_callbacks.append(callback)

    def on_connection_change(self, callback: Callable[[bool], None]):
        """
        Register callback for connection state changes

        Args:
            callback: Function to call with connection state
        """
        self._connection_callbacks.append(callback)

    def _notify_segment(self, segment: Dict[str, Any]):
        """Notify all segment callbacks"""
        for callback in self._segment_callbacks:
            try:
                callback(segment)
            except Exception as e:
                logger.error(f"Error in segment callback: {e}")

    def _notify_error(self, error: str):
        """Notify all error callbacks"""
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    def _notify_connection(self, connected: bool):
        """Notify all connection callbacks"""
        for callback in self._connection_callbacks:
            try:
                callback(connected)
            except Exception as e:
                logger.error(f"Error in connection callback: {e}")

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information

        Args:
            session_id: Session identifier

        Returns:
            Session info dict or None if not found
        """
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        return {
            'session_id': session.session_id,
            'chunks_sent': session.chunks_sent,
            'segments_received': session.segments_received,
            'last_activity': session.last_activity.isoformat(),
            'config': session.config
        }

    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active sessions

        Returns:
            Dict mapping session_id to session info
        """
        return {
            sid: self.get_session_info(sid)
            for sid in self.sessions.keys()
        }
