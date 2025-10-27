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

    # Whisper configuration (extracted from config for easy access)
    model_name: str = "large-v3-turbo"
    language: str = "en"
    beam_size: int = 5
    sample_rate: int = 16000
    task: str = "transcribe"
    target_language: str = "en"
    enable_vad: bool = True

    # Domain prompt fields (optional)
    domain: Optional[str] = None
    custom_terms: Optional[list] = None
    initial_prompt: Optional[str] = None

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
            config: Whisper configuration with fields:
                - model_name: Whisper model (default: "large-v3-turbo")
                - language: Source language (default: "en")
                - beam_size: Beam size for decoding (default: 5)
                - sample_rate: Audio sample rate (default: 16000)
                - task: "transcribe" or "translate" (default: "transcribe")
                - target_language: Target language (default: "en")
                - enable_vad: Enable VAD (default: True)
                - domain: Optional domain name for prompts
                - custom_terms: Optional list of custom terms
                - initial_prompt: Optional initial prompt text

        Returns:
            str: Session ID

        Raises:
            RuntimeError: If not connected
        """
        if not self.connected:
            raise RuntimeError("Not connected to Whisper service")

        # Create session state with extracted config
        session_state = WhisperSessionState(
            session_id=session_id,
            config=config,
            # Extract Whisper parameters from config
            model_name=config.get('model_name', 'large-v3-turbo'),
            language=config.get('language', 'en'),
            beam_size=config.get('beam_size', 5),
            sample_rate=config.get('sample_rate', 16000),
            task=config.get('task', 'transcribe'),
            target_language=config.get('target_language', 'en'),
            enable_vad=config.get('enable_vad', True),
            # Domain prompt fields (optional)
            domain=config.get('domain'),
            custom_terms=config.get('custom_terms'),
            initial_prompt=config.get('initial_prompt')
        )
        self.sessions[session_id] = session_state

        # Send join_session event to Whisper
        logger.debug(f"📤 Sending join_session to Whisper with config: {config}")
        await self.sio.emit('join_session', {
            'session_id': session_id,
            'config': config
        })

        logger.info(f"🎬 Started stream for session: {session_id}")
        logger.info(f"   Model: {session_state.model_name}, Language: {session_state.language}")
        logger.info(f"   Task: {session_state.task}, Target: {session_state.target_language}")
        if session_state.domain or session_state.custom_terms or session_state.initial_prompt:
            logger.info(f"   Domain prompts: domain={session_state.domain}, "
                       f"terms={len(session_state.custom_terms) if session_state.custom_terms else 0}, "
                       f"prompt={'yes' if session_state.initial_prompt else 'no'}")

        return session_id

    async def send_audio_chunk(
        self,
        session_id: str,
        audio_data: bytes,
        timestamp: Optional[str] = None
    ):
        """
        Send audio chunk to Whisper for processing with full configuration

        Args:
            session_id: Session identifier
            audio_data: Raw audio data (int16 bytes)
            timestamp: Chunk timestamp (optional, not used by Whisper)

        Raises:
            RuntimeError: If not connected or session not found
        """
        if not self.connected:
            raise RuntimeError("Not connected to Whisper service")

        if session_id not in self.sessions:
            raise RuntimeError(f"Session not found: {session_id}")

        # Get session state with configuration
        session = self.sessions[session_id]

        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        # Build complete transcribe_stream request matching whisper service expectations
        request_data = {
            'session_id': session_id,
            'audio_data': audio_base64,
            # REQUIRED Whisper parameters (missing these causes empty results!)
            'model_name': session.model_name,
            'language': session.language,
            'beam_size': session.beam_size,
            'sample_rate': session.sample_rate,
            'task': session.task,
            'target_language': session.target_language,
            'enable_vad': session.enable_vad,
        }

        # Add domain prompt fields if present (only on first chunk or when explicitly set)
        if session.chunks_sent == 0:  # First chunk - include domain prompts
            if session.domain:
                request_data['domain'] = session.domain
            if session.custom_terms:
                request_data['custom_terms'] = session.custom_terms
            if session.initial_prompt:
                request_data['initial_prompt'] = session.initial_prompt

        # Send transcribe_stream event with complete configuration
        await self.sio.emit('transcribe_stream', request_data)

        # Update session stats
        session.chunks_sent += 1
        session.update_activity()

        logger.debug(f"🎵 Sent audio chunk {session.chunks_sent} for session {session_id} "
                    f"({len(audio_data)} bytes, model={session.model_name})")

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
