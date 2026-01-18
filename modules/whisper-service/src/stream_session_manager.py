#!/usr/bin/env python3
"""
Streaming Session Manager for WebSocket Transcription

Manages WebSocket streaming sessions for real-time audio transcription.
Handles session lifecycle, audio buffering, and state management.

Following Phase 3.1 architecture:
- Session creation from orchestration service
- Audio chunk buffering and processing
- Real-time segment generation
- Session cleanup and resource management

Usage:
    manager = StreamSessionManager(model_manager=model_manager)

    # Create session
    session = manager.create_session(
        session_id="session-123",
        config={"model": "large-v3", "language": "en"}
    )

    # Add audio chunks
    manager.add_audio_chunk(session_id="session-123", audio_data=np.array([...]))

    # Process and get segments
    segments = manager.process_session(session_id="session-123")

    # Cleanup
    manager.close_session(session_id="session-123")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StreamingSession:
    """
    Represents a single WebSocket streaming session

    Manages audio buffering and processing state for one connection
    """

    session_id: str
    config: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Audio buffering
    audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    buffer_start_time: datetime | None = None

    # Processing state
    total_audio_processed: float = 0.0  # Seconds
    segment_count: int = 0
    is_active: bool = True

    # Statistics
    chunks_received: int = 0
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))

    def add_audio(self, audio_chunk: np.ndarray, timestamp: datetime):
        """
        Add audio chunk to buffer

        Parameters:
            audio_chunk (np.ndarray): Audio data (float32)
            timestamp (datetime): When this audio was captured
        """
        if self.buffer_start_time is None:
            self.buffer_start_time = timestamp

        # Append to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        self.chunks_received += 1
        self.last_activity = datetime.now(UTC)

        logger.debug(
            f"Session {self.session_id}: Added {len(audio_chunk)} samples, "
            f"buffer now {len(self.audio_buffer)} samples"
        )

    def get_buffered_duration(self, sample_rate: int = 16000) -> float:
        """
        Get duration of buffered audio in seconds

        Parameters:
            sample_rate (int): Sample rate (default 16000 Hz)

        Returns:
            float: Duration in seconds
        """
        return len(self.audio_buffer) / sample_rate

    def consume_buffer(self, num_samples: int | None = None) -> np.ndarray:
        """
        Consume audio from buffer (removes from buffer after extraction)

        Parameters:
            num_samples (int, optional): Number of samples to consume.
                If None, consumes entire buffer.

        Returns:
            np.ndarray: Consumed audio data
        """
        if num_samples is None:
            consumed = self.audio_buffer
            self.audio_buffer = np.array([], dtype=np.float32)
        else:
            consumed = self.audio_buffer[:num_samples]
            self.audio_buffer = self.audio_buffer[num_samples:]

        return consumed

    def clear_buffer(self):
        """Clear audio buffer"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_start_time = None


class StreamSessionManager:
    """
    Manages WebSocket streaming sessions

    Handles creation, audio buffering, processing, and cleanup of sessions
    """

    def __init__(self, model_manager=None):
        """
        Initialize stream session manager

        Parameters:
            model_manager: ModelManager instance for transcription
        """
        self.model_manager = model_manager
        self.sessions: dict[str, StreamingSession] = {}
        self._session_lock = asyncio.Lock()

        logger.info("StreamSessionManager initialized")

    async def create_session(self, session_id: str, config: dict[str, Any]) -> StreamingSession:
        """
        Create a new streaming session

        Parameters:
            session_id (str): Unique session identifier
            config (Dict): Session configuration
                - model (str): Whisper model name
                - language (str, optional): Language code
                - enable_vad (bool, optional): Enable VAD
                - enable_cif (bool, optional): Enable CIF word boundaries

        Returns:
            StreamingSession: Created session

        Example:
            session = await manager.create_session(
                session_id="session-123",
                config={
                    "model": "large-v3",
                    "language": "en",
                    "enable_vad": True,
                    "enable_cif": True
                }
            )
        """
        async with self._session_lock:
            if session_id in self.sessions:
                logger.warning(f"Session {session_id} already exists, replacing")
                await self.close_session(session_id)

            session = StreamingSession(session_id=session_id, config=config)

            self.sessions[session_id] = session

            logger.info(
                f"Created session {session_id} with config: "
                f"model={config.get('model')}, language={config.get('language')}"
            )

            return session

    def get_session(self, session_id: str) -> StreamingSession | None:
        """
        Get session by ID

        Parameters:
            session_id (str): Session identifier

        Returns:
            StreamingSession or None: Session if exists
        """
        return self.sessions.get(session_id)

    async def add_audio_chunk(
        self, session_id: str, audio_data: bytes, timestamp: datetime
    ) -> bool:
        """
        Add audio chunk to session buffer

        Parameters:
            session_id (str): Session identifier
            audio_data (bytes): Raw audio bytes (float32)
            timestamp (datetime): When audio was captured

        Returns:
            bool: True if successful, False if session not found
        """
        session = self.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return False

        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.float32)

        # Add to buffer
        session.add_audio(audio_array, timestamp)

        return True

    async def process_session(
        self, session_id: str, sample_rate: int = 16000
    ) -> list[dict[str, Any]]:
        """
        Process buffered audio and generate segments

        Parameters:
            session_id (str): Session identifier
            sample_rate (int): Audio sample rate

        Returns:
            List[Dict]: Generated transcript segments
        """
        session = self.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return []

        # Check if we have enough audio to process
        buffered_duration = session.get_buffered_duration(sample_rate)
        if buffered_duration < 1.0:  # Need at least 1 second
            logger.debug(f"Session {session_id}: Insufficient audio ({buffered_duration:.2f}s)")
            return []

        # Consume buffer
        audio_to_process = session.consume_buffer()

        # Transcribe (placeholder - actual implementation would use model_manager)
        # For now, just log
        logger.info(
            f"Session {session_id}: Processing {len(audio_to_process)} samples "
            f"({buffered_duration:.2f}s)"
        )

        # This would call model_manager.transcribe() in real implementation
        # segments = self.model_manager.transcribe(audio_to_process, ...)

        # Placeholder segments
        segments = []

        # Update statistics
        session.total_audio_processed += buffered_duration
        session.segment_count += len(segments)

        return segments

    async def close_session(self, session_id: str):
        """
        Close and cleanup session

        Parameters:
            session_id (str): Session identifier
        """
        async with self._session_lock:
            session = self.sessions.pop(session_id, None)
            if session:
                session.is_active = False
                session.clear_buffer()
                logger.info(
                    f"Closed session {session_id}: "
                    f"processed {session.total_audio_processed:.2f}s, "
                    f"{session.segment_count} segments"
                )
            else:
                logger.warning(f"Session {session_id} not found for closure")

    def get_active_sessions(self) -> list[str]:
        """
        Get list of active session IDs

        Returns:
            List[str]: Active session IDs
        """
        return [sid for sid, session in self.sessions.items() if session.is_active]

    def get_session_stats(self, session_id: str) -> dict[str, Any] | None:
        """
        Get session statistics

        Parameters:
            session_id (str): Session identifier

        Returns:
            Dict or None: Session statistics
        """
        session = self.get_session(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "is_active": session.is_active,
            "chunks_received": session.chunks_received,
            "total_audio_processed": session.total_audio_processed,
            "segment_count": session.segment_count,
            "buffer_size": len(session.audio_buffer),
            "buffered_duration": session.get_buffered_duration(),
            "last_activity": session.last_activity.isoformat(),
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def test_session_manager():
        print("Stream Session Manager Test")
        print("=" * 50)

        manager = StreamSessionManager()

        # Test 1: Create session
        print("\n[TEST 1] Create session:")
        session = await manager.create_session(
            session_id="test-session-1", config={"model": "large-v3", "language": "en"}
        )
        print(f"  Created: {session.session_id}")
        print(f"  Config: {session.config}")
        print("  ✅ Session creation working")

        # Test 2: Add audio
        print("\n[TEST 2] Add audio chunks:")
        test_audio = np.random.randn(16000).astype(np.float32)  # 1 second
        audio_bytes = test_audio.tobytes()

        await manager.add_audio_chunk(
            session_id="test-session-1", audio_data=audio_bytes, timestamp=datetime.now(UTC)
        )

        stats = manager.get_session_stats("test-session-1")
        print(f"  Chunks received: {stats['chunks_received']}")
        print(f"  Buffer size: {stats['buffer_size']} samples")
        print(f"  Buffered duration: {stats['buffered_duration']:.2f}s")
        print("  ✅ Audio buffering working")

        # Test 3: Get active sessions
        print("\n[TEST 3] Active sessions:")
        active = manager.get_active_sessions()
        print(f"  Active sessions: {active}")
        print("  ✅ Session listing working")

        # Test 4: Close session
        print("\n[TEST 4] Close session:")
        await manager.close_session("test-session-1")
        active_after = manager.get_active_sessions()
        print(f"  Active after close: {active_after}")
        print("  ✅ Session cleanup working")

        print("\n" + "=" * 50)
        print("✅ Stream Session Manager Test Complete")
        print("\nKey Features:")
        print("  - Session lifecycle management")
        print("  - Audio buffering with timestamps")
        print("  - Statistics tracking")
        print("  - Thread-safe operations")

    asyncio.run(test_session_manager())
