#!/usr/bin/env python3
"""
Fireflies Router Tests - ACTUAL TESTS (No Mocks)

Tests the router, session manager, and API models.
"""

import sys
from pathlib import Path
from datetime import datetime
import pytest

# Add src to path
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(src_path))

from routers.fireflies import (
    FirefliesSessionManager,
    ConnectRequest,
    SessionResponse,
    DisconnectRequest,
    GetMeetingsRequest,
)

from models.fireflies import (
    FirefliesSession,
    FirefliesConnectionStatus,
)


class TestFirefliesSessionManager:
    """Test FirefliesSessionManager - in-memory session management."""

    def test_manager_initialization(self):
        """Test manager starts empty."""
        manager = FirefliesSessionManager()

        assert manager._sessions == {}
        assert manager._clients == {}
        assert manager.get_all_sessions() == []

    def test_get_session_none(self):
        """Test get_session returns None for unknown session."""
        manager = FirefliesSessionManager()
        assert manager.get_session("unknown") is None

    def test_get_client_none(self):
        """Test get_client returns None for unknown session."""
        manager = FirefliesSessionManager()
        assert manager.get_client("unknown") is None

    def test_session_storage(self):
        """Test storing and retrieving sessions."""
        manager = FirefliesSessionManager()

        # Manually add a session (simulating create_session result)
        session = FirefliesSession(
            session_id="s1",
            fireflies_transcript_id="t1",
            connection_status=FirefliesConnectionStatus.CONNECTED,
        )
        manager._sessions["s1"] = session

        # Retrieve
        retrieved = manager.get_session("s1")
        assert retrieved is not None
        assert retrieved.session_id == "s1"
        assert retrieved.fireflies_transcript_id == "t1"

    def test_get_all_sessions(self):
        """Test getting all sessions."""
        manager = FirefliesSessionManager()

        manager._sessions["s1"] = FirefliesSession(
            session_id="s1", fireflies_transcript_id="t1"
        )
        manager._sessions["s2"] = FirefliesSession(
            session_id="s2", fireflies_transcript_id="t2"
        )

        sessions = manager.get_all_sessions()
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_disconnect_unknown_session(self):
        """Test disconnecting unknown session returns False."""
        manager = FirefliesSessionManager()
        result = await manager.disconnect_session("unknown")
        assert result is False


class TestConnectRequest:
    """Test ConnectRequest model."""

    def test_minimal_request(self):
        """Test request with minimal fields."""
        request = ConnectRequest(transcript_id="t1")

        assert request.transcript_id == "t1"
        assert request.api_key is None
        assert request.target_languages is None

    def test_full_request(self):
        """Test request with all fields."""
        request = ConnectRequest(
            api_key="key",
            transcript_id="t1",
            target_languages=["es", "fr"],
            glossary_id="g1",
            domain="medical",
            pause_threshold_ms=1000.0,
            max_buffer_words=50,
            context_window_size=5,
        )

        assert request.api_key == "key"
        assert request.transcript_id == "t1"
        assert request.target_languages == ["es", "fr"]
        assert request.glossary_id == "g1"
        assert request.domain == "medical"
        assert request.pause_threshold_ms == 1000.0


class TestSessionResponse:
    """Test SessionResponse model."""

    def test_response_creation(self):
        """Test creating session response."""
        response = SessionResponse(
            session_id="s1",
            transcript_id="t1",
            connection_status="connected",
            chunks_received=10,
            sentences_produced=3,
            translations_completed=9,
            speakers_detected=["Alice", "Bob"],
            connected_at=datetime.utcnow(),
            error_count=0,
            last_error=None,
        )

        assert response.session_id == "s1"
        assert response.chunks_received == 10
        assert len(response.speakers_detected) == 2


class TestDisconnectRequest:
    """Test DisconnectRequest model."""

    def test_request_creation(self):
        """Test creating disconnect request."""
        request = DisconnectRequest(session_id="s1")
        assert request.session_id == "s1"


class TestGetMeetingsRequest:
    """Test GetMeetingsRequest model."""

    def test_empty_request(self):
        """Test request with no filters."""
        request = GetMeetingsRequest()
        assert request.api_key is None
        assert request.email is None

    def test_with_filters(self):
        """Test request with filters."""
        request = GetMeetingsRequest(api_key="key", email="user@example.com")
        assert request.api_key == "key"
        assert request.email == "user@example.com"


class TestSessionManagerWithSessions:
    """Test session manager with actual session data."""

    def test_session_statistics(self):
        """Test session statistics tracking."""
        manager = FirefliesSessionManager()

        session = FirefliesSession(
            session_id="s1",
            fireflies_transcript_id="t1",
            connection_status=FirefliesConnectionStatus.CONNECTED,
            chunks_received=50,
            sentences_produced=15,
            translations_completed=45,
            speakers_detected=["Alice", "Bob", "Charlie"],
        )
        manager._sessions["s1"] = session

        retrieved = manager.get_session("s1")
        assert retrieved.chunks_received == 50
        assert retrieved.sentences_produced == 15
        assert retrieved.translations_completed == 45
        assert len(retrieved.speakers_detected) == 3

    def test_multiple_sessions_isolation(self):
        """Test sessions are isolated from each other."""
        manager = FirefliesSessionManager()

        session1 = FirefliesSession(
            session_id="s1",
            fireflies_transcript_id="t1",
            chunks_received=10,
        )
        session2 = FirefliesSession(
            session_id="s2",
            fireflies_transcript_id="t2",
            chunks_received=20,
        )

        manager._sessions["s1"] = session1
        manager._sessions["s2"] = session2

        # Modify session1
        manager._sessions["s1"].chunks_received = 15

        # session2 should be unaffected
        assert manager._sessions["s2"].chunks_received == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
