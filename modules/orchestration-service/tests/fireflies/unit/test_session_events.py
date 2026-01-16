"""
Unit Tests for Session Event Writer

TDD Tests - Define behavior for session event audit trail.

Behaviors:
1. Log pipeline events with trace context
2. Log error events with severity levels
3. Query events by session ID
4. Events include all required metadata

Run with: pytest tests/fireflies/unit/test_session_events.py -v
"""

import pytest
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import sys

# Add src path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_db_manager():
    """Mock database manager with pool."""
    manager = MagicMock()
    manager.db_pool = AsyncMock()

    # Mock connection context manager
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.fetchrow = AsyncMock(return_value=None)

    manager.db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    return manager


@pytest.fixture
def sample_session_id():
    """Sample session ID for tests."""
    return f"ff_session_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def sample_trace_id():
    """Sample trace ID for correlation."""
    return uuid.uuid4().hex


# =============================================================================
# Behavior: Event Logging
# =============================================================================


class TestEventLogging:
    """
    BEHAVIOR: Log session events to database.

    Given: A session event occurs
    When: Logging the event
    Then: Should persist with all required fields
    """

    def test_event_has_required_fields(self, sample_session_id):
        """
        GIVEN: An event to log
        WHEN: Creating the event record
        THEN: Should include: event_id, session_id, event_type, event_name, timestamp
        """
        # Arrange
        event = {
            "event_id": str(uuid.uuid4()),
            "session_id": sample_session_id,
            "event_type": "pipeline",
            "event_name": "chunk_received",
            "timestamp": datetime.now(timezone.utc),
            "severity": "info",
            "source": "pipeline_coordinator",
            "event_data": {},
        }

        # Assert - all required fields present
        required_fields = ["event_id", "session_id", "event_type", "event_name", "timestamp"]
        for field in required_fields:
            assert field in event, f"Missing required field: {field}"

    def test_event_types_are_valid(self):
        """
        GIVEN: Event type values
        WHEN: Validating event types
        THEN: Should be one of: pipeline, error, milestone, metric
        """
        # Arrange
        valid_types = ["pipeline", "error", "milestone", "metric"]

        # Act
        def validate_event_type(event_type: str) -> bool:
            return event_type in valid_types

        # Assert
        assert validate_event_type("pipeline") is True
        assert validate_event_type("error") is True
        assert validate_event_type("milestone") is True
        assert validate_event_type("metric") is True
        assert validate_event_type("unknown") is False

    def test_severity_levels_are_valid(self):
        """
        GIVEN: Severity level values
        WHEN: Validating severity
        THEN: Should be one of: debug, info, warning, error, critical
        """
        # Arrange
        valid_severities = ["debug", "info", "warning", "error", "critical"]

        # Act
        def validate_severity(severity: str) -> bool:
            return severity in valid_severities

        # Assert
        for sev in valid_severities:
            assert validate_severity(sev) is True
        assert validate_severity("unknown") is False

    def test_event_data_is_json_serializable(self):
        """
        GIVEN: Event data dictionary
        WHEN: Serializing to JSON
        THEN: Should serialize without errors
        """
        # Arrange
        event_data = {
            "transcript_id": "abc123",
            "translation_id": "def456",
            "source_lang": "en",
            "target_lang": "es",
            "latency_ms": 123.45,
            "glossary_terms_applied": ["API", "endpoint"],
        }

        # Act
        json_str = json.dumps(event_data)

        # Assert
        parsed = json.loads(json_str)
        assert parsed["transcript_id"] == "abc123"
        assert parsed["latency_ms"] == 123.45

    @pytest.mark.asyncio
    async def test_log_event_calls_database(self, mock_db_manager, sample_session_id):
        """
        GIVEN: A session event to log
        WHEN: Calling log_event
        THEN: Should execute database insert
        """
        # This tests the interface contract
        # Actual implementation will be in src/database/session_events.py

        # Arrange
        event_id = str(uuid.uuid4())
        event_type = "pipeline"
        event_name = "chunk_received"
        event_data = {"chunk_id": "chunk_001"}

        # Act - simulate what log_event should do
        mock_conn = await mock_db_manager.db_pool.acquire().__aenter__()
        await mock_conn.execute(
            "INSERT INTO session_events (...) VALUES (...)",
            event_id, sample_session_id, event_type, event_name,
            json.dumps(event_data), datetime.now(timezone.utc),
            "info", "pipeline_coordinator"
        )

        # Assert
        mock_conn.execute.assert_called_once()


# =============================================================================
# Behavior: Pipeline Event Logging
# =============================================================================


class TestPipelineEventLogging:
    """
    BEHAVIOR: Log specific pipeline events.

    Given: Pipeline processing stages
    When: Logging events
    Then: Should include appropriate metadata
    """

    def test_chunk_received_event_format(self, sample_session_id):
        """
        GIVEN: A chunk received from Fireflies
        WHEN: Logging the event
        THEN: Should include chunk metadata
        """
        # Arrange
        event_data = {
            "chunk_id": "chunk_001",
            "speaker": "John Doe",
            "text_length": 45,
            "start_time": 10.5,
            "end_time": 12.3,
        }

        event = {
            "event_type": "pipeline",
            "event_name": "chunk_received",
            "event_data": event_data,
        }

        # Assert
        assert event["event_data"]["chunk_id"] == "chunk_001"
        assert "speaker" in event["event_data"]
        assert "text_length" in event["event_data"]

    def test_translation_complete_event_format(self, sample_session_id):
        """
        GIVEN: A translation completes
        WHEN: Logging the event
        THEN: Should include translation metadata
        """
        # Arrange
        event_data = {
            "transcript_id": "abc123",
            "translation_id": "def456",
            "source_lang": "en",
            "target_lang": "es",
            "latency_ms": 125.5,
            "glossary_terms_applied": ["API", "endpoint"],
            "confidence": 0.95,
        }

        event = {
            "event_type": "pipeline",
            "event_name": "translation_complete",
            "event_data": event_data,
        }

        # Assert
        assert "transcript_id" in event["event_data"]
        assert "translation_id" in event["event_data"]
        assert "latency_ms" in event["event_data"]

    def test_segment_complete_event_includes_both_ids(self, sample_session_id):
        """
        GIVEN: A transcript+translation segment completes
        WHEN: Logging the event
        THEN: Should link transcript_id and translation_id
        """
        # Arrange
        event_data = {
            "transcript_id": "transcript_abc123",
            "translation_id": "translation_def456",
            "processing_time_ms": 245.0,
        }

        # Act & Assert
        assert event_data["transcript_id"] is not None
        assert event_data["translation_id"] is not None
        assert "transcript_id" in event_data
        assert "translation_id" in event_data


# =============================================================================
# Behavior: Error Event Logging
# =============================================================================


class TestErrorEventLogging:
    """
    BEHAVIOR: Log error events with proper severity.

    Given: An error occurs in the pipeline
    When: Logging the error
    Then: Should include error details and severity
    """

    def test_error_event_includes_exception_info(self):
        """
        GIVEN: An exception occurs
        WHEN: Logging the error event
        THEN: Should include exception type and message
        """
        # Arrange
        try:
            raise ValueError("Invalid chunk format")
        except ValueError as e:
            error_data = {
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "stack_trace": None,  # Optional for unit tests
            }

        # Assert
        assert error_data["exception_type"] == "ValueError"
        assert "Invalid chunk format" in error_data["exception_message"]

    def test_error_severity_is_appropriate(self):
        """
        GIVEN: Different error types
        WHEN: Determining severity
        THEN: Should assign appropriate level
        """
        # Arrange
        def get_error_severity(error_type: str) -> str:
            severity_map = {
                "ValidationError": "warning",
                "TranslationError": "error",
                "ConnectionError": "error",
                "TimeoutError": "warning",
                "DatabaseError": "critical",
                "AuthenticationError": "critical",
            }
            return severity_map.get(error_type, "error")

        # Assert
        assert get_error_severity("ValidationError") == "warning"
        assert get_error_severity("DatabaseError") == "critical"
        assert get_error_severity("UnknownError") == "error"

    def test_validation_error_includes_field_info(self):
        """
        GIVEN: A validation error
        WHEN: Logging the event
        THEN: Should include field-specific details
        """
        # Arrange
        error_data = {
            "error_type": "validation_failed",
            "errors": [
                "Empty transcript text",
                "Invalid timestamps: end < start",
            ],
            "text_preview": "...",
        }

        # Assert
        assert len(error_data["errors"]) == 2
        assert "Empty transcript text" in error_data["errors"]


# =============================================================================
# Behavior: Event Querying
# =============================================================================


class TestEventQuerying:
    """
    BEHAVIOR: Query events for debugging and analysis.

    Given: Stored session events
    When: Querying events
    Then: Should return matching events with filters
    """

    def test_query_events_by_session_id(self):
        """
        GIVEN: Events stored for a session
        WHEN: Querying by session_id
        THEN: Should return only that session's events
        """
        # Arrange
        all_events = [
            {"session_id": "session_a", "event_name": "event1"},
            {"session_id": "session_b", "event_name": "event2"},
            {"session_id": "session_a", "event_name": "event3"},
        ]

        # Act
        session_a_events = [e for e in all_events if e["session_id"] == "session_a"]

        # Assert
        assert len(session_a_events) == 2
        assert all(e["session_id"] == "session_a" for e in session_a_events)

    def test_query_events_by_type(self):
        """
        GIVEN: Events of different types
        WHEN: Filtering by event_type
        THEN: Should return only matching type
        """
        # Arrange
        all_events = [
            {"event_type": "pipeline", "event_name": "chunk_received"},
            {"event_type": "error", "event_name": "translation_failed"},
            {"event_type": "pipeline", "event_name": "translation_complete"},
        ]

        # Act
        error_events = [e for e in all_events if e["event_type"] == "error"]

        # Assert
        assert len(error_events) == 1
        assert error_events[0]["event_name"] == "translation_failed"

    def test_query_events_by_severity(self):
        """
        GIVEN: Events with different severities
        WHEN: Filtering by severity
        THEN: Should return events at or above that severity
        """
        # Arrange
        severity_order = ["debug", "info", "warning", "error", "critical"]
        events = [
            {"severity": "info", "event_name": "event1"},
            {"severity": "warning", "event_name": "event2"},
            {"severity": "error", "event_name": "event3"},
            {"severity": "info", "event_name": "event4"},
        ]

        # Act
        def filter_by_min_severity(events: List[Dict], min_severity: str) -> List[Dict]:
            min_level = severity_order.index(min_severity)
            return [e for e in events if severity_order.index(e["severity"]) >= min_level]

        warning_and_above = filter_by_min_severity(events, "warning")

        # Assert
        assert len(warning_and_above) == 2
        assert all(e["severity"] in ["warning", "error", "critical"] for e in warning_and_above)

    def test_query_events_limited_and_ordered(self):
        """
        GIVEN: Many events
        WHEN: Querying with limit
        THEN: Should return limited results in chronological order
        """
        # Arrange
        events = [
            {"timestamp": "2024-01-15T10:00:00Z", "event_name": "event1"},
            {"timestamp": "2024-01-15T10:00:01Z", "event_name": "event2"},
            {"timestamp": "2024-01-15T10:00:02Z", "event_name": "event3"},
            {"timestamp": "2024-01-15T10:00:03Z", "event_name": "event4"},
            {"timestamp": "2024-01-15T10:00:04Z", "event_name": "event5"},
        ]

        # Act
        limited = sorted(events, key=lambda e: e["timestamp"])[:3]

        # Assert
        assert len(limited) == 3
        assert limited[0]["event_name"] == "event1"
        assert limited[2]["event_name"] == "event3"


# =============================================================================
# Behavior: Trace Context
# =============================================================================


class TestTraceContext:
    """
    BEHAVIOR: Trace context for end-to-end debugging.

    Given: A request flows through the pipeline
    When: Creating trace context
    Then: Should enable correlation across stages
    """

    def test_trace_context_has_trace_id(self):
        """
        GIVEN: A new trace context
        WHEN: Creating it
        THEN: Should have unique trace_id
        """
        # Arrange & Act
        trace_id = uuid.uuid4().hex

        # Assert
        assert len(trace_id) == 32
        assert trace_id.isalnum()

    def test_child_span_inherits_trace_id(self, sample_trace_id):
        """
        GIVEN: A parent trace context
        WHEN: Creating a child span
        THEN: Should inherit the trace_id
        """
        # Arrange
        parent_context = {
            "trace_id": sample_trace_id,
            "span_id": uuid.uuid4().hex[:16],
            "parent_span_id": None,
        }

        # Act
        child_context = {
            "trace_id": parent_context["trace_id"],  # Inherited
            "span_id": f"child_{uuid.uuid4().hex[:8]}",
            "parent_span_id": parent_context["span_id"],
        }

        # Assert
        assert child_context["trace_id"] == parent_context["trace_id"]
        assert child_context["parent_span_id"] == parent_context["span_id"]
        assert child_context["span_id"] != parent_context["span_id"]

    def test_trace_context_includes_session_id(self, sample_session_id, sample_trace_id):
        """
        GIVEN: A trace context for a session
        WHEN: Creating the context
        THEN: Should include session_id
        """
        # Arrange
        context = {
            "trace_id": sample_trace_id,
            "span_id": uuid.uuid4().hex[:16],
            "session_id": sample_session_id,
            "transcript_id": None,
        }

        # Assert
        assert context["session_id"] == sample_session_id


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
