"""
Session Event Writer

Provides audit trail logging for session events.
Events are stored in the session_events table for debugging and analytics.

Usage:
    writer = SessionEventWriter(db_manager)

    # Log pipeline events
    await writer.log_event(
        session_id="ff_session_abc123",
        event_type="pipeline",
        event_name="chunk_received",
        event_data={"chunk_id": "chunk_001", "text_length": 45},
    )

    # Log errors
    await writer.log_error(
        session_id="ff_session_abc123",
        error_name="translation_failed",
        error_message="Service timeout",
        error_data={"service": "ollama", "timeout_ms": 5000},
    )
"""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from livetranslate_common.logging import get_logger

logger = get_logger()


# =============================================================================
# Event Type Definitions
# =============================================================================


class EventType:
    """Valid event types for session events."""

    PIPELINE = "pipeline"  # Normal processing events
    ERROR = "error"  # Error events
    MILESTONE = "milestone"  # Major pipeline milestones
    METRIC = "metric"  # Performance metrics


class EventSeverity:
    """Severity levels for events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PipelineEvents:
    """Standard pipeline event names."""

    CHUNK_RECEIVED = "chunk_received"
    SENTENCE_AGGREGATED = "sentence_aggregated"
    TRANSCRIPT_STORED = "transcript_stored"
    TRANSLATION_STARTED = "translation_started"
    TRANSLATION_COMPLETE = "translation_complete"
    TRANSLATION_STORED = "translation_stored"
    CAPTION_BROADCAST = "caption_broadcast"
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    GLOSSARY_APPLIED = "glossary_applied"


class ErrorEvents:
    """Standard error event names."""

    VALIDATION_FAILED = "validation_failed"
    TRANSLATION_FAILED = "translation_failed"
    STORAGE_FAILED = "storage_failed"
    CONNECTION_FAILED = "connection_failed"
    TIMEOUT = "timeout"


# =============================================================================
# Session Event Data Classes
# =============================================================================


@dataclass
class SessionEvent:
    """Represents a session event to be logged."""

    event_id: str
    session_id: str
    event_type: str
    event_name: str
    timestamp: datetime
    severity: str = EventSeverity.INFO
    source: str = "pipeline_coordinator"
    event_data: dict[str, Any] | None = None
    trace_id: str | None = None
    span_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "event_type": self.event_type,
            "event_name": self.event_name,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "source": self.source,
            "event_data": self.event_data or {},
            "trace_id": self.trace_id,
            "span_id": self.span_id,
        }


# =============================================================================
# Session Event Writer
# =============================================================================


class SessionEventWriter:
    """
    Write audit events to session_events table.

    Provides methods for logging various types of session events
    with consistent formatting and error handling.
    """

    def __init__(self, db_manager=None):
        """
        Initialize the event writer.

        Args:
            db_manager: Database manager with db_pool for asyncpg connections.
                        If None, events will be logged but not persisted.
        """
        self.db_manager = db_manager
        self._buffer: list[SessionEvent] = []
        self._buffer_size = 10  # Flush after this many events

    async def log_event(
        self,
        session_id: str,
        event_type: str,
        event_name: str,
        event_data: dict[str, Any] | None = None,
        severity: str = EventSeverity.INFO,
        source: str = "pipeline_coordinator",
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> str | None:
        """
        Log a session event.

        Args:
            session_id: The session this event belongs to
            event_type: Type of event (pipeline, error, milestone, metric)
            event_name: Specific event name (e.g., "chunk_received")
            event_data: Additional event data as dictionary
            severity: Event severity level
            source: Source component that generated the event
            trace_id: Optional trace ID for correlation
            span_id: Optional span ID for correlation

        Returns:
            Event ID if persisted, None otherwise
        """
        event_id = str(uuid.uuid4())

        event = SessionEvent(
            event_id=event_id,
            session_id=session_id,
            event_type=event_type,
            event_name=event_name,
            timestamp=datetime.now(UTC),
            severity=severity,
            source=source,
            event_data=event_data,
            trace_id=trace_id,
            span_id=span_id,
        )

        # Log locally
        log_level = {
            EventSeverity.DEBUG: logging.DEBUG,
            EventSeverity.INFO: logging.INFO,
            EventSeverity.WARNING: logging.WARNING,
            EventSeverity.ERROR: logging.ERROR,
            EventSeverity.CRITICAL: logging.CRITICAL,
        }.get(severity, logging.INFO)

        trace_prefix = f"[trace:{trace_id[:8]}] " if trace_id else ""
        logger.log(
            log_level,
            f"{trace_prefix}[{session_id}] {event_type}.{event_name}: {json.dumps(event_data or {})}",
        )

        # Persist to database if available
        if self.db_manager and hasattr(self.db_manager, "db_pool") and self.db_manager.db_pool:
            try:
                await self._persist_event(event)
                return event_id
            except Exception as e:
                logger.error(f"Failed to persist event: {e}")
                return None
        else:
            # Buffer for later persistence
            self._buffer.append(event)
            if len(self._buffer) >= self._buffer_size:
                logger.debug(f"Event buffer at {len(self._buffer)} events (no DB connected)")

        return event_id

    async def _persist_event(self, event: SessionEvent) -> None:
        """Persist a single event to the database."""
        async with self.db_manager.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO session_events (
                    event_id, session_id, event_type, event_name,
                    event_data, timestamp, severity, source,
                    trace_id, span_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                event.event_id,
                event.session_id,
                event.event_type,
                event.event_name,
                json.dumps(event.event_data or {}),
                event.timestamp,
                event.severity,
                event.source,
                event.trace_id,
                event.span_id,
            )

    async def log_pipeline_event(
        self,
        session_id: str,
        event_name: str,
        event_data: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> str | None:
        """
        Log a pipeline event (convenience method).

        Args:
            session_id: The session ID
            event_name: Name from PipelineEvents
            event_data: Additional data
            trace_id: Optional trace ID
        """
        return await self.log_event(
            session_id=session_id,
            event_type=EventType.PIPELINE,
            event_name=event_name,
            event_data=event_data,
            severity=EventSeverity.INFO,
            trace_id=trace_id,
        )

    async def log_error(
        self,
        session_id: str,
        error_name: str,
        error_message: str,
        error_data: dict[str, Any] | None = None,
        severity: str = EventSeverity.ERROR,
        trace_id: str | None = None,
    ) -> str | None:
        """
        Log an error event.

        Args:
            session_id: The session ID
            error_name: Name from ErrorEvents
            error_message: Human-readable error message
            error_data: Additional error context
            severity: Severity level (default: error)
            trace_id: Optional trace ID
        """
        data = error_data or {}
        data["error_message"] = error_message

        return await self.log_event(
            session_id=session_id,
            event_type=EventType.ERROR,
            event_name=error_name,
            event_data=data,
            severity=severity,
            source="error_handler",
            trace_id=trace_id,
        )

    async def log_milestone(
        self,
        session_id: str,
        milestone_name: str,
        milestone_data: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> str | None:
        """
        Log a milestone event (major pipeline events).

        Args:
            session_id: The session ID
            milestone_name: Name of the milestone
            milestone_data: Additional data
            trace_id: Optional trace ID
        """
        return await self.log_event(
            session_id=session_id,
            event_type=EventType.MILESTONE,
            event_name=milestone_name,
            event_data=milestone_data,
            severity=EventSeverity.INFO,
            source="milestone_tracker",
            trace_id=trace_id,
        )

    async def log_metric(
        self,
        session_id: str,
        metric_name: str,
        metric_value: float,
        metric_unit: str = "",
        additional_data: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> str | None:
        """
        Log a performance metric.

        Args:
            session_id: The session ID
            metric_name: Name of the metric (e.g., "translation_latency")
            metric_value: Numeric value
            metric_unit: Unit of measurement (e.g., "ms", "bytes")
            additional_data: Additional context
            trace_id: Optional trace ID
        """
        data = additional_data or {}
        data["value"] = metric_value
        data["unit"] = metric_unit

        return await self.log_event(
            session_id=session_id,
            event_type=EventType.METRIC,
            event_name=metric_name,
            event_data=data,
            severity=EventSeverity.DEBUG,
            source="metrics",
            trace_id=trace_id,
        )

    async def flush_buffer(self) -> int:
        """
        Flush buffered events to database.

        Returns:
            Number of events flushed
        """
        if not self._buffer:
            return 0

        if not self.db_manager or not self.db_manager.db_pool:
            logger.warning(f"Cannot flush {len(self._buffer)} events - no database connection")
            return 0

        count = 0
        for event in self._buffer[:]:
            try:
                await self._persist_event(event)
                self._buffer.remove(event)
                count += 1
            except Exception as e:
                logger.error(f"Failed to flush event {event.event_id}: {e}")

        logger.info(f"Flushed {count} buffered events to database")
        return count

    def get_buffer_size(self) -> int:
        """Get current number of buffered events."""
        return len(self._buffer)


# =============================================================================
# Query Functions
# =============================================================================


async def get_session_events(
    db_manager,
    session_id: str,
    event_type: str | None = None,
    severity: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    Query events for a session.

    Args:
        db_manager: Database manager
        session_id: Session to query
        event_type: Optional filter by event type
        severity: Optional filter by minimum severity
        limit: Maximum events to return
        offset: Pagination offset

    Returns:
        List of event dictionaries
    """
    if not db_manager or not db_manager.db_pool:
        return []

    severity_order = ["debug", "info", "warning", "error", "critical"]

    async with db_manager.db_pool.acquire() as conn:
        query = """
            SELECT event_id, session_id, event_type, event_name,
                   event_data, timestamp, severity, source,
                   trace_id, span_id
            FROM session_events
            WHERE session_id = $1
        """
        params = [session_id]
        param_idx = 2

        if event_type:
            query += f" AND event_type = ${param_idx}"
            params.append(event_type)
            param_idx += 1

        if severity:
            min_level = severity_order.index(severity.lower())
            valid_severities = severity_order[min_level:]
            placeholders = ", ".join(
                f"${i}" for i in range(param_idx, param_idx + len(valid_severities))
            )
            query += f" AND severity IN ({placeholders})"
            params.extend(valid_severities)
            param_idx += len(valid_severities)

        query += f" ORDER BY timestamp DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
        params.extend([limit, offset])

        rows = await conn.fetch(query, *params)

        return [
            {
                "event_id": row["event_id"],
                "session_id": row["session_id"],
                "event_type": row["event_type"],
                "event_name": row["event_name"],
                "event_data": json.loads(row["event_data"]) if row["event_data"] else {},
                "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                "severity": row["severity"],
                "source": row["source"],
                "trace_id": row["trace_id"],
                "span_id": row["span_id"],
            }
            for row in rows
        ]


async def get_trace_events(
    db_manager,
    trace_id: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Get all events for a specific trace.

    Args:
        db_manager: Database manager
        trace_id: Trace ID to query
        limit: Maximum events to return

    Returns:
        List of event dictionaries ordered by timestamp
    """
    if not db_manager or not db_manager.db_pool:
        return []

    async with db_manager.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT event_id, session_id, event_type, event_name,
                   event_data, timestamp, severity, source,
                   trace_id, span_id
            FROM session_events
            WHERE trace_id = $1
            ORDER BY timestamp ASC
            LIMIT $2
            """,
            trace_id,
            limit,
        )

        return [
            {
                "event_id": row["event_id"],
                "session_id": row["session_id"],
                "event_type": row["event_type"],
                "event_name": row["event_name"],
                "event_data": json.loads(row["event_data"]) if row["event_data"] else {},
                "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                "severity": row["severity"],
                "source": row["source"],
                "trace_id": row["trace_id"],
                "span_id": row["span_id"],
            }
            for row in rows
        ]
