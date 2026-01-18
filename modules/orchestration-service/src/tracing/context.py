"""
Trace Context for End-to-End Debugging

Provides trace context propagation for tracking requests through the
Fireflies → Orchestration → Translation pipeline.

Usage:
    # Create a trace for a new request
    ctx = TraceContext.create(session_id="ff_session_abc123")

    # Create child spans for operations
    with ctx.span("process_chunk") as span:
        # ... processing code ...
        span.set_attribute("chunk_id", "chunk_001")

    # Or manually create spans
    translate_span = ctx.child_span("translate")
    # ... translation code ...
    translate_span.end()
"""

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TraceSpan:
    """
    A single span within a trace.

    Represents one operation/stage in the processing pipeline.
    """

    trace_id: str
    span_id: str
    operation_name: str
    parent_span_id: str | None = None
    session_id: str | None = None
    transcript_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "in_progress"  # in_progress, success, error
    error_message: str | None = None

    @property
    def duration_ms(self) -> float | None:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def set_status(self, status: str, error_message: str | None = None) -> None:
        """Set span status."""
        self.status = status
        if error_message:
            self.error_message = error_message

    def end(self, status: str = "success") -> None:
        """End the span."""
        self.end_time = time.time()
        self.status = status
        logger.debug(
            f"[trace:{self.trace_id[:8]}] Span '{self.operation_name}' ended: "
            f"duration={self.duration_ms:.2f}ms, status={self.status}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary for logging/storage."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "operation_name": self.operation_name,
            "parent_span_id": self.parent_span_id,
            "session_id": self.session_id,
            "transcript_id": self.transcript_id,
            "start_time": datetime.fromtimestamp(self.start_time, tz=UTC).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time, tz=UTC).isoformat()
            if self.end_time
            else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "status": self.status,
            "error_message": self.error_message,
        }


@dataclass
class TraceContext:
    """
    Trace context for end-to-end request tracking.

    Maintains trace_id across all operations in a request flow.
    """

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    root_span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    session_id: str | None = None
    transcript_id: str | None = None
    current_span: TraceSpan | None = None
    spans: list[TraceSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls, session_id: str | None = None, transcript_id: str | None = None, **metadata
    ) -> "TraceContext":
        """Create a new trace context."""
        ctx = cls(
            session_id=session_id,
            transcript_id=transcript_id,
            metadata=metadata,
        )
        logger.debug(
            f"[trace:{ctx.trace_id[:8]}] Created trace context "
            f"(session={session_id}, transcript={transcript_id})"
        )
        return ctx

    @classmethod
    def from_parent(
        cls,
        parent: "TraceContext",
        session_id: str | None = None,
        transcript_id: str | None = None,
    ) -> "TraceContext":
        """Create a child trace context inheriting parent trace_id."""
        return cls(
            trace_id=parent.trace_id,
            session_id=session_id or parent.session_id,
            transcript_id=transcript_id or parent.transcript_id,
            metadata=parent.metadata.copy(),
        )

    def child_span(self, operation: str) -> TraceSpan:
        """
        Create a child span for a sub-operation.

        Args:
            operation: Name of the operation (e.g., "store_transcript", "translate")

        Returns:
            TraceSpan for tracking the operation
        """
        span = TraceSpan(
            trace_id=self.trace_id,
            span_id=f"{operation}_{uuid.uuid4().hex[:8]}",
            operation_name=operation,
            parent_span_id=self.current_span.span_id if self.current_span else self.root_span_id,
            session_id=self.session_id,
            transcript_id=self.transcript_id,
        )
        self.spans.append(span)
        logger.debug(
            f"[trace:{self.trace_id[:8]}] Started span '{operation}' "
            f"(span_id={span.span_id[:12]})"
        )
        return span

    @contextmanager
    def span(self, operation: str):
        """
        Context manager for creating and automatically ending a span.

        Usage:
            with ctx.span("process_chunk") as span:
                span.set_attribute("chunk_id", "123")
                # ... processing code ...
        """
        span = self.child_span(operation)
        previous_span = self.current_span
        self.current_span = span
        try:
            yield span
            span.end("success")
        except Exception as e:
            span.end("error")
            span.error_message = str(e)
            raise
        finally:
            self.current_span = previous_span

    def set_metadata(self, key: str, value: Any) -> None:
        """Set trace-level metadata."""
        self.metadata[key] = value

    def get_log_context(self) -> dict[str, Any]:
        """
        Get context for structured logging.

        Returns dict suitable for use as extra= in logger calls.
        """
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "transcript_id": self.transcript_id,
            "span_id": self.current_span.span_id if self.current_span else self.root_span_id,
        }

    def to_header(self) -> str:
        """Convert to header value for propagation."""
        return f"{self.trace_id};{self.root_span_id};{self.session_id or ''}"

    @classmethod
    def from_header(cls, header: str) -> Optional["TraceContext"]:
        """Parse trace context from header value."""
        try:
            parts = header.split(";")
            if len(parts) >= 2:
                return cls(
                    trace_id=parts[0],
                    root_span_id=parts[1],
                    session_id=parts[2] if len(parts) > 2 and parts[2] else None,
                )
        except Exception as e:
            logger.warning(f"Failed to parse trace header: {e}")
        return None

    def get_timing_summary(self) -> dict[str, Any]:
        """
        Get a summary of all span timings.

        Useful for performance analysis.
        """
        completed_spans = [s for s in self.spans if s.end_time]
        if not completed_spans:
            return {"total_spans": 0, "total_duration_ms": 0, "spans": []}

        return {
            "total_spans": len(completed_spans),
            "total_duration_ms": sum(s.duration_ms or 0 for s in completed_spans),
            "spans": [
                {
                    "operation": s.operation_name,
                    "duration_ms": s.duration_ms,
                    "status": s.status,
                }
                for s in completed_spans
            ],
        }

    def __str__(self) -> str:
        return f"TraceContext(trace_id={self.trace_id[:8]}..., session={self.session_id})"

    def __repr__(self) -> str:
        return (
            f"TraceContext(trace_id={self.trace_id!r}, "
            f"session_id={self.session_id!r}, "
            f"spans={len(self.spans)})"
        )
