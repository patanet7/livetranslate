"""Ring buffer for capturing recent log entries for dashboard display."""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any


@dataclass
class LogEntry:
    """A captured log entry."""

    timestamp: str
    level: str
    event: str
    service: str
    filename: str | None = None
    func_name: str | None = None
    lineno: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class LogBuffer:
    """Thread-safe ring buffer for recent log entries."""

    def __init__(self, max_size: int = 500) -> None:
        self._buffer: deque[LogEntry] = deque(maxlen=max_size)
        self._lock = Lock()

    def append(self, entry: LogEntry) -> None:
        """Add a log entry to the buffer."""
        with self._lock:
            self._buffer.append(entry)

    def get_recent(self, limit: int = 100, level: str | None = None) -> list[LogEntry]:
        """Get recent log entries, optionally filtered by level."""
        with self._lock:
            entries = list(self._buffer)

        # Filter by level if specified
        if level:
            level_upper = level.upper()
            entries = [e for e in entries if e.level.upper() == level_upper]

        # Return most recent first
        return list(reversed(entries[-limit:]))

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


# Global buffer instance
_log_buffer: LogBuffer | None = None


def get_log_buffer() -> LogBuffer:
    """Get or create the global log buffer."""
    global _log_buffer
    if _log_buffer is None:
        _log_buffer = LogBuffer()
    return _log_buffer


def buffer_processor(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor that captures log entries to the ring buffer."""
    buf = get_log_buffer()

    # Extract standard fields
    entry = LogEntry(
        timestamp=event_dict.get("timestamp", datetime.now(UTC).isoformat()),
        level=event_dict.get("level", "info"),
        event=event_dict.get("event", ""),
        service=event_dict.get("service", "unknown"),
        filename=event_dict.get("filename"),
        func_name=event_dict.get("func_name"),
        lineno=event_dict.get("lineno"),
    )

    # Capture extra fields (excluding standard ones)
    skip_keys = {
        "timestamp",
        "level",
        "event",
        "service",
        "filename",
        "func_name",
        "lineno",
    }
    entry.extra = {k: v for k, v in event_dict.items() if k not in skip_keys}

    buf.append(entry)
    return event_dict
