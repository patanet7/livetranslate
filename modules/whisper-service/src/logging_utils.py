#!/usr/bin/env python3
"""
Structured Logging Utilities

Provides performance-focused logging with proper log levels and structured output.

Usage:
    from logging_utils import PerformanceLogger, log_audio_stats

    perf_logger = PerformanceLogger("session_manager")
    with perf_logger.measure("process_chunk"):
        result = process_audio(chunk)

    log_audio_stats(audio, logger, level=logging.DEBUG)
"""

import logging
import time
from contextlib import contextmanager
from typing import Any

import numpy as np


class PerformanceLogger:
    """
    Performance measurement logger with proper log levels.

    Logs timing information at DEBUG level to avoid cluttering production logs.
    """

    def __init__(self, component_name: str, logger: logging.Logger | None = None):
        """
        Initialize performance logger.

        Args:
            component_name: Name of component being measured
            logger: Logger instance (defaults to root logger)
        """
        self.component_name = component_name
        self.logger = logger or logging.getLogger(__name__)
        self.metrics: dict[str, list] = {}

    @contextmanager
    def measure(self, operation: str):
        """
        Context manager for measuring operation timing.

        Args:
            operation: Name of operation being measured

        Example:
            with perf_logger.measure("vad_detection"):
                vad_result = vad.check_speech(audio)
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = (time.time() - start_time) * 1000  # Convert to ms

            # Store metric
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(elapsed)

            # Log at DEBUG level (high-frequency operations shouldn't clutter INFO logs)
            self.logger.debug(f"[PERF] {self.component_name}.{operation}: {elapsed:.2f}ms")

    def log_summary(self, level: int = logging.INFO):
        """
        Log performance summary statistics.

        Args:
            level: Log level for summary (default: INFO)
        """
        if not self.metrics:
            return

        self.logger.log(level, f"=== Performance Summary: {self.component_name} ===")

        for operation, timings in self.metrics.items():
            if not timings:
                continue

            count = len(timings)
            avg = sum(timings) / count
            min_time = min(timings)
            max_time = max(timings)

            self.logger.log(
                level,
                f"  {operation}: count={count}, avg={avg:.2f}ms, "
                f"min={min_time:.2f}ms, max={max_time:.2f}ms",
            )

    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()


def log_audio_stats(
    audio: np.ndarray, logger: logging.Logger, prefix: str = "Audio", level: int = logging.DEBUG
):
    """
    Log audio statistics at specified log level.

    Args:
        audio: Audio numpy array
        logger: Logger instance
        prefix: Prefix for log message
        level: Log level (default: DEBUG)

    Example:
        log_audio_stats(chunk, logger, prefix="VAD Input", level=logging.DEBUG)
    """
    if len(audio) == 0:
        logger.log(level, f"{prefix}: EMPTY")
        return

    rms = float(np.sqrt(np.mean(audio**2)))
    max_amp = float(np.max(np.abs(audio)))
    duration = len(audio) / 16000  # Assume 16kHz

    logger.log(
        level,
        f"{prefix}: {len(audio)} samples, {duration:.2f}s, " f"RMS={rms:.6f}, Max={max_amp:.6f}",
    )


def log_vad_event(
    event: dict[str, float] | None, logger: logging.Logger, level: int = logging.INFO
):
    """
    Log VAD event with proper formatting.

    Args:
        event: VAD event dict with 'start' and/or 'end' keys
        logger: Logger instance
        level: Log level (default: INFO)

    Example:
        log_vad_event(vad_result, logger, level=logging.INFO)
    """
    if event is None:
        # No event - log at DEBUG only
        logger.debug("VAD: No state change")
        return

    parts = []
    if "start" in event:
        parts.append(f"START @ {event['start']:.2f}s")
    if "end" in event:
        parts.append(f"END @ {event['end']:.2f}s")

    logger.log(level, f"🎤 VAD: {' + '.join(parts)}")


def log_language_switch(
    from_lang: str,
    to_lang: str,
    margin: float,
    frames: int,
    duration_ms: float,
    logger: logging.Logger,
    level: int = logging.INFO,
):
    """
    Log language switch event with structured information.

    Args:
        from_lang: Source language
        to_lang: Target language
        margin: Confidence margin
        frames: Number of dwell frames
        duration_ms: Dwell duration in milliseconds
        logger: Logger instance
        level: Log level (default: INFO)

    Example:
        log_language_switch('en', 'zh', 0.35, 8, 320.0, logger)
    """
    logger.log(
        level,
        f"🔄 Language Switch: {from_lang} → {to_lang} | "
        f"margin={margin:.3f}, frames={frames}, duration={duration_ms:.0f}ms",
    )


def log_session_event(
    event_type: str,
    language: str,
    details: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
):
    """
    Log session lifecycle events.

    Args:
        event_type: Event type ('create', 'finish', 'switch')
        language: Language code
        details: Optional event details
        logger: Logger instance
        level: Log level (default: INFO)

    Example:
        log_session_event('create', 'en', {'samples': 16000}, logger)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    icons = {"create": "🆕", "finish": "⏹️", "switch": "🔄"}
    icon = icons.get(event_type, "📌")

    msg = f"{icon} Session {event_type}: {language}"
    if details:
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        msg += f" ({detail_str})"

    logger.log(level, msg)


class MetricsCollector:
    """
    Collect and aggregate metrics over time.

    Useful for performance monitoring and debugging.
    """

    def __init__(self, name: str):
        """
        Initialize metrics collector.

        Args:
            name: Name of metrics collector
        """
        self.name = name
        self.counters: dict[str, int] = {}
        self.timers: dict[str, list] = {}
        self.gauges: dict[str, float] = {}

    def increment(self, counter: str, value: int = 1):
        """Increment a counter"""
        self.counters[counter] = self.counters.get(counter, 0) + value

    def record_time(self, timer: str, value_ms: float):
        """Record a timing value"""
        if timer not in self.timers:
            self.timers[timer] = []
        self.timers[timer].append(value_ms)

    def set_gauge(self, gauge: str, value: float):
        """Set a gauge value"""
        self.gauges[gauge] = value

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary"""
        summary = {
            "name": self.name,
            "counters": self.counters.copy(),
            "gauges": self.gauges.copy(),
            "timers": {},
        }

        for timer, values in self.timers.items():
            if values:
                summary["timers"][timer] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }

        return summary

    def reset(self):
        """Reset all metrics"""
        self.counters.clear()
        self.timers.clear()
        self.gauges.clear()


# Pre-configured loggers for common components
def get_component_logger(component: str, level: int | None = None) -> logging.Logger:
    """
    Get a logger for a specific component with consistent naming.

    Args:
        component: Component name (e.g., 'vad', 'lid', 'session_manager')
        level: Optional log level override

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"whisper_service.{component}")

    if level is not None:
        logger.setLevel(level)

    return logger
