"""
LLM Client Shared Models

Shared data types used across all LLM client implementations:
- CircuitBreaker: Resilience pattern for service calls
- PromptTranslationResult: Result from LLM prompt completion
- StreamChunk: Streaming chunk from LLM response
"""

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitBreaker:
    """
    Simple circuit breaker for service calls.

    States:
    - CLOSED: Requests pass through normally
    - OPEN: Requests fail immediately (fast-fail)
    - HALF_OPEN: One test request allowed through

    Opens after `failure_threshold` consecutive failures.
    Resets to HALF_OPEN after `recovery_timeout` seconds.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._success_count = 0

    @property
    def state(self) -> str:
        if self._state == self.OPEN:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = self.HALF_OPEN
                logger.info("Circuit breaker: OPEN -> HALF_OPEN (recovery timeout elapsed)")
        return self._state

    @property
    def is_available(self) -> bool:
        return self.state != self.OPEN

    def record_success(self) -> None:
        if self._state == self.HALF_OPEN:
            logger.info("Circuit breaker: HALF_OPEN -> CLOSED (successful request)")
        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count += 1

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            if self._state != self.OPEN:
                logger.warning(
                    f"Circuit breaker: {self._state} -> OPEN "
                    f"({self._failure_count} consecutive failures)"
                )
            self._state = self.OPEN


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PromptTranslationResult:
    """Result from LLM prompt completion."""

    text: str
    processing_time_ms: float
    backend_used: str
    model_used: str
    tokens_used: int | None = None


@dataclass
class StreamChunk:
    """Streaming chunk from LLM response."""

    chunk: str | None = None
    done: bool = False
    processing_time_ms: float | None = None
    backend_used: str | None = None
    model_used: str | None = None
    error: str | None = None
