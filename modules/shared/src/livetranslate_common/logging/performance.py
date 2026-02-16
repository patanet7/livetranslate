"""Performance measurement logging."""

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import structlog


@contextmanager
def log_performance(
    logger: structlog.stdlib.BoundLogger, operation: str, **extra: Any
) -> Generator[None, None, None]:
    """Context manager that logs operation duration."""
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.error("operation_failed", operation=operation, duration_ms=elapsed_ms, **extra)
        raise
    else:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "operation_completed",
            operation=operation,
            duration_ms=elapsed_ms,
            **extra,
        )
