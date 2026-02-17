"""Request/response logging middleware."""

import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger()

_SKIP_PATHS = frozenset({"/health", "/healthz", "/metrics", "/ready", "/readyz"})


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request and response with structured context.

    Health-check paths (``/health``, ``/healthz``, ``/metrics``, ``/ready``,
    ``/readyz``) are silently passed through to avoid log noise.

    For non-skipped paths the middleware emits:
    * ``request_started`` at the beginning of every request.
    * ``request_completed`` when the response is produced (with ``duration_ms``).
    * ``request_failed`` if an unhandled exception propagates (with ``duration_ms``).
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if path in _SKIP_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        logger.info(
            "request_started",
            method=request.method,
            path=path,
            client_ip=request.client.host if request.client else None,
        )
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error(
                "request_failed",
                method=request.method,
                path=path,
                duration_ms=duration_ms,
                exc_info=True,
            )
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_method = (
            logger.error
            if response.status_code >= 500
            else logger.warning
            if response.status_code >= 400
            else logger.info
        )
        log_method(
            "request_completed",
            method=request.method,
            path=path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response
