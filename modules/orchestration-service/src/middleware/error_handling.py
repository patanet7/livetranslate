"""
Error Handling Middleware

Provides comprehensive error handling and recovery for FastAPI applications.
"""

import time
import traceback
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request, Response
from livetranslate_common.logging import get_logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

logger = get_logger()


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Error handling middleware for FastAPI applications
    """

    def __init__(self, app, config: dict | None = None):
        super().__init__(app)
        self.config = config or {}
        self.include_traceback = self.config.get("include_traceback", False)
        self.log_errors = self.config.get("log_errors", True)
        self.error_response_format = self.config.get("error_response_format", "json")

        # Error counters for monitoring
        self.error_counts = {}
        self.last_reset = time.time()
        self.reset_interval = 3600  # Reset hourly

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Handle errors and provide consistent error responses
        """
        try:
            response = await call_next(request)
            return response

        except HTTPException as e:
            # FastAPI HTTPException - pass through with proper formatting
            return await self._handle_http_exception(request, e)

        except Exception as e:
            # Unexpected error - log and provide generic response
            return await self._handle_unexpected_error(request, e)

    async def _handle_http_exception(self, request: Request, exc: HTTPException) -> Response:
        """
        Handle FastAPI HTTPException with consistent formatting
        """
        # Log HTTP exceptions based on status code
        if exc.status_code >= 500:
            logger.error(
                f"HTTP {exc.status_code} error on {request.method} {request.url}: {exc.detail}"
            )
        elif exc.status_code >= 400:
            logger.warning(
                f"HTTP {exc.status_code} error on {request.method} {request.url}: {exc.detail}"
            )

        # Update error counts
        self._update_error_count(exc.status_code)

        # Format error response
        error_response = {
            "error": {
                "type": "http_exception",
                "status_code": exc.status_code,
                "message": exc.detail,
                "timestamp": time.time(),
                "path": str(request.url),
                "method": request.method,
            }
        }

        # Add correlation ID if available
        if hasattr(request.state, "correlation_id"):
            error_response["error"]["correlation_id"] = request.state.correlation_id

        return JSONResponse(status_code=exc.status_code, content=error_response)

    async def _handle_unexpected_error(self, request: Request, exc: Exception) -> Response:
        """
        Handle unexpected errors with proper logging and response
        """
        # Generate correlation ID for tracking
        correlation_id = str(time.time_ns())[-8:]

        # Log the error with full context
        if self.log_errors:
            logger.error(
                f"Unexpected error [{correlation_id}] on {request.method} {request.url}: {exc!s}",
                extra={
                    "correlation_id": correlation_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "method": request.method,
                    "url": str(request.url),
                    "client_ip": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent", ""),
                    "traceback": traceback.format_exc() if self.include_traceback else None,
                },
            )

        # Update error counts
        self._update_error_count(500)

        # Format error response
        error_response = {
            "error": {
                "type": "internal_server_error",
                "status_code": 500,
                "message": "An unexpected error occurred. Please try again later.",
                "correlation_id": correlation_id,
                "timestamp": time.time(),
                "path": str(request.url),
                "method": request.method,
            }
        }

        # Include traceback in development mode
        if self.include_traceback:
            error_response["error"]["traceback"] = traceback.format_exc()
            error_response["error"]["details"] = str(exc)

        return JSONResponse(status_code=HTTP_500_INTERNAL_SERVER_ERROR, content=error_response)

    def _update_error_count(self, status_code: int):
        """
        Update error counters for monitoring
        """
        # Reset counters if interval has passed
        current_time = time.time()
        if current_time - self.last_reset > self.reset_interval:
            self.error_counts = {}
            self.last_reset = current_time

        # Update count
        self.error_counts[status_code] = self.error_counts.get(status_code, 0) + 1

    def get_error_stats(self) -> dict[str, Any]:
        """
        Get current error statistics
        """
        return {
            "error_counts": self.error_counts.copy(),
            "last_reset": self.last_reset,
            "reset_interval": self.reset_interval,
            "total_errors": sum(self.error_counts.values()),
        }

    def reset_error_stats(self):
        """
        Reset error statistics
        """
        self.error_counts = {}
        self.last_reset = time.time()


class ErrorReporter:
    """
    Error reporting utility for integration with external services
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        self.webhook_url = self.config.get("webhook_url")
        self.email_alerts = self.config.get("email_alerts", False)
        self.min_level = self.config.get("min_level", "ERROR")

    async def report_error(self, error_data: dict[str, Any]):
        """
        Report error to external monitoring systems
        """
        if not self.enabled:
            return

        try:
            # Send to webhook if configured
            if self.webhook_url:
                await self._send_webhook(error_data)

            # Send email alert if configured
            if self.email_alerts:
                await self._send_email_alert(error_data)

        except Exception as e:
            logger.error(f"Failed to report error: {e}")

    async def _send_webhook(self, error_data: dict[str, Any]):
        """
        Send error to webhook URL
        """
        import aiohttp

        async with aiohttp.ClientSession() as session:
            await session.post(self.webhook_url, json=error_data, timeout=5)

    async def _send_email_alert(self, error_data: dict[str, Any]):
        """
        Send email alert (placeholder implementation)
        """
        # This would integrate with email service
        logger.info(f"Would send email alert for error: {error_data.get('correlation_id')}")


class CustomExceptionHandler:
    """
    Custom exception handlers for specific error types
    """

    def __init__(self):
        self.handlers = {}

    def register_handler(self, exception_type: type, handler: Callable):
        """
        Register custom handler for specific exception type
        """
        self.handlers[exception_type] = handler

    async def handle_exception(self, request: Request, exc: Exception) -> Response | None:
        """
        Handle exception with custom handler if available
        """
        exception_type = type(exc)

        if exception_type in self.handlers:
            try:
                return await self.handlers[exception_type](request, exc)
            except Exception as handler_error:
                logger.error(f"Custom exception handler failed: {handler_error}")
                return None

        return None


# Global exception handler instance
exception_handler = CustomExceptionHandler()


# Common custom handlers
async def handle_validation_error(request: Request, exc: Exception) -> Response:
    """
    Handle validation errors with detailed messages
    """
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "validation_error",
                "status_code": 422,
                "message": "Validation failed",
                "details": str(exc),
                "timestamp": time.time(),
                "path": str(request.url),
                "method": request.method,
            }
        },
    )


async def handle_database_error(request: Request, exc: Exception) -> Response:
    """
    Handle database connection errors
    """
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "type": "database_error",
                "status_code": 503,
                "message": "Database temporarily unavailable",
                "timestamp": time.time(),
                "path": str(request.url),
                "method": request.method,
            }
        },
    )


async def handle_rate_limit_error(request: Request, exc: Exception) -> Response:
    """
    Handle rate limiting errors
    """
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "type": "rate_limit_error",
                "status_code": 429,
                "message": "Too many requests. Please try again later.",
                "timestamp": time.time(),
                "path": str(request.url),
                "method": request.method,
                "retry_after": 60,
            }
        },
    )


# Register common handlers
exception_handler.register_handler(ValueError, handle_validation_error)
exception_handler.register_handler(ConnectionError, handle_database_error)
