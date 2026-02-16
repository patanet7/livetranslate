"""
Security Middleware

Provides security features including:
- IP blocking
- Request size limits
- Security headers (HSTS, CSP, etc.)
- Request timing

Uses centralized error handling from errors module.
"""

import logging
import time
from collections.abc import Callable

from errors import (
    APIError,
    AuthorizationError,
    ValidationError,
    error_response,
)
from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for FastAPI applications.

    Features:
    - IP blocking
    - Request size validation
    - Security headers injection
    - Request timing metrics

    Configuration:
        config = {
            "max_request_size": 10 * 1024 * 1024,  # 10MB
            "blocked_ips": ["1.2.3.4"],
            "allowed_origins": ["*"],
        }
    """

    def __init__(self, app, config: dict | None = None):
        super().__init__(app)
        self.config = config or {}
        self.max_request_size = self.config.get("max_request_size", 10 * 1024 * 1024)  # 10MB
        self.blocked_ips = set(self.config.get("blocked_ips", []))
        self.allowed_origins = self.config.get("allowed_origins", ["*"])

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through security middleware."""
        start_time = time.time()
        request_id = getattr(request.state, "request_id", None)

        try:
            # Skip middleware for WebSocket connections
            if self._is_websocket(request):
                return await call_next(request)

            # Security checks
            self._check_blocked_ip(request)
            self._check_request_size(request)

            # Process request
            response = await call_next(request)

            # Add security headers
            self._add_security_headers(response)

            # Add timing header
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = f"{process_time:.4f}"

            return response

        except HTTPException:
            # Let FastAPI handle HTTPExceptions properly
            raise

        except APIError as e:
            # Use centralized error response
            logger.warning(f"[{request_id}] Security check failed: {e.message}")
            return error_response(e, request_id=request_id)

        except Exception as e:
            # Log unexpected errors but don't expose details
            logger.exception(f"[{request_id}] Security middleware error: {e}")

            # Use centralized error handling
            api_error = APIError.from_exception(e)
            return error_response(api_error, request_id=request_id)

    def _is_websocket(self, request: Request) -> bool:
        """Check if request is a WebSocket connection."""
        return (
            request.url.path == "/ws"
            or request.url.path.startswith("/api/websocket/")
            or request.headers.get("upgrade", "").lower() == "websocket"
        )

    def _check_blocked_ip(self, request: Request) -> None:
        """Check if client IP is blocked."""
        client_ip = request.client.host if request.client else "unknown"
        if client_ip in self.blocked_ips:
            logger.warning(f"Blocked IP attempted access: {client_ip}")
            raise AuthorizationError(
                message="Access denied",
                details={"reason": "IP blocked"},
            )

    def _check_request_size(self, request: Request) -> None:
        """Check if request exceeds size limit."""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_request_size:
                    client_ip = request.client.host if request.client else "unknown"
                    logger.warning(f"Request too large: {size} bytes from {client_ip}")
                    raise ValidationError(
                        field="content-length",
                        message=f"Request body too large ({size} bytes). Maximum: {self.max_request_size} bytes",
                        status_code=413,
                    )
            except ValueError:
                pass  # Invalid content-length header, let FastAPI handle it

    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
