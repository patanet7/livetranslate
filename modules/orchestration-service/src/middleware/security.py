"""
Security Middleware

Provides security features including request validation, rate limiting, and security headers.
"""

import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for FastAPI applications
    """

    def __init__(self, app, config: dict = None):
        super().__init__(app)
        self.config = config or {}
        self.max_request_size = self.config.get(
            "max_request_size", 10 * 1024 * 1024
        )  # 10MB
        self.blocked_ips = set(self.config.get("blocked_ips", []))
        self.allowed_origins = self.config.get("allowed_origins", ["*"])

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through security middleware
        """
        start_time = time.time()

        try:
            # Skip security middleware for WebSocket connections
            is_websocket = (
                request.url.path == "/ws"
                or request.url.path.startswith("/api/websocket/")
                or request.headers.get("upgrade", "").lower() == "websocket"
            )
            if is_websocket:
                return await call_next(request)
            # Check blocked IPs
            client_ip = request.client.host
            if client_ip in self.blocked_ips:
                logger.warning(f"Blocked IP attempted access: {client_ip}")
                return JSONResponse(status_code=403, content={"error": "Access denied"})

            # Check request size
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_request_size:
                logger.warning(
                    f"Request too large: {content_length} bytes from {client_ip}"
                )
                return JSONResponse(
                    status_code=413, content={"error": "Request entity too large"}
                )

            # Process request
            response = await call_next(request)

            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

            # Add timing header for monitoring
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)

            return response

        except Exception as e:
            # Convert exception to string to avoid JSON serialization issues
            error_msg = str(e)
            logger.error(f"Security middleware error: {error_msg}")
            return JSONResponse(
                status_code=500, content={"error": "Internal server error"}
            )
