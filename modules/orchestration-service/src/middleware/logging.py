"""
Logging Middleware

Provides structured logging for all HTTP requests and responses.
"""

import logging
import time
import json
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import uuid

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logging middleware for FastAPI applications
    """

    def __init__(self, app, config: dict = None):
        super().__init__(app)
        self.config = config or {}
        self.log_level = self.config.get("log_level", "INFO")
        self.log_requests = self.config.get("log_requests", True)
        self.log_responses = self.config.get("log_responses", True)
        self.log_body = self.config.get("log_body", False)
        self.max_body_size = self.config.get("max_body_size", 1024)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log request and response information
        """
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]

        # Start timing
        start_time = time.time()

        # Log request
        if self.log_requests:
            await self._log_request(request, request_id)

        # Process request
        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log response
            if self.log_responses:
                await self._log_response(response, request_id, process_time)

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            logger.error(
                f"Request {request_id} failed: {str(e)}",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "process_time": process_time,
                    "method": request.method,
                    "url": str(request.url),
                },
            )
            raise

    async def _log_request(self, request: Request, request_id: str):
        """
        Log incoming request
        """
        try:
            # Basic request info
            log_data = {
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent", ""),
                "timestamp": time.time(),
            }

            # Log request body if enabled
            if self.log_body:
                try:
                    body = await request.body()
                    if body and len(body) <= self.max_body_size:
                        # Try to decode as JSON
                        try:
                            log_data["body"] = json.loads(body.decode("utf-8"))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            log_data["body"] = body.decode("utf-8", errors="replace")[
                                : self.max_body_size
                            ]
                    elif body:
                        log_data["body_size"] = len(body)
                        log_data["body"] = f"<body too large: {len(body)} bytes>"
                except Exception as e:
                    log_data["body_error"] = str(e)

            logger.info(
                f"Request {request_id}: {request.method} {request.url.path}",
                extra=log_data,
            )

        except Exception as e:
            logger.error(f"Failed to log request {request_id}: {e}")

    async def _log_response(
        self, response: Response, request_id: str, process_time: float
    ):
        """
        Log outgoing response
        """
        try:
            log_data = {
                "request_id": request_id,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "process_time": process_time,
                "timestamp": time.time(),
            }

            # Log response body if enabled and it's a reasonable size
            if self.log_body and hasattr(response, "body"):
                try:
                    if response.body and len(response.body) <= self.max_body_size:
                        try:
                            log_data["body"] = json.loads(response.body.decode("utf-8"))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            log_data["body"] = response.body.decode(
                                "utf-8", errors="replace"
                            )[: self.max_body_size]
                    elif response.body:
                        log_data["body_size"] = len(response.body)
                        log_data["body"] = (
                            f"<body too large: {len(response.body)} bytes>"
                        )
                except Exception as e:
                    log_data["body_error"] = str(e)

            # Determine log level based on status code
            if response.status_code >= 500:
                log_level = logging.ERROR
            elif response.status_code >= 400:
                log_level = logging.WARNING
            else:
                log_level = logging.INFO

            logger.log(
                log_level,
                f"Response {request_id}: {response.status_code} ({process_time:.3f}s)",
                extra=log_data,
            )

        except Exception as e:
            logger.error(f"Failed to log response {request_id}: {e}")

    def configure_logging(self):
        """
        Configure logging format and handlers
        """
        # Create formatter for structured logging
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level.upper()))

        # Add console handler if not already present
        if not root_logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
