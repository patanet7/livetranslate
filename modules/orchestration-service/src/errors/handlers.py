"""
Error Response Handlers

Provides centralized error response formatting and FastAPI exception handlers.

Usage:
    from errors import register_exception_handlers, error_response

    # Register handlers on FastAPI app
    register_exception_handlers(app)

    # Create error response manually
    return error_response(NotFoundError("Session", "abc123"))
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional, Union
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError as PydanticValidationError

from .exceptions import APIError, ValidationError, NotFoundError

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    """
    Standard error codes for consistent API responses.

    Use these codes for machine-readable error identification.
    """

    # Client errors (4xx)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    BAD_REQUEST = "BAD_REQUEST"
    CONFLICT = "CONFLICT"

    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"


def error_response(
    error: Union[APIError, Exception],
    request_id: Optional[str] = None,
    include_timestamp: bool = True,
) -> JSONResponse:
    """
    Create a standardized JSON error response.

    Args:
        error: APIError instance or any exception
        request_id: Optional request ID for tracing
        include_timestamp: Whether to include timestamp

    Returns:
        JSONResponse with consistent error structure

    Response Format:
        {
            "error": {
                "code": "ERROR_CODE",
                "message": "Human-readable message",
                "details": {...},  # Optional
                "timestamp": "2026-01-16T14:30:00Z",
                "request_id": "req_xyz"  # If provided
            }
        }
    """
    # Convert to APIError if needed
    if not isinstance(error, APIError):
        error = APIError.from_exception(error)

    # Build response body
    error_body = error.to_dict(include_timestamp=include_timestamp)

    if request_id:
        error_body["request_id"] = request_id

    # Add headers for specific error types
    headers = {}
    if hasattr(error, "retry_after") and error.retry_after:
        headers["Retry-After"] = str(error.retry_after)

    return JSONResponse(
        status_code=error.status_code,
        content={"error": error_body},
        headers=headers if headers else None,
    )


def _format_validation_errors(errors: list) -> Dict[str, Any]:
    """
    Format Pydantic/FastAPI validation errors into readable structure.

    Transforms:
        [{"loc": ["body", "text"], "msg": "field required", "type": "missing"}]
    Into:
        {"fields": {"text": "field required"}, "count": 1}
    """
    formatted_fields = {}

    for error in errors:
        # Get field path (skip 'body' prefix)
        loc = error.get("loc", [])
        field_path = ".".join(str(part) for part in loc if part != "body")
        if not field_path:
            field_path = "request"

        # Get error message
        msg = error.get("msg", "Invalid value")

        # Combine multiple errors for same field
        if field_path in formatted_fields:
            formatted_fields[field_path] += f"; {msg}"
        else:
            formatted_fields[field_path] = msg

    return {
        "fields": formatted_fields,
        "count": len(errors),
    }


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handler for APIError exceptions."""
    request_id = getattr(request.state, "request_id", None)

    # Log the error
    if exc.status_code >= 500:
        logger.error(
            f"[{request_id}] {exc.code}: {exc.message}",
            extra={"error_details": exc.details},
        )
    else:
        logger.warning(f"[{request_id}] {exc.code}: {exc.message}")

    return error_response(exc, request_id=request_id)


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handler for FastAPI request validation errors."""
    request_id = getattr(request.state, "request_id", None)

    # Format validation errors
    details = _format_validation_errors(exc.errors())

    # Create readable message
    field_count = details["count"]
    if field_count == 1:
        field_name = list(details["fields"].keys())[0]
        message = f"Validation failed: {field_name} - {details['fields'][field_name]}"
    else:
        message = f"Validation failed for {field_count} fields"

    error = ValidationError.__new__(ValidationError)
    error.status_code = 422
    error.code = ErrorCode.VALIDATION_ERROR
    error.message = message
    error.details = details
    error.timestamp = datetime.now(timezone.utc)

    logger.info(f"[{request_id}] Validation error: {message}")

    return error_response(error, request_id=request_id)


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handler for Starlette/FastAPI HTTP exceptions."""
    request_id = getattr(request.state, "request_id", None)

    # Map HTTP status to error code
    code_map = {
        400: ErrorCode.BAD_REQUEST,
        401: ErrorCode.AUTHENTICATION_ERROR,
        403: ErrorCode.AUTHORIZATION_ERROR,
        404: ErrorCode.NOT_FOUND,
        409: ErrorCode.CONFLICT,
        429: ErrorCode.RATE_LIMIT_EXCEEDED,
        500: ErrorCode.INTERNAL_ERROR,
        502: ErrorCode.EXTERNAL_SERVICE_ERROR,
        503: ErrorCode.SERVICE_UNAVAILABLE,
    }

    error = APIError(
        status_code=exc.status_code,
        code=code_map.get(exc.status_code, ErrorCode.INTERNAL_ERROR),
        message=exc.detail if isinstance(exc.detail, str) else str(exc.detail),
    )

    return error_response(error, request_id=request_id)


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler for unhandled exceptions."""
    request_id = getattr(request.state, "request_id", None)

    # Log the full exception for debugging
    logger.exception(
        f"[{request_id}] Unhandled exception: {type(exc).__name__}: {exc}"
    )

    # Don't expose internal details in production
    error = APIError(
        status_code=500,
        code=ErrorCode.INTERNAL_ERROR,
        message="An unexpected error occurred",
        details={"type": type(exc).__name__},
    )

    return error_response(error, request_id=request_id)


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all exception handlers on a FastAPI application.

    This should be called during app initialization:

        from errors import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)

    Handlers registered:
        - APIError and subclasses -> api_error_handler
        - RequestValidationError -> validation_error_handler
        - HTTPException -> http_exception_handler
        - Exception (catch-all) -> generic_exception_handler
    """
    # Custom API errors
    app.add_exception_handler(APIError, api_error_handler)

    # FastAPI validation errors (422)
    app.add_exception_handler(RequestValidationError, validation_error_handler)

    # Starlette HTTP exceptions
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    # Catch-all for unhandled exceptions
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("Registered centralized exception handlers")
