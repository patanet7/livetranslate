"""
Centralized Error Handling Module

Provides consistent, well-formed error responses across the entire API.

Usage:
    from errors import (
        APIError,
        ValidationError,
        NotFoundError,
        ServiceUnavailableError,
        error_response,
    )

    # Raise specific errors
    raise NotFoundError("Session", session_id)
    raise ValidationError("text", "Cannot be empty")

    # In exception handlers
    return error_response(APIError.from_exception(e))

Error Response Format:
    {
        "error": {
            "code": "NOT_FOUND",
            "message": "Session 'abc123' not found",
            "details": {...},  # Optional additional context
            "timestamp": "2026-01-16T14:30:00Z",
            "request_id": "req_xyz789"  # If available
        }
    }
"""

from .exceptions import (
    APIError,
    AuthenticationError,
    AuthorizationError,
    DatabaseError,
    ExternalServiceError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    ValidationError,
)
from .handlers import (
    ErrorCode,
    error_response,
    register_exception_handlers,
)

__all__ = [
    # Exception classes
    "APIError",
    "AuthenticationError",
    "AuthorizationError",
    "DatabaseError",
    "ErrorCode",
    "ExternalServiceError",
    "NotFoundError",
    "RateLimitError",
    "ServiceUnavailableError",
    "ValidationError",
    # Handlers
    "error_response",
    "register_exception_handlers",
]
