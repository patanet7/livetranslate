"""
Custom Exception Classes

Defines a hierarchy of exceptions for consistent error handling.

Exception Hierarchy:
    APIError (base)
    ├── ValidationError (400/422)
    ├── AuthenticationError (401)
    ├── AuthorizationError (403)
    ├── NotFoundError (404)
    ├── RateLimitError (429)
    ├── ServiceUnavailableError (503)
    ├── DatabaseError (500)
    └── ExternalServiceError (502)
"""

from typing import Any, Dict, Optional
from datetime import datetime, timezone


class APIError(Exception):
    """
    Base exception for all API errors.

    Provides consistent error structure with:
    - HTTP status code
    - Error code (machine-readable)
    - Human-readable message
    - Optional details for debugging
    - Timestamp

    Usage:
        raise APIError(
            status_code=400,
            code="INVALID_REQUEST",
            message="Request body is malformed",
            details={"field": "text", "issue": "required"}
        )
    """

    def __init__(
        self,
        status_code: int = 500,
        code: str = "INTERNAL_ERROR",
        message: str = "An unexpected error occurred",
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}
        self.original_exception = original_exception
        self.timestamp = datetime.now(timezone.utc)
        super().__init__(message)

    def to_dict(self, include_timestamp: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        # Ensure code is a string (handles Enum values)
        code_value = str(self.code.value) if hasattr(self.code, 'value') else str(self.code)

        result = {
            "code": code_value,
            "message": str(self.message),
        }
        if self.details:
            result["details"] = self.details
        if include_timestamp and self.timestamp:
            result["timestamp"] = self.timestamp.isoformat()
        return result

    @classmethod
    def from_exception(cls, exc: Exception, status_code: int = 500) -> "APIError":
        """Create APIError from any exception."""
        if isinstance(exc, APIError):
            return exc
        return cls(
            status_code=status_code,
            code="INTERNAL_ERROR",
            message=str(exc) or "An unexpected error occurred",
            original_exception=exc,
        )

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        return f"APIError(status_code={self.status_code}, code={self.code!r}, message={self.message!r})"


class ValidationError(APIError):
    """
    Validation error for invalid input data.

    HTTP Status: 400 (Bad Request) or 422 (Unprocessable Entity)

    Usage:
        raise ValidationError("text", "Cannot be empty")
        raise ValidationError("email", "Invalid email format", value="not-an-email")
    """

    def __init__(
        self,
        field: str,
        message: str,
        value: Any = None,
        status_code: int = 422,
    ):
        details = {"field": field}
        if value is not None:
            details["received_value"] = str(value)[:100]  # Truncate long values

        super().__init__(
            status_code=status_code,
            code="VALIDATION_ERROR",
            message=f"Validation failed for '{field}': {message}",
            details=details,
        )
        self.field = field


class NotFoundError(APIError):
    """
    Resource not found error.

    HTTP Status: 404 (Not Found)

    Usage:
        raise NotFoundError("Session", "abc123")
        raise NotFoundError("Glossary", glossary_id, details={"searched_in": "active"})
    """

    def __init__(
        self,
        resource_type: str,
        resource_id: Any,
        details: Optional[Dict[str, Any]] = None,
    ):
        error_details = {
            "resource_type": resource_type,
            "resource_id": str(resource_id),
        }
        if details:
            error_details.update(details)

        super().__init__(
            status_code=404,
            code="NOT_FOUND",
            message=f"{resource_type} '{resource_id}' not found",
            details=error_details,
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class AuthenticationError(APIError):
    """
    Authentication error (missing or invalid credentials).

    HTTP Status: 401 (Unauthorized)

    Usage:
        raise AuthenticationError("Invalid API key")
        raise AuthenticationError("Token expired", details={"expired_at": "..."})
    """

    def __init__(
        self,
        message: str = "Authentication required",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=401,
            code="AUTHENTICATION_ERROR",
            message=message,
            details=details,
        )


class AuthorizationError(APIError):
    """
    Authorization error (insufficient permissions).

    HTTP Status: 403 (Forbidden)

    Usage:
        raise AuthorizationError("Cannot access this session")
        raise AuthorizationError("Admin access required", required_role="admin")
    """

    def __init__(
        self,
        message: str = "Access denied",
        required_role: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        error_details = details or {}
        if required_role:
            error_details["required_role"] = required_role

        super().__init__(
            status_code=403,
            code="AUTHORIZATION_ERROR",
            message=message,
            details=error_details if error_details else None,
        )


class RateLimitError(APIError):
    """
    Rate limit exceeded error.

    HTTP Status: 429 (Too Many Requests)

    Usage:
        raise RateLimitError(retry_after=60)
        raise RateLimitError(retry_after=30, limit=100, window=60)
    """

    def __init__(
        self,
        retry_after: int = 60,
        limit: Optional[int] = None,
        window: Optional[int] = None,
    ):
        details = {"retry_after_seconds": retry_after}
        if limit is not None:
            details["limit"] = limit
        if window is not None:
            details["window_seconds"] = window

        super().__init__(
            status_code=429,
            code="RATE_LIMIT_EXCEEDED",
            message=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            details=details,
        )
        self.retry_after = retry_after


class ServiceUnavailableError(APIError):
    """
    Service temporarily unavailable.

    HTTP Status: 503 (Service Unavailable)

    Usage:
        raise ServiceUnavailableError("Translation service")
        raise ServiceUnavailableError("Database", retry_after=30)
    """

    def __init__(
        self,
        service_name: str,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        error_details = {"service": service_name}
        if retry_after is not None:
            error_details["retry_after_seconds"] = retry_after
        if details:
            error_details.update(details)

        super().__init__(
            status_code=503,
            code="SERVICE_UNAVAILABLE",
            message=f"{service_name} is temporarily unavailable",
            details=error_details,
        )
        self.service_name = service_name
        self.retry_after = retry_after


class DatabaseError(APIError):
    """
    Database operation error.

    HTTP Status: 500 (Internal Server Error)

    Usage:
        raise DatabaseError("Connection failed")
        raise DatabaseError("Query timeout", operation="SELECT")
    """

    def __init__(
        self,
        message: str = "Database operation failed",
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        error_details = details or {}
        if operation:
            error_details["operation"] = operation

        super().__init__(
            status_code=500,
            code="DATABASE_ERROR",
            message=message,
            details=error_details if error_details else None,
        )


class ExternalServiceError(APIError):
    """
    External service (API) error.

    HTTP Status: 502 (Bad Gateway)

    Usage:
        raise ExternalServiceError("Fireflies API", "Invalid response")
        raise ExternalServiceError("Translation API", "Timeout", status=504)
    """

    def __init__(
        self,
        service_name: str,
        message: str,
        upstream_status: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        error_details = {"service": service_name}
        if upstream_status is not None:
            error_details["upstream_status"] = upstream_status
        if details:
            error_details.update(details)

        super().__init__(
            status_code=502,
            code="EXTERNAL_SERVICE_ERROR",
            message=f"{service_name}: {message}",
            details=error_details,
        )
        self.service_name = service_name
        self.upstream_status = upstream_status
