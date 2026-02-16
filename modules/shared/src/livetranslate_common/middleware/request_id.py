"""Request ID middleware -- generates or propagates X-Request-ID."""

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate or propagate a request ID and bind it to structlog context.

    If the incoming request contains an ``X-Request-ID`` header, the value is
    reused. Otherwise a new UUID4 is generated. The ID is bound to structlog
    context vars so all downstream log entries include it, and it is echoed
    back in the response ``X-Request-ID`` header.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
