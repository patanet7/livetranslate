"""FastAPI exception handlers for LiveTranslate errors."""

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from livetranslate_common.errors.exceptions import LiveTranslateError

logger = structlog.get_logger()


def register_error_handlers(app: FastAPI) -> None:
    """Register exception handlers on a FastAPI application.

    Args:
        app: The FastAPI application instance to register handlers on.
    """

    @app.exception_handler(LiveTranslateError)
    async def handle_livetranslate_error(request: Request, exc: LiveTranslateError) -> JSONResponse:
        logger.error(
            "application_error",
            error_code=exc.error_code,
            message=str(exc),
            path=request.url.path,
            **exc.context,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_code": exc.error_code,
                "message": str(exc),
                **exc.context,
            },
        )
